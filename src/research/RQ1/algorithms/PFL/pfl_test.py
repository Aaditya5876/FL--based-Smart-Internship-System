"""
Personalized Federated Learning (PFL) Implementation
Using FedBN and Fine-Tuning for Job Recommendation
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# =====================================
# STEP 1: Define the Neural Network Model
# =====================================

class JobRecommendationModel(nn.Module):
    """
    Simple neural network for job matching score prediction.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(JobRecommendationModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # output layer
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# =====================================
# STEP 2: Data Preprocessing
# =====================================

def preprocess_client_data(client_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for a single client.
    """
    df = pd.read_csv(f"{client_path}/data.csv")
    
    numerical_features = ['GPA', 'age', 'salary_min', 'salary_max']
    categorical_features = ['sex', 'major', 'role', 'work_type', 'industry']
    
    # Handle missing values
    for col in numerical_features:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    for col in categorical_features:
        if col in df.columns:
            df[col].fillna("Unknown", inplace=True)
    
    features = []
    scaler = StandardScaler()
    if all(col in df.columns for col in numerical_features):
        num_features = scaler.fit_transform(df[numerical_features])
        features.append(num_features)
    
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str))
            features.append(encoded.reshape(-1, 1))
    
    X = np.hstack(features)
    y = df["match_score"].values
    
    return X, y


# =====================================
# STEP 3: PFL Client
# =====================================

class PFLClient:
    def __init__(self, client_id: str, data_path: str, model_params: dict):
        self.client_id = client_id
        self.data_path = data_path
        
        # Load data
        X, y = preprocess_client_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        self.test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
        
        # Model
        input_dim = X_train.shape[1]
        self.model = JobRecommendationModel(input_dim)
        
        # Params
        self.epochs = model_params.get("epochs", 5)
        self.batch_size = model_params.get("batch_size", 32)
        self.learning_rate = model_params.get("learning_rate", 0.001)
    
    def train(self) -> float:
        """
        Local training on client's dataset
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        total_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss = epoch_loss / len(train_loader)
        
        return total_loss
    
    def evaluate(self) -> Dict:
        """
        Evaluate on test data
        """
        test_loader = DataLoader(self.test_dataset, batch_size=64)
        self.model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                p = self.model(X_batch)
                preds.extend(p.numpy().flatten())
                actuals.extend(y_batch.numpy().flatten())
        
        mse = mean_squared_error(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)
        
        return {"client_id": self.client_id, "mse": mse, "mae": mae, "r2": r2}
    
    def finetune(self, epochs: int = 1, lr: float = 0.001) -> float:
        """
        Fine-tune global model on local data
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss = epoch_loss / len(train_loader)
        return total_loss
    
    def get_model_weights(self) -> Dict:
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_weights(self, weights: Dict):
        for name, param in self.model.named_parameters():
            param.data = weights[name].clone()


# =====================================
# STEP 4: PFL Server
# =====================================

class PFLServer:
    def __init__(self, clients: List[PFLClient]):
        self.clients = clients
        self.global_model = None
        if clients:
            input_dim = clients[0].train_dataset.tensors[0].shape[1]
            self.global_model = JobRecommendationModel(input_dim)
    
    def aggregate_weights(self, client_weights: List[Tuple[Dict, int]]) -> Dict:
        total_samples = sum(n for _, n in client_weights)
        aggregated = {}
        for k in client_weights[0][0].keys():
            aggregated[k] = torch.zeros_like(client_weights[0][0][k])
            for weights, n in client_weights:
                aggregated[k] += weights[k] * (n / total_samples)
        return aggregated
    
    def train_round(self, round_num: int):
        print(f"\n=== Round {round_num} ===")
        
        global_weights = {n: p.data.clone() for n, p in self.global_model.named_parameters()}
        client_weights = []
        
        for client in self.clients:
            client.set_model_weights(global_weights)
            loss = client.train()
            weights = client.get_model_weights()
            client_weights.append((weights, len(client.train_dataset)))
            print(f"  Client {client.client_id}: Loss = {loss:.4f}")
        
        aggregated_weights = self.aggregate_weights(client_weights)
        for n, p in self.global_model.named_parameters():
            p.data = aggregated_weights[n]
    
    def evaluate_global(self):
        metrics = []
        global_weights = {n: p.data.clone() for n, p in self.global_model.named_parameters()}
        for client in self.clients:
            client.set_model_weights(global_weights)
            m = client.evaluate()
            metrics.append(m)
        return metrics
    
    def personalize(self):
        """
        Apply fine-tuning for each client after federated training
        """
        results = []
        for client in self.clients:
            loss = client.finetune(epochs=1, lr=0.001)
            metrics = client.evaluate()
            metrics["finetune_loss"] = loss
            results.append(metrics)
            print(f"Client {client.client_id} fine-tuned: Loss = {loss:.4f}")
        return results


# =====================================
# STEP 5: Run PFL Experiment
# =====================================

def run_pfl_experiment(
    data_dir: str = "../../../data/processed",
    num_rounds: int = 5,
    model_params: dict = None
):
    if model_params is None:
        model_params = {"epochs": 5, "batch_size": 32, "learning_rate": 0.001}
    
    print("=" * 50)
    print("PERSONALIZED FEDERATED LEARNING (PFL) EXPERIMENT")
    print("=" * 50)
    
    # Load clients
    clients = []
    client_dirs = [d for d in os.listdir(data_dir) if d.startswith("university_client_")]
    for client_dir in client_dirs[:6]:
        client_id = client_dir.split("_")[-1]
        path = os.path.join(data_dir, client_dir)
        client = PFLClient(client_id, path, model_params)
        clients.append(client)
        print(f"✓ Loaded client {client_id}")
    
    server = PFLServer(clients)
    
    # Train rounds
    for r in range(1, num_rounds + 1):
        server.train_round(r)
    
    # Evaluate
    print("\nEvaluating Global Model...")
    global_metrics = server.evaluate_global()
    
    # Personalization
    print("\nApplying Personalization (Fine-Tuning)...")
    personalized_metrics = server.personalize()
    
    results = {
        "global_eval": global_metrics,
        "personalized_eval": personalized_metrics
    }
    
    with open("./pfl_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ PFL experiment completed.")
    return server, results


# =====================================
# STEP 6: Main
# =====================================

if __name__ == "__main__":
    run_pfl_experiment(
        data_dir="../../../data/processed",
        num_rounds=5,
        model_params={"epochs": 5, "batch_size": 32, "learning_rate": 0.001}
    )
