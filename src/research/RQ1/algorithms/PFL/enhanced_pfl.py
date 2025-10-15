"""
Enhanced Personalized Federated Learning (PFL) Implementation
With Advanced Adaptive Strategies for Job Recommendation
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
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')


# =====================================
# STEP 1: Enhanced Neural Network with Batch Normalization
# =====================================

class EnhancedJobRecommendationModel(nn.Module):
    """
    Enhanced neural network with batch normalization for better personalization
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(EnhancedJobRecommendationModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Batch normalization for better training
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# =====================================
# STEP 2: Enhanced Data Preprocessing
# =====================================

def enhanced_preprocess_client_data(client_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Enhanced preprocessing with heterogeneity analysis
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
    
    # Calculate heterogeneity metrics
    heterogeneity_metrics = {
        'data_size': len(df),
        'target_mean': np.mean(y),
        'target_std': np.std(y),
        'feature_variance': np.var(X, axis=0).mean(),
        'categorical_diversity': len(set(df['major'].values)) if 'major' in df.columns else 0
    }
    
    return X, y, heterogeneity_metrics


# =====================================
# STEP 3: Enhanced PFL Client
# =====================================

class EnhancedPFLClient:
    def __init__(self, client_id: str, data_path: str, model_params: dict):
        self.client_id = client_id
        self.data_path = data_path
        
        # Load data with heterogeneity analysis
        X, y, self.heterogeneity_metrics = enhanced_preprocess_client_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        self.test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))
        
        # Model
        input_dim = X_train.shape[1]
        self.model = EnhancedJobRecommendationModel(input_dim)
        
        # Enhanced parameters
        self.epochs = model_params.get("epochs", 5)
        self.batch_size = model_params.get("batch_size", 32)
        self.learning_rate = model_params.get("learning_rate", 0.001)
        
        # Adaptive parameters
        self.adaptation_history = []
        self.convergence_threshold = 0.001
        
    def compute_heterogeneity_score(self) -> float:
        """Compute client-specific heterogeneity measure"""
        # Combine multiple heterogeneity indicators
        data_size_factor = min(1.0, self.heterogeneity_metrics['data_size'] / 1000)
        variance_factor = self.heterogeneity_metrics['feature_variance']
        diversity_factor = self.heterogeneity_metrics['categorical_diversity'] / 10
        
        heterogeneity_score = (variance_factor + diversity_factor) / (1 + data_size_factor)
        return min(2.0, max(0.1, heterogeneity_score))
    
    def adaptive_learning_rate(self, base_lr: float) -> float:
        """Adapt learning rate based on local heterogeneity"""
        heterogeneity_factor = self.compute_heterogeneity_score()
        return base_lr * (1 + heterogeneity_factor * 0.3)
    
    def adaptive_epochs(self, base_epochs: int) -> int:
        """Adapt training epochs based on convergence history"""
        if len(self.adaptation_history) > 2:
            recent_improvement = self.adaptation_history[-1] - self.adaptation_history[-3]
            if recent_improvement < self.convergence_threshold:
                return min(base_epochs * 2, 15)
        return base_epochs
    
    def train(self) -> float:
        """Enhanced local training with adaptive parameters"""
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Adaptive learning rate
        adaptive_lr = self.adaptive_learning_rate(self.learning_rate)
        optimizer = optim.Adam(self.model.parameters(), lr=adaptive_lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        total_loss = 0
        
        # Adaptive epochs
        adaptive_epochs = self.adaptive_epochs(self.epochs)
        
        for epoch in range(adaptive_epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss = epoch_loss / len(train_loader)
        
        # Store for adaptation analysis
        self.adaptation_history.append(total_loss)
        return total_loss
    
    def evaluate(self) -> Dict:
        """Enhanced evaluation with additional metrics"""
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
        
        # Additional metrics
        mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / (np.array(actuals) + 1e-8))) * 100
        
        return {
            "client_id": self.client_id,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "heterogeneity_score": self.compute_heterogeneity_score()
        }
    
    def progressive_finetune(self, base_epochs: int = 3) -> Dict:
        """Progressive fine-tuning with decreasing learning rates"""
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.MSELoss()
        
        # Progressive learning rates
        learning_rates = [0.001, 0.0005, 0.0001]
        total_loss = 0
        
        for i, lr in enumerate(learning_rates):
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.model.train()
            
            for epoch in range(base_epochs):
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    preds = self.model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                total_loss = epoch_loss / len(train_loader)
        
        return {
            "finetune_loss": total_loss,
            "learning_rates_used": learning_rates,
            "total_epochs": base_epochs * len(learning_rates)
        }
    
    def get_model_weights(self) -> Dict:
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_weights(self, weights: Dict):
        for name, param in self.model.named_parameters():
            param.data = weights[name].clone()


# =====================================
# STEP 4: Enhanced PFL Server with Clustering
# =====================================

class EnhancedPFLServer:
    def __init__(self, clients: List[EnhancedPFLClient]):
        self.clients = clients
        self.global_model = None
        self.client_clusters = None
        
        if clients:
            input_dim = clients[0].train_dataset.tensors[0].shape[1]
            self.global_model = EnhancedJobRecommendationModel(input_dim)
            self.cluster_clients()
    
    def cluster_clients(self):
        """Cluster clients based on data characteristics"""
        client_features = []
        for client in self.clients:
            features = [
                client.heterogeneity_metrics['data_size'],
                client.heterogeneity_metrics['target_mean'],
                client.heterogeneity_metrics['target_std'],
                client.heterogeneity_metrics['feature_variance'],
                client.heterogeneity_metrics['categorical_diversity']
            ]
            client_features.append(features)
        
        # Normalize features
        client_features = np.array(client_features)
        client_features = (client_features - client_features.mean(axis=0)) / (client_features.std(axis=0) + 1e-8)
        
        # Perform clustering
        n_clusters = min(3, len(self.clients))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.client_clusters = kmeans.fit_predict(client_features)
        
        print(f"Client clustering: {dict(zip([c.client_id for c in self.clients], self.client_clusters))}")
    
    def adaptive_aggregate_weights(self, client_weights: List[Tuple[Dict, int]], performance_metrics: List[Dict]) -> Dict:
        """Adaptive aggregation based on performance and data quality"""
        total_samples = sum(n for _, n in client_weights)
        aggregated = {}
        
        # Calculate adaptive weights
        adaptive_weights = []
        for i, ((weights, n_samples), metrics) in enumerate(zip(client_weights, performance_metrics)):
            # Weight based on performance (better performance = higher weight)
            performance_weight = 1.0 / (1.0 + metrics['mse'])
            
            # Weight based on data size
            size_weight = np.sqrt(n_samples)
            
            # Weight based on heterogeneity (more heterogeneous = higher weight for personalization)
            heterogeneity_weight = 1.0 + metrics['heterogeneity_score']
            
            total_weight = performance_weight * size_weight * heterogeneity_weight
            adaptive_weights.append((weights, total_weight))
        
        # Normalize weights
        total_adaptive_weight = sum(w for _, w in adaptive_weights)
        
        for k in client_weights[0][0].keys():
            aggregated[k] = torch.zeros_like(client_weights[0][0][k])
            for (weights, adaptive_weight) in adaptive_weights:
                normalized_weight = adaptive_weight / total_adaptive_weight
                aggregated[k] += weights[k] * normalized_weight
        
        return aggregated
    
    def train_round(self, round_num: int):
        print(f"\n=== Enhanced Round {round_num} ===")
        
        global_weights = {n: p.data.clone() for n, p in self.global_model.named_parameters()}
        client_weights = []
        performance_metrics = []
        
        for client in self.clients:
            client.set_model_weights(global_weights)
            loss = client.train()
            weights = client.get_model_weights()
            metrics = client.evaluate()
            
            client_weights.append((weights, len(client.train_dataset)))
            performance_metrics.append(metrics)
            
            print(f"  Client {client.client_id}: Loss = {loss:.4f}, MSE = {metrics['mse']:.4f}, "
                  f"Heterogeneity = {metrics['heterogeneity_score']:.3f}")
        
        # Use adaptive aggregation
        aggregated_weights = self.adaptive_aggregate_weights(client_weights, performance_metrics)
        for n, p in self.global_model.named_parameters():
            p.data = aggregated_weights[n]
    
    def evaluate_global(self):
        """Evaluate global model on all clients"""
        metrics = []
        global_weights = {n: p.data.clone() for n, p in self.global_model.named_parameters()}
        
        for client in self.clients:
            client.set_model_weights(global_weights)
            m = client.evaluate()
            metrics.append(m)
        
        return metrics
    
    def enhanced_personalize(self):
        """Enhanced personalization with progressive fine-tuning"""
        results = []
        
        for client in self.clients:
            print(f"Personalizing client {client.client_id}...")
            
            # Progressive fine-tuning
            finetune_info = client.progressive_finetune(base_epochs=2)
            metrics = client.evaluate()
            
            # Combine results
            result = {**metrics, **finetune_info}
            results.append(result)
            
            print(f"  Final MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return results


# =====================================
# STEP 5: Enhanced PFL Experiment
# =====================================

def run_enhanced_pfl_experiment(
    data_dir: str = "../shared/test-data/processed",
    num_rounds: int = 8,
    model_params: dict = None
):
    if model_params is None:
        model_params = {"epochs": 5, "batch_size": 32, "learning_rate": 0.001}
    
    print("=" * 60)
    print("ENHANCED PERSONALIZED FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load clients
    clients = []
    client_dirs = [d for d in os.listdir(data_dir) if d.startswith("university_client_")]
    
    for client_dir in client_dirs[:6]:
        client_id = client_dir.split("_")[-1]
        path = os.path.join(data_dir, client_dir)
        client = EnhancedPFLClient(client_id, path, model_params)
        clients.append(client)
        print(f"[OK] Loaded client {client_id} (Heterogeneity: {client.compute_heterogeneity_score():.3f})")
    
    server = EnhancedPFLServer(clients)
    
    # Enhanced federated training
    print(f"\n[START] Starting Enhanced Federated Training ({num_rounds} rounds)...")
    for r in range(1, num_rounds + 1):
        server.train_round(r)
    
    # Evaluate global model
    print("\n[EVAL] Evaluating Global Model...")
    global_metrics = server.evaluate_global()
    
    # Enhanced personalization
    print("\n[PERSONALIZE] Applying Enhanced Personalization...")
    personalized_metrics = server.enhanced_personalize()
    
    # Calculate improvements
    improvements = []
    for global_m, personalized_m in zip(global_metrics, personalized_metrics):
        mse_improvement = ((global_m['mse'] - personalized_m['mse']) / global_m['mse']) * 100
        r2_improvement = personalized_m['r2'] - global_m['r2']
        
        improvements.append({
            'client_id': global_m['client_id'],
            'mse_improvement_pct': mse_improvement,
            'r2_improvement': r2_improvement
        })
    
    # Compile results
    results = {
        "algorithm": "Enhanced PFL",
        "timestamp": datetime.now().isoformat(),
        "experiment_duration": time.time() - start_time,
        "global_eval": global_metrics,
        "personalized_eval": personalized_metrics,
        "improvements": improvements,
        "heterogeneity_analysis": {
            client.client_id: client.heterogeneity_metrics 
            for client in clients
        }
    }
    
    # Save results
    with open("./enhanced_pfl_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n[SUCCESS] Enhanced PFL experiment completed!")
    print(f"[TIME] Total time: {results['experiment_duration']:.2f} seconds")
    
    return server, results


# =====================================
# STEP 6: Main Execution
# =====================================

if __name__ == "__main__":
    run_enhanced_pfl_experiment(
        data_dir="../../../data/processed",
        num_rounds=8,
        model_params={"epochs": 5, "batch_size": 32, "learning_rate": 0.001}
    )
