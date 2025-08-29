"""
Federated Averaging (FedAvg) Implementation for Job Recommendation System
Testing RQ1: Handling extreme data heterogeneity in cross-organizational scenarios
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =====================================
# STEP 1: Define the Neural Network Model
# =====================================

class JobRecommendationModel(nn.Module):
    """
    Simple neural network for job matching score prediction.
    This model will be trained locally on each client (university).
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(JobRecommendationModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer for match score prediction
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# =====================================
# STEP 2: Create Custom Dataset Class
# =====================================

class JobMatchDataset(Dataset):
    """
    Custom dataset for job matching data.
    Handles the preprocessed data from each university client.
    """
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# =====================================
# STEP 3: Data Preprocessing Functions
# =====================================

def preprocess_client_data(client_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for a single client (university).
    Converts categorical variables and creates feature vectors.
    """
    # Load client data
    df = pd.read_csv(f'{client_path}/data.csv')
    
    # Select features for the model
    numerical_features = ['GPA', 'age', 'salary_min', 'salary_max']
    categorical_features = ['sex', 'major', 'role', 'work_type', 'industry']
    
    # Handle missing values
    for col in numerical_features:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    
    for col in categorical_features:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)
    
    # Create feature matrix
    features = []
    
    # Add numerical features (normalized)
    scaler = StandardScaler()
    if all(col in df.columns for col in numerical_features):
        num_features = scaler.fit_transform(df[numerical_features])
        features.append(num_features)
    
    # Encode categorical features
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str))
            features.append(encoded.reshape(-1, 1))
    
    # Combine all features
    X = np.hstack(features)
    
    # Target variable (match_score)
    y = df['match_score'].values
    
    return X, y

# =====================================
# STEP 4: Federated Learning Client
# =====================================

class FederatedClient:
    """
    Represents a single client (university) in the federated learning system.
    Each client trains on its local data.
    """
    def __init__(self, client_id: str, data_path: str, model_params: dict):
        self.client_id = client_id
        self.data_path = data_path
        
        # Load and preprocess data
        X, y = preprocess_client_data(data_path)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create datasets
        self.train_dataset = JobMatchDataset(X_train, y_train)
        self.test_dataset = JobMatchDataset(X_test, y_test)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = JobRecommendationModel(input_dim)
        
        # Training parameters
        self.epochs = model_params.get('epochs', 5)
        self.batch_size = model_params.get('batch_size', 32)
        self.learning_rate = model_params.get('learning_rate', 0.001)
        
        # Data statistics for heterogeneity analysis
        self.data_stats = {
            'num_samples': len(X),
            'feature_dim': input_dim,
            'mean_target': np.mean(y),
            'std_target': np.std(y)
        }
    
    def train(self) -> Dict:
        """
        Train the model on local data.
        Returns training metrics.
        """
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            total_loss = epoch_loss / len(train_loader)
        
        return {
            'client_id': self.client_id,
            'final_loss': total_loss,
            'num_samples': len(self.train_dataset)
        }
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test data.
        Returns evaluation metrics.
        """
        test_loader = DataLoader(self.test_dataset, batch_size=64)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                preds = self.model(features)
                predictions.extend(preds.numpy().flatten())
                actuals.extend(labels.numpy().flatten())
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'client_id': self.client_id,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'num_test_samples': len(actuals)
        }
    
    def get_model_weights(self) -> Dict:
        """
        Get model weights for aggregation.
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_weights(self, weights: Dict):
        """
        Set model weights from global model.
        """
        for name, param in self.model.named_parameters():
            param.data = weights[name].clone()

# =====================================
# STEP 5: Federated Averaging Server
# =====================================

class FedAvgServer:
    """
    Central server that coordinates federated learning.
    Implements the FedAvg algorithm.
    """
    def __init__(self, clients: List[FederatedClient]):
        self.clients = clients
        self.global_model = None
        self.round_metrics = []
        
        # Initialize global model with same architecture
        if clients:
            # Use first client's model architecture
            first_client_model = clients[0].model
            self.global_model = JobRecommendationModel(
                input_dim=clients[0].data_stats['feature_dim']
            )
            # Initialize with first client's weights
            self.global_model.load_state_dict(first_client_model.state_dict())
    
    def aggregate_weights(self, client_weights: List[Tuple[Dict, int]]) -> Dict:
        """
        Perform weighted averaging of client models.
        Weights are based on number of samples per client.
        """
        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples in client_weights)
        
        # Initialize aggregated weights
        aggregated = {}
        
        # Weighted average for each parameter
        for param_name in client_weights[0][0].keys():
            aggregated[param_name] = torch.zeros_like(client_weights[0][0][param_name])
            
            for weights, num_samples in client_weights:
                weight_factor = num_samples / total_samples
                aggregated[param_name] += weights[param_name] * weight_factor
        
        return aggregated
    
    def train_round(self, round_num: int, sample_fraction: float = 1.0) -> Dict:
        """
        Execute one round of federated training.
        """
        print(f"\n=== Round {round_num} ===")
        
        # Sample clients for this round
        num_clients = max(1, int(sample_fraction * len(self.clients)))
        selected_clients = np.random.choice(self.clients, num_clients, replace=False)
        
        # Distribute global model to selected clients
        global_weights = {name: param.data.clone() 
                         for name, param in self.global_model.named_parameters()}
        
        for client in selected_clients:
            client.set_model_weights(global_weights)
        
        # Local training
        client_weights = []
        training_metrics = []
        
        for client in selected_clients:
            # Train locally
            train_metrics = client.train()
            training_metrics.append(train_metrics)
            
            # Get updated weights
            weights = client.get_model_weights()
            num_samples = len(client.train_dataset)
            client_weights.append((weights, num_samples))
            
            print(f"  Client {client.client_id}: Loss = {train_metrics['final_loss']:.4f}")
        
        # Aggregate weights
        aggregated_weights = self.aggregate_weights(client_weights)
        
        # Update global model
        for name, param in self.global_model.named_parameters():
            param.data = aggregated_weights[name]
        
        # Evaluate global model on all clients
        eval_metrics = self.evaluate_global_model()
        
        # Store metrics
        round_metrics = {
            'round': round_num,
            'training_metrics': training_metrics,
            'eval_metrics': eval_metrics,
            'avg_mse': np.mean([m['mse'] for m in eval_metrics]),
            'avg_mae': np.mean([m['mae'] for m in eval_metrics]),
            'avg_r2': np.mean([m['r2'] for m in eval_metrics])
        }
        self.round_metrics.append(round_metrics)
        
        print(f"  Global Performance: MSE = {round_metrics['avg_mse']:.4f}, "
              f"MAE = {round_metrics['avg_mae']:.4f}, R² = {round_metrics['avg_r2']:.4f}")
        
        return round_metrics
    
    def evaluate_global_model(self) -> List[Dict]:
        """
        Evaluate global model on all clients' test data.
        """
        eval_results = []
        
        # Set global weights to all clients
        global_weights = {name: param.data.clone() 
                         for name, param in self.global_model.named_parameters()}
        
        for client in self.clients:
            client.set_model_weights(global_weights)
            eval_metrics = client.evaluate()
            eval_results.append(eval_metrics)
        
        return eval_results

# =====================================
# STEP 6: Heterogeneity Analysis
# =====================================

def analyze_data_heterogeneity(clients: List[FederatedClient]) -> pd.DataFrame:
    """
    Analyze and visualize data heterogeneity across clients.
    """
    heterogeneity_stats = []
    
    for client in clients:
        stats = client.data_stats.copy()
        stats['client_id'] = client.client_id
        heterogeneity_stats.append(stats)
    
    df_stats = pd.DataFrame(heterogeneity_stats)
    
    # Calculate heterogeneity metrics
    print("\n=== Data Heterogeneity Analysis ===")
    print(f"Sample size variance: {df_stats['num_samples'].var():.2f}")
    print(f"Target mean variance: {df_stats['mean_target'].var():.4f}")
    print(f"Target std variance: {df_stats['std_target'].var():.4f}")
    
    return df_stats

# =====================================
# STEP 7: Visualization Functions
# =====================================

def plot_training_progress(round_metrics: List[Dict]):
    """
    Visualize training progress across federated rounds.
    """
    rounds = [m['round'] for m in round_metrics]
    avg_mse = [m['avg_mse'] for m in round_metrics]
    avg_mae = [m['avg_mae'] for m in round_metrics]
    avg_r2 = [m['avg_r2'] for m in round_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # MSE plot
    axes[0].plot(rounds, avg_mse, 'b-o')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Average MSE')
    axes[0].set_title('MSE Across Federated Rounds')
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(rounds, avg_mae, 'g-o')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Average MAE')
    axes[1].set_title('MAE Across Federated Rounds')
    axes[1].grid(True, alpha=0.3)
    
    # R² plot
    axes[2].plot(rounds, avg_r2, 'r-o')
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Average R²')
    axes[2].set_title('R² Score Across Federated Rounds')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./logs/fedavg_training_progress.png', dpi=100)
    plt.show()

def plot_client_performance_variance(round_metrics: List[Dict]):
    """
    Visualize performance variance across clients.
    """
    # Get final round metrics
    final_round = round_metrics[-1]
    eval_metrics = final_round['eval_metrics']
    
    # Create dataframe for plotting
    df_metrics = pd.DataFrame(eval_metrics)
    
    # Create box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # MSE distribution
    axes[0].boxplot([m['mse'] for m in eval_metrics])
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE Distribution Across Clients')
    axes[0].set_xticklabels(['All Clients'])
    
    # MAE distribution
    axes[1].boxplot([m['mae'] for m in eval_metrics])
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Distribution Across Clients')
    axes[1].set_xticklabels(['All Clients'])
    
    # R² distribution
    axes[2].boxplot([m['r2'] for m in eval_metrics])
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² Distribution Across Clients')
    axes[2].set_xticklabels(['All Clients'])
    
    plt.tight_layout()
    plt.savefig('./logs/fedavg_client_variance.png', dpi=100)
    plt.show()
    
    # Print variance statistics
    print("\n=== Performance Variance Across Clients ===")
    print(f"MSE std: {np.std([m['mse'] for m in eval_metrics]):.4f}")
    print(f"MAE std: {np.std([m['mae'] for m in eval_metrics]):.4f}")
    print(f"R² std: {np.std([m['r2'] for m in eval_metrics]):.4f}")

# =====================================
# STEP 8: Main Execution Function
# =====================================

def run_fedavg_experiment(
    data_dir: str = '../shared/test-data/processed',
    num_rounds: int = 10,
    client_fraction: float = 1.0,
    model_params: dict = None
):
    """
    Run complete FedAvg experiment for job recommendation.
    """
    print("=" * 50)
    print("FEDERATED AVERAGING (FedAvg) EXPERIMENT")
    print("Testing RQ1: Data Heterogeneity in Job Recommendation")
    print("=" * 50)
    
    # Default model parameters
    if model_params is None:
        model_params = {
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    
    # Step 1: Load all clients
    print("\n1. Loading client data...")
    clients = []
    client_dirs = [d for d in os.listdir(data_dir) 
                   if d.startswith('university_client_')]
    
    for client_dir in client_dirs[:6]:  # Limit to 6 clients for demo
        client_id = client_dir.split('_')[-1]
        client_path = os.path.join(data_dir, client_dir)
        
        if os.path.exists(f'{client_path}/data.csv'):
            try:
                client = FederatedClient(client_id, client_path, model_params)
                clients.append(client)
                print(f"  ✓ Loaded client {client_id}: {client.data_stats['num_samples']} samples")
            except Exception as e:
                print(f"  ✗ Failed to load client {client_id}: {e}")
    
    print(f"\nTotal clients loaded: {len(clients)}")
    
    # Step 2: Analyze data heterogeneity
    print("\n2. Analyzing data heterogeneity...")
    heterogeneity_df = analyze_data_heterogeneity(clients)
    
    # Step 3: Initialize FedAvg server
    print("\n3. Initializing FedAvg server...")
    server = FedAvgServer(clients)
    
    # Step 4: Run federated training
    print("\n4. Starting federated training...")
    for round_num in range(1, num_rounds + 1):
        round_metrics = server.train_round(round_num, client_fraction)
    
    # Step 5: Visualize results
    print("\n5. Generating visualizations...")
    plot_training_progress(server.round_metrics)
    plot_client_performance_variance(server.round_metrics)
    
    # Step 6: Save results
    print("\n6. Saving results...")
    results = {
        'heterogeneity_stats': heterogeneity_df.to_dict(),
        'training_metrics': server.round_metrics,
        'final_performance': {
            'avg_mse': server.round_metrics[-1]['avg_mse'],
            'avg_mae': server.round_metrics[-1]['avg_mae'],
            'avg_r2': server.round_metrics[-1]['avg_r2']
        }
    }

    with open('./logs/fedavg_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETED")
    print("=" * 50)
    print(f"Final Global Model Performance:")
    print(f"  MSE: {results['final_performance']['avg_mse']:.4f}")
    print(f"  MAE: {results['final_performance']['avg_mae']:.4f}")
    print(f"  R²:  {results['final_performance']['avg_r2']:.4f}")
    
    return server, results

# =====================================
# STEP 9: Run the Experiment
# =====================================

if __name__ == "__main__":
    # Run FedAvg experiment
    server, results = run_fedavg_experiment(
        data_dir='../shared/test-data/processed',
        num_rounds=10,
        client_fraction=1.0,  # Use all clients each round
        model_params={
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    )
    
    print("\n✅ FedAvg baseline established. Ready for comparison with proposed algorithm.")