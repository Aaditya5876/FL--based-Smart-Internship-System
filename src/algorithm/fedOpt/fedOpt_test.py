import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import copy
import time

# Configure logging
os.makedirs('./', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./fedopt_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JobRecommendationModel(nn.Module):
    """Neural network for job recommendation with extreme heterogeneity handling"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(JobRecommendationModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer for binary recommendation (match/no match)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class FedOptClient:
    """FedOpt Client implementing local training with momentum and adaptive learning"""
    
    def __init__(self, client_id: str, data_path: str, feature_dim: int):
        self.client_id = client_id
        self.data_path = data_path
        self.feature_dim = feature_dim
        self.model = JobRecommendationModel(feature_dim)
        self.local_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # FedOpt specific parameters
        self.local_momentum = {}  # Store momentum terms for each parameter
        self.local_lr = 0.01
        self.momentum_coeff = 0.9
        
        self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """Load and preprocess client-specific data"""
        try:
            self.local_data = pd.read_csv(self.data_path)
            logger.info(f"Client {self.client_id}: Loaded {len(self.local_data)} samples")
            
            # Extract features
            features = self._extract_features()
            targets = self.local_data['match_score'].values
            
            # Convert to binary classification (>0.5 as positive match)
            targets = (targets > 0.5).astype(int)
            
            # Split data
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                features, targets, test_size=0.2, random_state=42, stratify=targets
            )
            
            logger.info(f"Client {self.client_id}: Train samples: {len(self.X_train)}, Val samples: {len(self.X_val)}")
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Error loading data: {e}")
            raise
    
    def _extract_features(self) -> np.ndarray:
        """Extract features for the recommendation model"""
        features = []
        
        # Numerical features
        numerical_features = ['age', 'GPA_normalized']
        for col in numerical_features:
            if col in self.local_data.columns:
                features.append(self.local_data[col].fillna(0).values)
        
        # Categorical features (one-hot encoded)
        categorical_features = ['sex', 'work_type']
        for col in categorical_features:
            if col in self.local_data.columns:
                unique_vals = self.local_data[col].unique()
                for val in unique_vals:
                    features.append((self.local_data[col] == val).astype(int).values)
        
        # Salary range feature
        if 'salary_min' in self.local_data.columns and 'salary_max' in self.local_data.columns:
            salary_avg = (self.local_data['salary_min'].fillna(0) + self.local_data['salary_max'].fillna(0)) / 2
            features.append(salary_avg.values)
        
        # Skills similarity (simplified - using length as proxy)
        if 'skills' in self.local_data.columns:
            skills_length = self.local_data['skills'].fillna('').str.len()
            features.append(skills_length.values)
        
        # Combine all features
        features_array = np.column_stack(features)
        
        # Pad or truncate to match expected feature dimension
        if features_array.shape[1] < self.feature_dim:
            padding = np.zeros((features_array.shape[0], self.feature_dim - features_array.shape[1]))
            features_array = np.hstack([features_array, padding])
        elif features_array.shape[1] > self.feature_dim:
            features_array = features_array[:, :self.feature_dim]
        
        return features_array.astype(np.float32)
    
    def local_train(self, global_model_state: Dict, local_epochs: int = 5) -> Dict:
        """Perform local training with FedOpt momentum"""
        # Load global model
        self.model.load_state_dict(global_model_state)
        
        # Store initial parameters for momentum calculation
        initial_params = copy.deepcopy(dict(self.model.named_parameters()))
        
        # Local training setup
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, momentum=self.momentum_coeff)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(self.X_val)
        y_val_tensor = torch.FloatTensor(self.y_val).unsqueeze(1)
        
        self.model.train()
        train_losses = []
        
        for epoch in range(local_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        self.model.eval()
        with torch.no_grad():
            val_predictions = self.model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor)
            val_accuracy = accuracy_score(y_val_tensor.numpy(), (val_predictions.numpy() > 0.5).astype(int))
        
        # Calculate parameter updates (delta)
        final_params = dict(self.model.named_parameters())
        param_deltas = {}
        
        for name in initial_params:
            if name in final_params:
                param_deltas[name] = final_params[name].data - initial_params[name].data
            else:
                param_deltas[name] = torch.zeros_like(initial_params[name].data)
        
        # Client statistics for heterogeneity analysis
        client_stats = {
            'client_id': self.client_id,
            'samples': len(self.local_data),
            'train_loss': np.mean(train_losses),
            'val_loss': val_loss.item(),
            'val_accuracy': val_accuracy,
            'data_heterogeneity': self._calculate_data_heterogeneity()
        }
        
        logger.info(f"Client {self.client_id}: Train Loss: {client_stats['train_loss']:.4f}, "
                   f"Val Accuracy: {val_accuracy:.4f}")
        
        return {
            'param_deltas': param_deltas,
            'num_samples': len(self.X_train),
            'client_stats': client_stats
        }
    
    def _calculate_data_heterogeneity(self) -> Dict:
        """Calculate data heterogeneity metrics for this client"""
        heterogeneity = {}
        
        # GPA distribution
        if 'GPA' in self.local_data.columns:
            heterogeneity['gpa_mean'] = float(self.local_data['GPA'].mean())
            heterogeneity['gpa_std'] = float(self.local_data['GPA'].std())
        
        # Gender distribution
        if 'sex' in self.local_data.columns:
            gender_dist = self.local_data['sex'].value_counts(normalize=True)
            heterogeneity['gender_entropy'] = -sum(p * np.log2(p + 1e-10) for p in gender_dist.values)
        
        # Major diversity
        if 'major' in self.local_data.columns:
            heterogeneity['major_diversity'] = int(self.local_data['major'].nunique())
        
        # Industry diversity
        if 'industry' in self.local_data.columns:
            heterogeneity['industry_diversity'] = int(self.local_data['industry'].nunique())
        
        return heterogeneity

class FedOptServer:
    """FedOpt Server implementing adaptive server-side optimization"""
    
    def __init__(self, feature_dim: int, optimizer_type: str = 'adam', 
                 server_lr: float = 1.0, beta1: float = 0.9, beta2: float = 0.999):
        self.feature_dim = feature_dim
        self.global_model = JobRecommendationModel(feature_dim)
        self.optimizer_type = optimizer_type.lower()
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Server-side optimization states
        self.server_state = {}
        self.round_number = 0
        
        # Initialize server optimizer state
        self._initialize_server_state()
        
        # Metrics tracking
        self.training_history = {
            'rounds': [],
            'avg_train_loss': [],
            'avg_val_accuracy': [],
            'client_participation': [],
            'heterogeneity_metrics': []
        }
        
        logger.info(f"FedOpt Server initialized with {optimizer_type} optimizer")
    
    def _initialize_server_state(self):
        """Initialize server-side optimizer state"""
        for name, param in self.global_model.named_parameters():
            self.server_state[name] = {
                'momentum': torch.zeros_like(param.data),  # First moment (Adam) or momentum (SGD)
                'velocity': torch.zeros_like(param.data),  # Second moment (Adam)
                'step': 0
            }
    
    def aggregate_and_update(self, client_updates: List[Dict]) -> None:
        """FedOpt aggregation with server-side adaptive optimization"""
        if not client_updates:
            logger.warning("No client updates received")
            return
        
        self.round_number += 1
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Weighted aggregation of client deltas
        aggregated_deltas = {}
        
        # Initialize aggregated deltas
        for name, param in self.global_model.named_parameters():
            aggregated_deltas[name] = torch.zeros_like(param.data)
        
        # Aggregate client parameter updates
        for update in client_updates:
            weight = update['num_samples'] / total_samples
            param_deltas = update['param_deltas']
            
            for name in aggregated_deltas:
                if name in param_deltas:
                    aggregated_deltas[name] += weight * param_deltas[name]
        
        # Apply server-side optimization
        self._apply_server_optimization(aggregated_deltas)
        
        # Track metrics
        self._update_training_history(client_updates, total_samples)
        
        logger.info(f"Round {self.round_number}: Aggregated updates from {len(client_updates)} clients")
    
    def _apply_server_optimization(self, aggregated_deltas: Dict):
        """Apply server-side adaptive optimization (Adam, Yogi, etc.)"""
        current_params = dict(self.global_model.named_parameters())
        
        for name, param in current_params.items():
            if name not in aggregated_deltas:
                continue
                
            delta = aggregated_deltas[name]
            state = self.server_state[name]
            state['step'] += 1
            
            if self.optimizer_type == 'adam':
                # Adam optimizer
                state['momentum'] = self.beta1 * state['momentum'] + (1 - self.beta1) * delta
                state['velocity'] = self.beta2 * state['velocity'] + (1 - self.beta2) * (delta ** 2)
                
                # Bias correction
                momentum_corrected = state['momentum'] / (1 - self.beta1 ** state['step'])
                velocity_corrected = state['velocity'] / (1 - self.beta2 ** state['step'])
                
                # Update parameter
                param.data -= self.server_lr * momentum_corrected / (torch.sqrt(velocity_corrected) + 1e-8)
                
            elif self.optimizer_type == 'yogi':
                # Yogi optimizer (more aggressive than Adam)
                state['momentum'] = self.beta1 * state['momentum'] + (1 - self.beta1) * delta
                state['velocity'] = state['velocity'] - (1 - self.beta2) * torch.sign(state['velocity'] - delta ** 2) * (delta ** 2)
                
                # Bias correction
                momentum_corrected = state['momentum'] / (1 - self.beta1 ** state['step'])
                velocity_corrected = state['velocity'] / (1 - self.beta2 ** state['step'])
                
                # Update parameter
                param.data -= self.server_lr * momentum_corrected / (torch.sqrt(torch.abs(velocity_corrected)) + 1e-8)
                
            elif self.optimizer_type == 'adagrad':
                # Adagrad optimizer
                state['velocity'] += delta ** 2
                param.data -= self.server_lr * delta / (torch.sqrt(state['velocity']) + 1e-8)
                
            else:  # Default to SGD with momentum
                state['momentum'] = self.beta1 * state['momentum'] + delta
                param.data -= self.server_lr * state['momentum']
    
    def _update_training_history(self, client_updates: List[Dict], total_samples: int):
        """Update training metrics and heterogeneity analysis"""
        # Calculate average metrics
        avg_train_loss = np.mean([update['client_stats']['train_loss'] for update in client_updates])
        avg_val_accuracy = np.mean([update['client_stats']['val_accuracy'] for update in client_updates])
        
        # Heterogeneity metrics
        heterogeneity_metrics = self._analyze_client_heterogeneity(client_updates)
        
        # Update history
        self.training_history['rounds'].append(self.round_number)
        self.training_history['avg_train_loss'].append(avg_train_loss)
        self.training_history['avg_val_accuracy'].append(avg_val_accuracy)
        self.training_history['client_participation'].append(len(client_updates))
        self.training_history['heterogeneity_metrics'].append(heterogeneity_metrics)
        
        logger.info(f"Round {self.round_number} Metrics - Avg Train Loss: {avg_train_loss:.4f}, "
                   f"Avg Val Accuracy: {avg_val_accuracy:.4f}")
    
    def _analyze_client_heterogeneity(self, client_updates: List[Dict]) -> Dict:
        """Analyze data heterogeneity across clients"""
        heterogeneity_analysis = {
            'gpa_variance_across_clients': 0.0,
            'gender_entropy_variance': 0.0,
            'major_diversity_range': [0, 0],
            'industry_diversity_range': [0, 0],
            'performance_variance': 0.0
        }
        
        try:
            # Extract heterogeneity data from client stats
            gpa_means = []
            gender_entropies = []
            major_diversities = []
            industry_diversities = []
            val_accuracies = []
            
            for update in client_updates:
                het_data = update['client_stats']['data_heterogeneity']
                
                if 'gpa_mean' in het_data:
                    gpa_means.append(het_data['gpa_mean'])
                if 'gender_entropy' in het_data:
                    gender_entropies.append(het_data['gender_entropy'])
                if 'major_diversity' in het_data:
                    major_diversities.append(het_data['major_diversity'])
                if 'industry_diversity' in het_data:
                    industry_diversities.append(het_data['industry_diversity'])
                
                val_accuracies.append(update['client_stats']['val_accuracy'])
            
            # Calculate variances and ranges
            if gpa_means:
                heterogeneity_analysis['gpa_variance_across_clients'] = float(np.var(gpa_means))
            if gender_entropies:
                heterogeneity_analysis['gender_entropy_variance'] = float(np.var(gender_entropies))
            if major_diversities:
                heterogeneity_analysis['major_diversity_range'] = [int(min(major_diversities)), int(max(major_diversities))]
            if industry_diversities:
                heterogeneity_analysis['industry_diversity_range'] = [int(min(industry_diversities)), int(max(industry_diversities))]
            if val_accuracies:
                heterogeneity_analysis['performance_variance'] = float(np.var(val_accuracies))
                
        except Exception as e:
            logger.warning(f"Error analyzing heterogeneity: {e}")
        
        return heterogeneity_analysis
    
    def get_global_model_state(self) -> Dict:
        """Return current global model state"""
        return self.global_model.state_dict()
    
    def save_results(self, output_dir: str):
        """Save training results and analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training history
        results = {
            'algorithm': 'FedOpt',
            'optimizer': self.optimizer_type,
            'server_lr': self.server_lr,
            'total_rounds': self.round_number,
            'final_avg_accuracy': self.training_history['avg_val_accuracy'][-1] if self.training_history['avg_val_accuracy'] else 0,
            'training_history': self.training_history,
            'experiment_timestamp': datetime.now().isoformat()
        }
        
        with open(f'{output_dir}/fedopt_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")

def run_fedopt_experiment():
    """Run FedOpt experiment on heterogeneous internship data"""
    logger.info("Starting FedOpt experiment for Smart Internship System")
    
    # Configuration
    config = {
        'feature_dim': 20,  # Adjust based on your feature extraction
        'communication_rounds': 30,
        'local_epochs': 5,
        'client_fraction': 1.0,  # Use all available clients
        'optimizer_type': 'adam',  # Options: adam, yogi, adagrad, sgd
        'server_lr': 1.0,
        'beta1': 0.9,
        'beta2': 0.999
    }
    
    # Initialize server
    server = FedOptServer(
        feature_dim=config['feature_dim'],
        optimizer_type=config['optimizer_type'],
        server_lr=config['server_lr'],
        beta1=config['beta1'],
        beta2=config['beta2']
    )
    
    # Initialize clients (automatically discover client directories)
    clients = []
    processed_dir = '../shared/test-data/processed'
    
    # Auto-discover client directories
    client_data_dirs = []
    if os.path.exists(processed_dir):
        for item in os.listdir(processed_dir):
            item_path = os.path.join(processed_dir, item)
            if os.path.isdir(item_path) and item.startswith('university_client_'):
                client_data_dirs.append(item_path)
    
    logger.info(f"Found client directories: {client_data_dirs}")
    
    for client_dir in client_data_dirs:
        data_path = f'{client_dir}/data.csv'
        if os.path.exists(data_path):
            client_id = os.path.basename(client_dir).replace('university_client_', '')
            client = FedOptClient(client_id, data_path, config['feature_dim'])
            clients.append(client)
            logger.info(f"Successfully loaded client: {client_id}")
        else:
            logger.warning(f"Client data not found: {data_path}")
    
    logger.info(f"Initialized {len(clients)} clients")
    
    # Training loop
    start_time = time.time()
    
    for round_num in range(config['communication_rounds']):
        logger.info(f"Starting communication round {round_num + 1}/{config['communication_rounds']}")
        
        # Select clients (for simplicity, using all clients)
        selected_clients = clients
        
        # Get current global model state
        global_model_state = server.get_global_model_state()
        
        # Collect client updates
        client_updates = []
        for client in selected_clients:
            try:
                update = client.local_train(global_model_state, config['local_epochs'])
                client_updates.append(update)
            except Exception as e:
                logger.error(f"Error training client {client.client_id}: {e}")
        
        # Server aggregation and update
        server.aggregate_and_update(client_updates)
        
        # Progress update
        if (round_num + 1) % 5 == 0:
            current_accuracy = server.training_history['avg_val_accuracy'][-1] if server.training_history['avg_val_accuracy'] else 0
            logger.info(f"Round {round_num + 1}: Current average accuracy: {current_accuracy:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"FedOpt training completed in {training_time:.2f} seconds")
    
    # Save results
    output_dir = './'
    server.save_results(output_dir)
    
    # Final analysis
    final_accuracy = server.training_history['avg_val_accuracy'][-1] if server.training_history['avg_val_accuracy'] else 0
    final_loss = server.training_history['avg_train_loss'][-1] if server.training_history['avg_train_loss'] else 0
    
    logger.info(f"Final Results - Average Accuracy: {final_accuracy:.4f}, Average Loss: {final_loss:.4f}")
    
    # Heterogeneity analysis summary
    if server.training_history['heterogeneity_metrics']:
        final_het = server.training_history['heterogeneity_metrics'][-1]
        logger.info(f"Data Heterogeneity Analysis:")
        logger.info(f"  - GPA variance across clients: {final_het['gpa_variance_across_clients']:.4f}")
        logger.info(f"  - Performance variance: {final_het['performance_variance']:.4f}")
        logger.info(f"  - Major diversity range: {final_het['major_diversity_range']}")
        logger.info(f"  - Industry diversity range: {final_het['industry_diversity_range']}")
    
    return server.training_history

if __name__ == "__main__":
    try:
        results = run_fedopt_experiment()
        print("FedOpt experiment completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise