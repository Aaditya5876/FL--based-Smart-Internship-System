"""
Federated Averaging (FedAvg) - 10 Rounds Demo
Dataset: job recommendation (your provided data.csv)
Goal: Run FedAvg for 10 rounds to improve model performance
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from datetime import datetime
import json

# ---------------------------
# 0. Setup Logging
# ---------------------------
def setup_logging():
    """Configure logging to both console and file"""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/fedavg_10rounds_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ---------------------------
# 1. Data Preprocessing
# ---------------------------
def preprocess_client_data(data_path: str, logger):
    """Load and preprocess data.csv inside a client folder"""
    df = pd.read_csv(os.path.join(data_path, "data.csv"))

    # Drop non-useful identifiers
    drop_cols = ["user_id", "job_id", "name", "skills", 
                 "company_id", "title", "role", "location_y",
                 "skills_required", "company_name", "industry",
                 "location_x", "university"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode categorical variables
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Features = all except target
    X = df.drop(columns=["match_score"]).values
    y = df["match_score"].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logger.info(f"Processed data from {os.path.basename(data_path)}: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, scaler

# ---------------------------
# 2. Simple Neural Network
# ---------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# 3. Local Training (per client)
# ---------------------------
def local_train(model, X, y, epochs=3, lr=0.01, logger=None, client_id=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if logger and epoch % 1 == 0:
            logger.info(f"Client {client_id} - Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model.state_dict()

# ---------------------------
# 4. Federated Averaging
# ---------------------------
def fed_avg(models, logger):
    """Average model parameters"""
    avg_state = {}
    for key in models[0].keys():
        avg_state[key] = sum([m[key] for m in models]) / len(models)
    return avg_state

# ---------------------------
# 5. Evaluation
# ---------------------------
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy().flatten()
    
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    return mse, mae, r2, preds

# ---------------------------
# 6. Main FedAvg Loop (10 rounds)
# ---------------------------
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    logger.info("Starting FedAvg with 10 Rounds")
    
    # Find all client directories
    processed_dir = "../../../data/processed"
    client_dirs = []
    
    for item in os.listdir(processed_dir):
        item_path = os.path.join(processed_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "data.csv")):
            client_dirs.append(item_path)
    
    logger.info(f"Found {len(client_dirs)} clients: {[os.path.basename(d) for d in client_dirs]}")
    
    if not client_dirs:
        logger.error("No client directories found with data.csv")
        exit(1)

    # Preprocess data for all clients
    client_data = []
    for cdir in client_dirs:
        try:
            X, y, _ = preprocess_client_data(cdir, logger)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            client_data.append((X_train, X_test, y_train, y_test, os.path.basename(cdir)))
        except Exception as e:
            logger.error(f"Error processing {cdir}: {e}")

    if not client_data:
        logger.error("No client data processed successfully")
        exit(1)

    input_dim = client_data[0][0].shape[1]
    logger.info(f"Input dimension: {input_dim}")

    # Initialize global model
    global_model = SimpleNN(input_dim)
    logger.info("Initialized global model")

    # Store results for all rounds
    all_rounds_results = []
    
    # 10 Rounds of Federated Learning
    num_rounds = 10
    for round_num in range(num_rounds):
        logger.info(f"\n{'='*50}")
        logger.info(f"FEDERATED ROUND {round_num + 1}/{num_rounds}")
        logger.info(f"{'='*50}")
        
        # Local training on each client
        local_states = []
        for i, (X_train, X_test, y_train, y_test, client_name) in enumerate(client_data):
            logger.info(f"Training client {i+1}/{len(client_data)} ({client_name})...")
            local_model = SimpleNN(input_dim)
            local_model.load_state_dict(global_model.state_dict())
            state = local_train(local_model, X_train, y_train, logger=logger, client_id=client_name)
            local_states.append(state)

        # Federated Averaging
        global_state = fed_avg(local_states, logger)
        global_model.load_state_dict(global_state)
        logger.info("Global model updated with averaged parameters")

        # Evaluate after this round
        round_results = {"round": round_num + 1, "clients": []}
        
        logger.info("Evaluation Results:")
        for i, (X_train, X_test, y_train, y_test, client_name) in enumerate(client_data):
            mse, mae, r2, preds = evaluate(global_model, X_test, y_test)
            logger.info(f"📊 Client {client_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            round_results["clients"].append({
                "client": client_name,
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "samples": len(y_test)
            })
        
        all_rounds_results.append(round_results)
        logger.info(f"Completed Round {round_num + 1}/{num_rounds}")

    # Calculate and log final results
    logger.info(f"\n{'='*50}")
    logger.info("FINAL RESULTS AFTER 10 ROUNDS")
    logger.info(f"{'='*50}")
    
    final_results = all_rounds_results[-1]  # Results from the last round
    weighted_mse = weighted_mae = weighted_r2 = 0
    total_samples = 0
    
    for client_result in final_results["clients"]:
        client_weight = client_result["samples"]
        weighted_mse += client_result["mse"] * client_weight
        weighted_mae += client_result["mae"] * client_weight
        weighted_r2 += client_result["r2"] * client_weight
        total_samples += client_weight
    
    weighted_mse /= total_samples
    weighted_mae /= total_samples
    weighted_r2 /= total_samples
    
    logger.info(f"Overall Weighted Metrics:")
    logger.info(f"MSE: {weighted_mse:.4f}")
    logger.info(f"MAE: {weighted_mae:.4f}")
    logger.info(f"R²: {weighted_r2:.4f}")

    # Save comprehensive results to JSON file
    results_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rounds": num_rounds,
        "clients": [os.path.basename(d) for d in client_dirs],
        "round_results": all_rounds_results,
        "final_metrics": {
            "weighted_mse": float(weighted_mse),
            "weighted_mae": float(weighted_mae),
            "weighted_r2": float(weighted_r2),
            "total_samples": total_samples
        }
    }
    
    results_file = f"logs/fedavg_10rounds_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Detailed results saved to {results_file}")
    logger.info("FedAvg with 10 Rounds completed successfully!")