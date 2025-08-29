# Centralized Baseline for Smart Internship System
# Purpose: Train a model on ALL data to establish a performance upper bound.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Our model of choice
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Set a random seed for reproducibility (so we get the same result every time)
np.random.seed(42)

# 1. Find all the client data files
data_dir = "../../../data/processed"  # Path to your processed data folder
client_folders = [f for f in os.listdir(data_dir) if f.startswith('university_client_')]

# 2. Load and combine all client data into one big DataFrame
all_data = pd.DataFrame()
for client_folder in client_folders:
    client_file_path = os.path.join(data_dir, client_folder, 'data.csv')
    client_data = pd.read_csv(client_file_path)
    all_data = pd.concat([all_data, client_data], ignore_index=True)

print(f"Combined data from {len(client_folders)} clients.")
print(f"Total dataset size: {all_data.shape}")
print(all_data.head())

# 3. Separate Features (X) and Target (y)
# Assume the target column is called 'match_score'
X = all_data.drop('match_score', axis=1)  # Features: everything EXCEPT the match_score
y = all_data['match_score']               # Target: the match_score itself

# 4. Handle categorical data (a crucial step!)
# We'll use One-Hot Encoding for categorical features (like 'major', 'industry')
X = pd.get_dummies(X)

# 5. Split the data into Training and Testing sets
# We train on 80% of the data, and test on the remaining 20% to see how well we generalized.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# 6. Initialize and train the model
print("Training Centralized Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Make predictions on the test set (the unseen data)
y_pred = model.predict(X_test)

# 8. Evaluate the model's performance
centralized_mse = mean_squared_error(y_test, y_pred)
centralized_r2 = r2_score(y_test, y_pred)

print("--- Centralized Baseline Results ---")
print(f"Mean Squared Error (MSE): {centralized_mse:.6f}")
print(f"R² Score: {centralized_r2:.6f}")

# 9. Create a scatter plot to visualize predictions vs. reality
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Match Score')
plt.ylabel('Predicted Match Score')
plt.title('Centralized Model: Actual vs. Predicted')
# Plot a perfect prediction line
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--') # Red dashed line
plt.savefig('centralized_baseline_scatter_plot.png')
plt.show()

# 10. (Optional) Save the results to a JSON file for later comparison
import json
results = {
    "centralized_mse": centralized_mse,
    "centralized_r2": centralized_r2
}
with open('centralized_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results and plot saved.")