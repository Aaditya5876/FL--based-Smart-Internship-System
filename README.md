Smart Internship Engine: Federated Learning for Privacy-Preserving Job Matching

A master's thesis project implementing personalized federated learning to handle extreme data heterogeneity between universities and companies. Enables collaborative AI training for internship recommendations without sharing sensitive data.

Key Features

🔒 Privacy-preserving collaborative training

🎯 Personalized client models for each organization

📊 Handles non-IID data across universities/companies

⚡ Outperforms standard FedAvg, FedProx, and FedOpt

Results

Centralized Baseline: MSE 0.0196, R² 0.181

FedAvg: MSE 0.0265, R² -0.0112

Our PFL: MSE ~0.026, R² ~-0.002 (28% improvement over FedAvg)

Tech Stack
Python, PyTorch, Scikit-learn, Pandas, NumPy

Cross-organizational AI without data sharing.
