# src/algorithms/fedprox/tune.py

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging
from typing import Dict, List, Any

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fedprox_tuning.log'),
        logging.StreamHandler()  # Still show in terminal too
    ]
)

# Fix the Python path to import from the correct location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

try:
    from fedProx_expirement import run_fedprox_experiment, load_your_actual_data
    logging.info("✅ Successfully imported from fedProx_expirement")
except ImportError:
    logging.error("❌ Could not import directly, trying alternative path...")
    try:
        from src.algorithm.fedProx.fedProx_expirement import run_fedprox_experiment, load_your_actual_data
        logging.info("✅ Successfully imported from src.algorithm.fedProx.fedProx_expirement")
    except ImportError as e:
        logging.error(f"❌ Failed to import: {e}")
        sys.exit(1)

class JSONLogger:
    """Custom logger that saves all results to JSON file"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.log_data = {
            "experiment_type": "fedprox_hyperparameter_tuning",
            "start_time": datetime.datetime.now().isoformat(),
            "mu_values": [],
            "results": {},
            "summary": {},
            "logs": []
        }
    
    def log_message(self, level: str, message: str):
        """Log a message with timestamp"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.log_data["logs"].append(log_entry)
        print(f"{level.upper()} - {message}")
    
    def add_result(self, mu: float, result: Dict[str, Any]):
        """Add experiment result for a specific mu value"""
        if mu not in self.log_data["mu_values"]:
            self.log_data["mu_values"].append(mu)
        
        self.log_data["results"][str(mu)] = result
    
    def set_summary(self, summary: Dict[str, Any]):
        """Set the final summary"""
        self.log_data["summary"] = summary
        self.log_data["end_time"] = datetime.datetime.now().isoformat()
        self.log_data["duration_seconds"] = (
            datetime.datetime.fromisoformat(self.log_data["end_time"]) - 
            datetime.datetime.fromisoformat(self.log_data["start_time"])
        ).total_seconds()
    
    def save(self):
        """Save all data to JSON file"""
        filename = os.path.join(self.experiment_dir, 'tuning_experiment_log.json')
        with open(filename, 'w') as f:
            json.dump(self.log_data, f, indent=4, default=str)
        logging.info(f"✅ Full experiment log saved to {filename}")

def run_fedprox_tuning():
    """
    Run FedProx hyperparameter tuning for different mu values
    """
    # 1. Create experiment directory
    tune_experiment_dir = "../../experiments/fedprox_tuning"
    os.makedirs(tune_experiment_dir, exist_ok=True)
    
    # 2. Initialize JSON logger
    json_logger = JSONLogger(tune_experiment_dir)
    json_logger.log_message("info", "🎯 Starting FedProx Hyperparameter Tuning")
    json_logger.log_message("info", "=" * 60)
    
    # 3. Define the mu values to test
    mu_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    json_logger.log_message("info", f"Testing μ values: {mu_values}")
    
    # 4. Load client data
    json_logger.log_message("info", "🔄 Loading client data...")
    client_data = load_your_actual_data()
    
    if not client_data:
        json_logger.log_message("error", "❌ Failed to load client data!")
        json_logger.save()
        return
    
    all_results = {}
    mse_results = []
    r2_results = []
    
    # 5. Loop through each mu value
    for mu in mu_values:
        json_logger.log_message("info", f"\n{'='*50}")
        json_logger.log_message("info", f"🧪 Running FedProx with μ={mu}")
        json_logger.log_message("info", f"{'='*50}")
        
        # Run the experiment
        results, final_mse = run_fedprox_experiment(
            client_data=client_data,
            num_rounds=10,
            mu=mu,
            experiment_name=f"fedprox_mu_{mu}"
        )
        
        if results and final_mse is not None:
            # Store the result
            result_data = {
                'avg_mse': results['final_performance']['avg_mse'],
                'avg_mae': results['final_performance']['avg_mae'],
                'avg_r2': results['final_performance']['avg_r2'],
                'mse_improvement_pct': results['improvements']['mse_improvement_pct'],
                'r2_improvement_pct': results['improvements']['r2_improvement_pct'],
                'successful_clients': results['final_performance']['successful_clients'],
                'successful_evaluations': results['final_performance']['successful_evaluations']
            }
            
            all_results[mu] = result_data
            mse_results.append(results['final_performance']['avg_mse'])
            r2_results.append(results['final_performance']['avg_r2'])
            
            # Save individual result
            exp_dir = os.path.join(tune_experiment_dir, f"mu_{mu}")
            os.makedirs(exp_dir, exist_ok=True)
            
            with open(os.path.join(exp_dir, 'detailed_results.json'), 'w') as f:
                json.dump(results, f, indent=4)
            
            # Add to JSON logger
            json_logger.add_result(mu, result_data)
            
            json_logger.log_message("info", 
                f"✅ μ={mu}: MSE={results['final_performance']['avg_mse']:.6f}, "
                f"R²={results['final_performance']['avg_r2']:.6f}, "
                f"Clients: {results['final_performance']['successful_clients']}/6"
            )
        else:
            json_logger.log_message("error", f"❌ Failed to run experiment for μ={mu}")
            mse_results.append(float('nan'))
            r2_results.append(float('nan'))
            all_results[mu] = {"error": "Experiment failed"}
            json_logger.add_result(mu, {"error": "Experiment failed"})
    
    # 6. Create visualization
    create_tuning_plots(mu_values, mse_results, r2_results, tune_experiment_dir)
    json_logger.log_message("info", "✅ Tuning plots saved")
    
    # 7. Find best mu value and create summary
    valid_indices = [i for i, mse in enumerate(mse_results) if not np.isnan(mse)]
    summary = {}
    
    if valid_indices:
        valid_mse = [mse_results[i] for i in valid_indices]
        valid_mu = [mu_values[i] for i in valid_indices]
        valid_r2 = [r2_results[i] for i in valid_indices]
        
        best_idx = np.argmin(valid_mse)
        best_mu = valid_mu[best_idx]
        best_mse = valid_mse[best_idx]
        best_r2 = valid_r2[best_idx]
        
        summary = {
            "best_mu": best_mu,
            "best_mse": float(best_mse),
            "best_r2": float(best_r2),
            "total_experiments": len(mu_values),
            "successful_experiments": len(valid_indices),
            "failed_experiments": len(mu_values) - len(valid_indices),
            "all_results": all_results,
            "recommendation": f"Use μ={best_mu} for optimal performance (MSE: {best_mse:.6f})"
        }
        
        json_logger.log_message("info", f"\n{'='*60}")
        json_logger.log_message("info", "🎯 TUNING RESULTS SUMMARY")
        json_logger.log_message("info", f"{'='*60}")
        
        for mu, metrics in all_results.items():
            if "error" not in metrics:
                json_logger.log_message("info", 
                    f"μ={mu}: MSE={metrics['avg_mse']:.6f}, R²={metrics['avg_r2']:.6f}, "
                    f"MSE Imp={metrics['mse_improvement_pct']:+.2f}%"
                )
        
        json_logger.log_message("info", f"\n🏆 BEST PARAMETER: μ={best_mu}")
        json_logger.log_message("info", f"   Best MSE: {best_mse:.6f}")
        json_logger.log_message("info", f"   Best R²: {best_r2:.6f}")
        
    else:
        summary = {
            "best_mu": None,
            "best_mse": float('nan'),
            "best_r2": float('nan'),
            "total_experiments": len(mu_values),
            "successful_experiments": 0,
            "failed_experiments": len(mu_values),
            "all_results": all_results,
            "error": "All experiments failed"
        }
        json_logger.log_message("error", "❌ No successful experiments found!")
    
    json_logger.log_message("info", f"{'='*60}")
    
    # 8. Save summary and final log
    with open(os.path.join(tune_experiment_dir, 'tuning_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    json_logger.set_summary(summary)
    json_logger.save()
    
    json_logger.log_message("info", "✅ Tuning complete! Check the JSON files in experiments/fedprox_tuning/")
    
    return summary

def create_tuning_plots(mu_values, mse_results, r2_results, save_dir):
    """Create plots to visualize tuning results"""
    valid_indices = [i for i, mse in enumerate(mse_results) if not np.isnan(mse)]
    
    if not valid_indices:
        return
    
    valid_mu = [mu_values[i] for i in valid_indices]
    valid_mse = [mse_results[i] for i in valid_indices]
    valid_r2 = [r2_results[i] for i in valid_indices]
    
    plt.figure(figsize=(12, 5))
    
    # MSE plot
    plt.subplot(1, 2, 1)
    plt.semilogx(valid_mu, valid_mse, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Proximal Parameter (μ) - Log Scale')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('FedProx: MSE vs μ Parameter')
    plt.grid(True, which="both", ls="--")
    
    # R² plot
    plt.subplot(1, 2, 2)
    plt.semilogx(valid_mu, valid_r2, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Proximal Parameter (μ) - Log Scale')
    plt.ylabel('R² Score')
    plt.title('FedProx: R² vs μ Parameter')
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tuning_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_fedprox_tuning()