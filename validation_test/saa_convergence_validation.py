import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
app_path = os.path.join(project_root, 'app')
data_path = os.path.join(project_root, 'data')
sys.path.append(project_root)
sys.path.append(app_path)

from optimization_analytics import IntegratedCarbonOptimizer
from model_manager import ModelManager

BASE_SEED = 42


def run_saa_convergence_analysis():
    manager = ModelManager()
    json_path = os.path.join(data_path, 'model_template.json')
    success, message, _ = manager.load_complete_model(json_path)
    
    if not success:
        print(f"Error loading model: {message}")
        return

    sample_sizes = [25, 50, 100, 200, 500, 750, 1000]
    runs_per_n = 10
    
    results_list = []

    for n in sample_sizes:
        print(f"Processing N = {n}...")
        for run_idx in range(runs_per_n):
            np.random.seed(BASE_SEED + n * 100 + run_idx)
            optimizer = IntegratedCarbonOptimizer(manager=manager)
            
            result = optimizer.optimize_with_uncertainty(
                method='SLSQP',
                saa_samples=n,
                use_saa=True,
                stability_runs=1
            )
            
            if result.success:
                results_list.append({
                    'N': n,
                    'Objective_Value': result.saa_objective_value,
                    'Reduction_Pct': result.reduction_percent
                })

    df = pd.DataFrame(results_list)
    
    summary = df.groupby('N').agg({
        'Objective_Value': ['mean', 'std']
    }).reset_index()
    summary.columns = ['N', 'Obj_Mean', 'Obj_Std']
    
    print("\nSummary:")
    print(summary.to_string(index=False))
    
    # Output directory
    output_dir = os.path.join(current_dir, "validation_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    generate_convergence_plot(summary, output_dir)
    return df, summary


def generate_convergence_plot(summary, output_dir):
    """Clean convergence plot for thesis."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Main curve with default blue color
    ax.plot(summary['N'], summary['Obj_Mean'], 
            marker='o', markersize=6, linewidth=1.5)
    
    # Converged reference line
    converged_value = summary['Obj_Mean'].iloc[-1]
    ax.axhline(y=converged_value, color='red', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Sample Size (N)')
    ax.set_ylabel('Objective Value (kg CO2e)')
    ax.set_title('SAA Convergence Analysis')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "saa_convergence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    run_saa_convergence_analysis()