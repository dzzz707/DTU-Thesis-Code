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


def run_optimizer_stability_test():
    manager = ModelManager()
    json_path = os.path.join(data_path, 'model_template.json')
    manager.load_complete_model(json_path)
    
    optimizer = IntegratedCarbonOptimizer(manager=manager)
    
    methods = ['SLSQP', 'COBYLA', 'TRUST_CONSTR']
    num_runs = 20
    sample_n = 500

    all_raw_data = []
    summary_metrics = []

    for method in methods:
        print(f"Testing {method}...")
        result = optimizer.optimize_with_uncertainty(
            method=method,
            saa_samples=sample_n,
            use_saa=True,
            stability_runs=num_runs
        )
        
        if result.success and result.stability_analysis:
            stats = result.stability_analysis
            for reduction in stats['all_reductions']:
                all_raw_data.append({
                    'Optimizer': method, 
                    'Reduction_Pct': reduction
                })
            
            summary_metrics.append({
                'Optimizer': method,
                'Mean': stats['mean_reduction'],
                'Std': stats['std_reduction'],
                'CV': stats['cv']
            })
            print(f"  Mean: {stats['mean_reduction']:.2f}%, CV: {stats['cv']:.4f}")

    df_raw = pd.DataFrame(all_raw_data)
    df_summary = pd.DataFrame(summary_metrics)
    
    print("\nSummary:")
    print(df_summary.to_string(index=False))
    
    # Output directory
    output_dir = os.path.join(current_dir, "validation_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    generate_stability_plot(df_raw, df_summary, methods, output_dir)
    return df_raw, df_summary


def generate_stability_plot(df_raw, df_summary, methods, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = [df_raw[df_raw['Optimizer'] == m]['Reduction_Pct'].values for m in methods]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    box = ax.boxplot(plot_data, tick_labels=methods, patch_artist=True, widths=0.5)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
    
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # Scatter points
    for i, (method, color) in enumerate(zip(methods, colors)):
        data = df_raw[df_raw['Optimizer'] == method]['Reduction_Pct'].values
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, color=color, alpha=0.5, s=15)
    
    ax.set_ylabel('Emission Reduction (%)')
    ax.set_xlabel('Optimization Method')
    ax.set_title('Optimizer Stability Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "optimizer_stability.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    run_optimizer_stability_test()