"""
Experiment 6.4.2: Parameter Sensitivity Ranking Analysis
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from optimization_analytics import IntegratedCarbonOptimizer
from model_manager import ModelManager


def run_sensitivity_experiment():
    # Load model
    json_path = os.path.join(project_root, 'data', 'model_template.json')
    manager = ModelManager()
    manager.load_complete_model(json_path)
    
    optimizer = IntegratedCarbonOptimizer(manager=manager)
    
    # Run optimization
    result = optimizer.optimize_with_uncertainty(
        method='SLSQP',
        mc_simulations=5000,
        use_saa=True
    )
    
    if not result.success or not result.sensitivity_ranking:
        print("Experiment failed: No sensitivity data.")
        return None

    # Enrich with scope from optimizer.parameters
    sensitivity_data = []
    for item in result.sensitivity_ranking:
        stub = item.get('stub', '')
        scope = optimizer.parameters[stub].scope if stub in optimizer.parameters else 'N/A'
        sensitivity_data.append({
            'parameter': item['parameter'],
            'stub': stub,
            'correlation': item['correlation'],
            'abs_correlation': item['abs_correlation'],
            'scope': scope
        })
    
    top_params = sensitivity_data[:10]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "validation_outputs")

    # Print results
    print(f"{'Rank':<5} {'Parameter':<35} {'Correlation':>12} {'Scope':<10}")
    for i, p in enumerate(top_params, 1):
        print(f"{i:<5} {p['parameter']:<35} {p['correlation']:>+12.4f} {p['scope']:<10}")
    
    # Generate plot
    generate_tornado_chart(top_params, output_dir)
    
    return top_params


def generate_tornado_chart(sensitivity_data, output_dir):
    """Simple tornado chart for paper."""
    df = pd.DataFrame(sensitivity_data).sort_values('abs_correlation', ascending=True)
    
    plt.figure(figsize=(8, 6))
    
    colors = ['C3' if x > 0 else 'C0' for x in df['correlation']]
    plt.barh(df['parameter'], df['correlation'], color=colors, alpha=0.8)
    plt.axvline(x=0, color='black', linewidth=0.8)
    
    plt.xlabel("Correlation Coefficient")
    plt.title("Parameter Sensitivity Ranking (Pearson Correlation)")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "sensitivity_tornado.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    run_sensitivity_experiment()