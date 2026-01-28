import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from optimization_analytics import IntegratedCarbonOptimizer
from model_manager import ModelManager


def run_mc_distribution_experiment(sim_count=5000):
    # Load model
    json_path = os.path.join(project_root, 'data', 'model_template.json')
    manager = ModelManager()
    manager.load_complete_model(json_path)
    
    optimizer = IntegratedCarbonOptimizer(manager=manager)
    
    # Run optimization
    result = optimizer.optimize_with_uncertainty(
        method='SLSQP',
        mc_simulations=sim_count,
        use_saa=True,
        saa_samples=200
    )
    
    if not result.success:
        print(f"Optimization failed: {result.message}")
        return None

    base = result.mc_baseline
    opt = result.mc_optimized
    
    # Calculate metrics
    metrics = {
        "Mean Reduction (tonnes)": base['mean'] - opt['mean'],
        "Mean Reduction (%)": (base['mean'] - opt['mean']) / base['mean'] * 100,
        "Std Ratio (Opt/Base)": opt['std'] / base['std'],
        "P95 Improvement (%)": (base['p95'] - opt['p95']) / base['p95'] * 100,
        "CV Change (pp)": opt['cv'] - base['cv']
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "validation_outputs")

    # Print results
    for key, val in metrics.items():
        print(f"{key:<30} {val:>15.4f}")
    
    # Generate plot
    generate_shift_plot(base, opt, output_dir)
    
    return metrics

def generate_shift_plot(base, opt, output_dir):
    """Simple dual density histogram for paper."""
    plt.figure(figsize=(8, 5))
    
    plt.hist(base['data'], bins=50, alpha=0.6, label='Baseline', density=True)
    plt.hist(opt['data'], bins=50, alpha=0.6, label='Optimized', density=True)
    
    plt.axvline(base['mean'], color='C0', linestyle='--', linewidth=1.5)
    plt.axvline(opt['mean'], color='C1', linestyle='--', linewidth=1.5)
    
    plt.xlabel("Emissions (tonnes CO2e)")
    plt.ylabel("Density")
    plt.title("Monte Carlo Distribution: Baseline vs Optimized")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "mc_distribution_shift.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Plot saved: {out_path}")

if __name__ == "__main__":
    run_mc_distribution_experiment()