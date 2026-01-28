import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
app_path = os.path.join(project_root, 'app')
data_path = os.path.join(project_root, 'data')
sys.path.append(project_root)
sys.path.append(app_path)

from optimization_analytics import IntegratedCarbonOptimizer
from model_manager import ModelManager


def load_model_manager(json_path: str) -> ModelManager:
    manager = ModelManager()
    success, message, stats = manager.load_complete_model(json_path)
    
    if not success:
        raise RuntimeError(f"Failed to load model: {message}")
    
    print(f"Model loaded: {stats['models']} models, {stats['formulas']} formulas, {stats['parameters']} parameters")
    return manager


def run_accuracy_verification(sample_size: int = 1000, random_seed: int = 42):
    np.random.seed(random_seed)
    
    # Load model from JSON
    json_path = os.path.join(data_path, 'model_template.json')
    
    if not os.path.exists(json_path):
        print(f"Error: Model template not found at {json_path}")
        print("Please ensure model_template.json exists in the data folder.")
        return
    
    print(f"Loading model from: {json_path}")
    manager = load_model_manager(json_path)
    
    # Initialize optimizer with ModelManager
    optimizer = IntegratedCarbonOptimizer(manager=manager)
    
    # Build fast calculator (linearization)
    print("\nBuilding Fast Calculator (linear approximation)...")
    optimizer._build_fast_calculator()
    
    # Get optimizable parameters
    optimizable_params = [p for p in optimizer.parameters.values() if p.is_optimizable]
    print(f"Found {len(optimizable_params)} optimizable parameters")
    
    if len(optimizable_params) == 0:
        print("Error: No optimizable parameters found!")
        return
    
    # Print parameter info
    print("\nOptimizable Parameters:")
    print("-" * 70)
    for p in optimizable_params:
        print(f"  {p.stub:30s}: [{p.optimization_min:12.2f}, {p.optimization_max:12.2f}] {p.unit}")
    print("-" * 70)
    
    # Run accuracy test
    relative_errors = []
    exact_values = []
    fast_values = []
    
    print(f"\nStarting verification with {sample_size} random samples...")
    
    for i in range(sample_size):
        # Generate random parameter values within bounds
        random_test_input = {}
        for p in optimizable_params:
            random_test_input[p.stub] = np.random.uniform(
                p.optimization_min, 
                p.optimization_max
            )
        
        # Calculate using both methods
        exact_value = optimizer.calculate_emissions(random_test_input)['total']
        fast_value = optimizer._calculate_emissions_fast(random_test_input)
        
        # Compute relative error
        if exact_value > 0:
            error_pct = abs(fast_value - exact_value) / exact_value * 100
        else:
            error_pct = 0.0
            
        relative_errors.append(error_pct)
        exact_values.append(exact_value)
        fast_values.append(fast_value)
        
        # Progress indicator
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{sample_size} samples...")
    
    # Calculate statistics
    mre = np.mean(relative_errors)
    max_err = np.max(relative_errors)
    std_dev = np.std(relative_errors)
    median_err = np.median(relative_errors)
    p95_err = np.percentile(relative_errors, 95)
    p99_err = np.percentile(relative_errors, 99)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 6.3.1: LINEARIZATION ACCURACY RESULTS")
    print("=" * 60)
    print(f"Samples Tested:           {sample_size}")
    print(f"Random Seed:              {random_seed}")
    print("-" * 60)
    print(f"Mean Relative Error:      {mre:.8f}%  (Criterion: < 0.5%)")
    print(f"Max Relative Error:       {max_err:.8f}%  (Criterion: < 2.0%)")
    print(f"Median Relative Error:    {median_err:.8f}%")
    print(f"Std Deviation:            {std_dev:.8f}%")
    print(f"95th Percentile Error:    {p95_err:.8f}%")
    print(f"99th Percentile Error:    {p99_err:.8f}%")
    print("-" * 60)
    
    # Validation result
    mre_pass = mre < 0.5
    max_pass = max_err < 2.0
    
    print(f"MRE < 0.5%:               {'PASS' if mre_pass else 'FAIL'}")
    print(f"Max Error < 2.0%:         {'PASS' if max_pass else 'FAIL'}")
    
    if mre_pass and max_pass:
        print("Fast Calculator accuracy is ACCEPTABLE")
    else:
        print("Fast Calculator accuracy NEEDS REVIEW")
    
    # Generate plot
    output_dir = os.path.join(current_dir, "validation_outputs")
    generate_error_distribution_plot(relative_errors, mre, output_dir)
    
    # Return results for further analysis
    return {
        'sample_size': sample_size,
        'mre': mre,
        'max_error': max_err,
        'std_dev': std_dev,
        'median_error': median_err,
        'p95_error': p95_err,
        'p99_error': p99_err,
        'all_errors': relative_errors,
        'exact_values': exact_values,
        'fast_values': fast_values,
        'pass': mre_pass and max_pass
    }


def generate_error_distribution_plot(relative_errors, mre, output_dir):
    plt.figure(figsize=(10, 6))
    
    n, bins, patches = plt.hist(
        relative_errors, 
        bins=50, 
        color='forestgreen', 
        edgecolor='black', 
        alpha=0.7,
        density=False
    )
    
    plt.axvline(
        mre, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean Relative Error: {mre:.6f}%'
    )
    
    plt.axvline(0.5, color='orange', linestyle=':', linewidth=1.5, label='MRE Threshold: 0.5%')
    plt.title("Distribution of Linearization Errors", fontsize=14)
    plt.xlabel("Relative Error (%)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "fast_calculator_accuracy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    results = run_accuracy_verification(sample_size=1000, random_seed=42)