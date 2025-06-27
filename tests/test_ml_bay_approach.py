from kmerml.ml.pipeline import PhylogeneticPipeline
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import os
import gc
from kmerml.utils.memory_mapping import create_shared_kmer_dataset, load_shared_kmer_data

# Import skopt for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Set up base directories
base_dir = "data/results/bayesian_opt"
feature_dir = "data/processed/features"
Path(base_dir).mkdir(parents=True, exist_ok=True)

# Optional reference tree
reference_tree = None  # "data/reference/phylogeny.nwk" if you have one

# Define parameter space
param_space = [
    Categorical(["shannon_entropy", "count", "gc_percent"], name="metric"),
    Categorical(["random_forest", "mutual_info", "correlation"], name="feature_selection_method"),
    Categorical(["cosine", "euclidean", "correlation"], name="distance_metric"),
    Categorical(["upgma", "single", "complete", "average"], name="clustering_method")
]

# Names for parameters
param_names = ["metric", "feature_selection_method", "distance_metric", "clustering_method"]

# Prepare shared k-mer data (reuse from your grid search)
print("Preparing shared k-mer data...")
shared_data = {}
for metric in ["shannon_entropy", "count", "gc_percent"]:
    shared_data[metric] = create_shared_kmer_dataset(
        feature_dir=feature_dir,
        k_value=10,
        metric=metric
    )

# Track all results for visualization
all_results = []

# Define objective function for optimization
@use_named_args(param_space)
def objective_function(**params):
    """Evaluate pipeline performance for a parameter combination"""
    # Create a unique run ID
    run_id = "_".join([f"{k}-{v}" for k, v in params.items()])
    run_id = run_id.replace(" ", "_")
    
    # Create output directory for this run
    output_dir = f"{base_dir}/{run_id}"
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\nTesting parameters: {params}")
    
    try:
        # Initialize pipeline
        pipeline = PhylogeneticPipeline(
            feature_dir=feature_dir,
            output_dir=output_dir,
            reference_tree=reference_tree
        )
        
        # Use shared k-mer data
        start_time = time.time()
        
        # Get the shared data for this metric
        metric_data = shared_data.get(params['metric'])
        
        if metric_data:
            # Load the shared data
            feature_matrix = load_shared_kmer_data(metric_data)
            
            # Add prefix to column names
            k = 10  # We're using k=10 for all runs
            feature_matrix.columns = [f"k{k}_{col}" for col in feature_matrix.columns]
            
            # Run pipeline with the pre-loaded feature matrix
            run_results = pipeline._run_with_features(
                feature_matrices=[feature_matrix],
                n_features=500,
                feature_selection_method=params['feature_selection_method'],
                clustering_method=params['clustering_method'],
                distance_metric=params['distance_metric']
            )
        else:
            # Fallback to standard loading
            run_results = pipeline.run(
                metric=params['metric'],
                k_values=[10],
                feature_selection_method=params['feature_selection_method'],
                distance_metric=params['distance_metric'],
                clustering_method=params['clustering_method'],
                n_features=500
            )
            
        runtime = time.time() - start_time
        
        # Calculate evaluation metrics
        score = None
        
        # Calculate cophenetic correlation if available
        try:
            if 'linkage_matrix' in run_results and 'distance_matrix' in run_results:
                c, _ = cophenet(run_results['linkage_matrix'], pdist(run_results['distance_matrix']))
                score = c
                print(f"Cophenetic correlation: {c:.4f}")
        except Exception as e:
            print(f"Error calculating cophenetic correlation: {e}")
            score = 0.0
        
        # Add reference tree comparison if available
        tree_metrics = {}
        if 'tree_comparison' in run_results and run_results['tree_comparison']:
            for metric_name, value in run_results['tree_comparison'].items():
                tree_metrics[f'tree_{metric_name}'] = value
                
                # Robinson-Foulds distance is a better metric if available
                if metric_name == 'robinson_foulds':
                    # Convert to similarity (1 / (1+RF)) as we want to maximize
                    score = 1.0 / (1.0 + value)
                    print(f"Tree similarity (1/(1+RF)): {score:.4f}")
        
        # If no metric available, use negative runtime
        if score is None:
            score = -runtime / 3600.0  # Convert to hours for better scaling
            print(f"Using runtime as score: {-score:.4f} hours")
            
        # Store all results for later analysis
        result_entry = {
            **params,
            'runtime': runtime,
            'cophenetic_corr': score if 'c' in locals() else None,
            **tree_metrics,
            'run_id': run_id
        }
        all_results.append(result_entry)
        
        # Clean up memory
        gc.collect()
        
        # For Bayesian optimization, we want to maximize, so return negative if needed
        return -score if score < 0 else -score
            
    except Exception as e:
        print(f"Error evaluating parameters: {e}")
        # Return a poor score for failed runs
        return 1.0  # Assuming score is in [0, 1] range with 0 being best
    
# Run Bayesian optimization
n_calls = 25  # Number of parameter combinations to try
random_state = 42

print(f"Starting Bayesian optimization with {n_calls} iterations...")
result = gp_minimize(
    objective_function,
    param_space,
    n_calls=n_calls,
    n_random_starts=10,  # Try 10 random combinations before using the model
    random_state=random_state,
    verbose=True
)

# Print optimization results
print("\n===== OPTIMIZATION RESULTS =====")
print(f"Best parameters found:")
for i, param_name in enumerate(param_names):
    print(f"  {param_name}: {result.x[i]}")
print(f"Best score: {-result.fun:.4f}")

# Save all results
results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{base_dir}/bayesian_opt_results.csv", index=False)
print(f"Saved all results to {base_dir}/bayesian_opt_results.csv")

# Create convergence plot
plt.figure(figsize=(10, 6))
plot_convergence(result)
plt.savefig(f"{base_dir}/convergence_plot.png")

# Create objective plots for each parameter
plt.figure(figsize=(15, 10))
plot_objective(result, n_points=10)
plt.savefig(f"{base_dir}/objective_plots.png")

# Create parameter importance plot
try:
    from skopt.plots import plot_evaluations
    plt.figure(figsize=(15, 10))
    plot_evaluations(result)
    plt.savefig(f"{base_dir}/parameter_evaluations.png")
except:
    print("Could not create parameter evaluations plot")

print("\nBayesian optimization complete!")