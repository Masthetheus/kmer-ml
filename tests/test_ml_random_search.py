from kmerml.ml.pipeline import PhylogeneticPipeline
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import os
import gc
from kmerml.utils.memory_mapping import create_shared_kmer_dataset, load_shared_kmer_data

# Set up base directories
base_dir = "data/results/random_search"
feature_dir = "data/processed/features"
Path(base_dir).mkdir(parents=True, exist_ok=True)

# Optional reference tree
reference_tree = None  # "data/reference/phylogeny.nwk" if you have one

# Define parameter grid
param_grid = {
    'metric': ["shannon_entropy", "count", "gc_percent"],
    'feature_selection_method': ["random_forest", "mutual_info", "correlation"],
    'distance_metric': ["cosine", "euclidean", "correlation"],
    'clustering_method': ["upgma", "single", "complete", "average"]
}

# Parameters for random search
max_iterations = 25       # Maximum number of combinations to try
early_stopping_threshold = 0.95  # Stop if we find a combination with score > this
patience = 5              # Continue for this many iterations after finding a good result
                          # to see if we can find an even better one

# Prepare shared k-mer data
print("Preparing shared k-mer data...")
shared_data = {}
for metric in param_grid['metric']:
    shared_data[metric] = create_shared_kmer_dataset(
        feature_dir=feature_dir,
        k_value=10,
        metric=metric
    )

# Track results
results = []
best_score = 0.0
iterations_since_improvement = 0
best_params = None

def evaluate_parameters(params):
    """Evaluate a set of parameters and return score and metrics"""
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
        eval_metrics = {
            'runtime': runtime,
            'feature_count': run_results.get('feature_count', 0),
            'organism_count': run_results.get('organism_count', 0)
        }
        
        # Calculate cophenetic correlation if available
        score = None
        try:
            if 'linkage_matrix' in run_results and 'distance_matrix' in run_results:
                c, _ = cophenet(run_results['linkage_matrix'], pdist(run_results['distance_matrix']))
                eval_metrics['cophenetic_corr'] = c
                score = c
                print(f"Cophenetic correlation: {c:.4f}")
        except Exception as e:
            eval_metrics['cophenetic_corr'] = None
            
        # Add reference tree comparison if available
        if 'tree_comparison' in run_results and run_results['tree_comparison']:
            for metric_name, value in run_results['tree_comparison'].items():
                eval_metrics[f'tree_{metric_name}'] = value
                
                # Robinson-Foulds distance is a better metric if available
                if metric_name == 'robinson_foulds':
                    # Convert to similarity (1 / (1+RF)) as we want to maximize
                    score = 1.0 / (1.0 + value)
                    print(f"Tree similarity (1/(1+RF)): {score:.4f}")
        
        # If no metric available, use normalized runtime (lower is better)
        if score is None and runtime > 0:
            # Normalize runtime to be between 0 and 1 (assuming max runtime of 1 hour)
            score = max(0, 1 - (runtime / 3600.0))
            print(f"Using normalized runtime as score: {score:.4f}")
            
        # Clean up memory
        gc.collect()
        
        # Return the score and metrics
        result_entry = {**params, **eval_metrics, 'score': score, 'run_id': run_id}
        return score, result_entry
            
    except Exception as e:
        print(f"Error evaluating parameters: {e}")
        # Return a poor score for failed runs
        return 0.0, {**params, 'run_id': run_id, 'error': str(e), 'score': 0.0}

# Run random search with early stopping
print(f"Starting random search with up to {max_iterations} iterations...")
iteration = 0

while iteration < max_iterations:
    # Sample random parameters
    params = {k: random.choice(v) for k, v in param_grid.items()}
    
    # Evaluate this combination
    score, result_entry = evaluate_parameters(params)
    results.append(result_entry)
    
    # Track best score
    if score > best_score:
        best_score = score
        best_params = params
        iterations_since_improvement = 0
        print(f"New best score: {best_score:.4f} with parameters: {best_params}")
    else:
        iterations_since_improvement += 1
    
    # Check early stopping condition
    if best_score > early_stopping_threshold and iterations_since_improvement >= patience:
        print(f"Early stopping triggered after {iteration+1} iterations!")
        print(f"Found good solution (score: {best_score:.4f}) and no improvement for {patience} iterations.")
        break
        
    iteration += 1

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f"{base_dir}/random_search_results.csv", index=False)
print(f"Saved all results to {base_dir}/random_search_results.csv")

# Create visualizations
print("Generating visualizations...")
try:
    # Plot performance over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(results)+1), [r['score'] for r in results], 'bo-')
    plt.axhline(y=best_score, color='r', linestyle='--', label=f'Best score: {best_score:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Performance Score')
    plt.title('Random Search Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_dir}/search_progress.png")
    
    # Plot distribution of scores by parameter
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, param_name in enumerate(param_grid.keys()):
        ax = axes[i]
        param_values = param_grid[param_name]
        param_scores = []
        
        for value in param_values:
            value_scores = [r['score'] for r in results if r[param_name] == value and 'score' in r]
            if value_scores:
                param_scores.append((value, np.mean(value_scores)))
        
        param_scores.sort(key=lambda x: x[1], reverse=True)
        
        values = [x[0] for x in param_scores]
        scores = [x[1] for x in param_scores]
        
        ax.bar(values, scores)
        ax.set_title(f'Performance by {param_name}')
        ax.set_ylabel('Average Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Highlight best value
        best_value = best_params[param_name]
        best_idx = values.index(best_value) if best_value in values else None
        if best_idx is not None:
            ax.get_children()[best_idx].set_color('green')
            
    plt.tight_layout()
    plt.savefig(f"{base_dir}/parameter_performance.png")
    
except Exception as e:
    print(f"Error creating visualizations: {e}")

# Print best parameters
print("\n===== BEST PARAMETERS FOUND =====")
print(f"Best score: {best_score:.4f}")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print("\nRandom search complete!")