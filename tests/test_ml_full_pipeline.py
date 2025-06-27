from kmerml.ml.pipeline import PhylogeneticPipeline
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import itertools
import seaborn as sns
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import os
import concurrent.futures
from tqdm import tqdm
from kmerml.utils.tree import *
from kmerml.utils.memory_mapping import create_shared_kmer_dataset, load_shared_kmer_data
# Set up base directories
base_dir = "data/results/grid_search"
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

# Track results
results = []

# Optional: Reduce search space (comment out for full grid search)
# Uncomment if you want to reduce number of combinations
"""
param_grid = {
    'metric': ["shannon_entropy", "count"],  # Reduced from 3 to 2
    'feature_selection_method': ["random_forest", "correlation"],  # Reduced from 3 to 2
    'distance_metric': ["cosine", "euclidean"],  # Reduced from 3 to 2
    'clustering_method': ["upgma", "average"]  # Reduced from 4 to 2
}
"""

# Calculate total combinations
total_combinations = np.prod([len(values) for values in param_grid.values()])
print(f"Running grid search with {total_combinations} parameter combinations")

# Create parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

print("Preparing shared k-mer data...")
shared_kmer_info = create_shared_kmer_dataset(
    feature_dir=feature_dir,
    k_value=10,
    metric="shannon_entropy"  # Start with one metric
)

# For each metric, create shared data
shared_data = {}
for metric in param_grid['metric']:
    shared_data[metric] = create_shared_kmer_dataset(
        feature_dir=feature_dir,
        k_value=10,
        metric=metric
    )

# Modify the process_parameter_combination function
def process_parameter_combination(combination_data):
    i, combination = combination_data
    
    # Convert combination to parameter dictionary
    params = dict(zip(param_names, combination))
    
    # Create identifier for this run
    run_id = "_".join([f"{k}-{v}" for k, v in params.items()])
    run_id = run_id.replace(" ", "_")
    
    # Create output directory for this run
    output_dir = f"{base_dir}/{run_id}"
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Initialize pipeline
        pipeline = PhylogeneticPipeline(
            feature_dir=feature_dir,
            output_dir=output_dir,
            reference_tree=reference_tree
        )
        
        # MEMORY-OPTIMIZED: Use shared k-mer data
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
                feature_matrices=[feature_matrix],  # Wrap in list for compatibility
                n_features=500,
                feature_selection_method=params['feature_selection_method'],
                clustering_method=params['clustering_method'],
                distance_metric=params['distance_metric']
            )
        else:
            # Fallback to standard loading if shared data isn't available
            run_results = pipeline.run(
                metric=params['metric'],
                k_values=[10],  # Use only k=10 to save memory
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
        try:
            if 'linkage_matrix' in run_results and 'distance_matrix' in run_results:
                c, _ = cophenet(run_results['linkage_matrix'], pdist(run_results['distance_matrix']))
                eval_metrics['cophenetic_corr'] = c
        except Exception as e:
            eval_metrics['cophenetic_corr'] = None
        
        # Add reference tree comparison if available
        if 'tree_comparison' in run_results and run_results['tree_comparison']:
            for metric_name, value in run_results['tree_comparison'].items():
                eval_metrics[f'tree_{metric_name}'] = value
        
        # Clean up memory explicitly
        import gc
        gc.collect()
        
        # Return results
        result_entry = {**params, **eval_metrics, 'run_id': run_id}
        return result_entry
            
    except Exception as e:
        # Return error information
        return {
            **params, 
            'run_id': run_id,
            'error': str(e),
            'runtime': float('inf')
        }

# Reduce number of parallel workers to lower memory usage
n_workers = min(os.cpu_count(), 2)  # Use fewer workers to save memory
print(f"Running grid search with {total_combinations} parameter combinations using {n_workers} parallel workers")
# Run grid search in parallel
results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    # Submit all tasks
    future_to_params = {
        executor.submit(process_parameter_combination, (i, combination)): 
        (i, combination) for i, combination in enumerate(param_combinations)
    }
    
    # Process results as they complete
    for future in tqdm(concurrent.futures.as_completed(future_to_params), 
                      total=len(future_to_params),
                      desc="Processing parameter combinations"):
        i, _ = future_to_params[future]
        try:
            result = future.result()
            results.append(result)
            
            # Print a brief summary for monitoring
            if 'error' not in result:
                print(f"\nCompleted [{i+1}/{total_combinations}]: {result['run_id']}")
                print(f"  Runtime: {result.get('runtime', 'N/A'):.2f}s, Cophenetic corr: {result.get('cophenetic_corr', 'N/A')}")
            else:
                print(f"\nFailed [{i+1}/{total_combinations}]: {result['run_id']} - {result['error']}")
                
        except Exception as exc:
            print(f"\nError processing combination {i+1}: {exc}")


# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f"{base_dir}/grid_search_results.csv", index=False)
print(f"\nSaved detailed results to {base_dir}/grid_search_results.csv")

# Create visualizations
print("\nGenerating visualizations...")
try:
    # 1. Heat map of performance by parameter combinations
    plt.figure(figsize=(15, 12))
    
    # Choose metrics to visualize (adjust based on available metrics)
    for metric_idx, metric_name in enumerate(['runtime', 'cophenetic_corr']):
        if metric_name in results_df.columns:
            # Create a pivot table for this metric
            for i, param1 in enumerate(['metric', 'feature_selection_method']):
                for j, param2 in enumerate(['distance_metric', 'clustering_method']):
                    # Skip if we've already plotted this combination
                    if i > j:
                        continue
                        
                    plt.subplot(2, 3, i*2 + j + 1)
                    
                    # Create pivot table
                    pivot = results_df.pivot_table(
                        index=param1, 
                        columns=param2, 
                        values=metric_name,
                        aggfunc='mean'
                    )
                    
                    # Determine color map based on metric (lower is better for runtime)
                    cmap = 'rocket_r' if metric_name == 'runtime' else 'rocket'
                    
                    # Create heatmap
                    sns.heatmap(pivot, annot=True, cmap=cmap, fmt='.2f')
                    plt.title(f'{metric_name} by {param1} and {param2}')
                    plt.tight_layout()
    
    plt.savefig(f"{base_dir}/heatmap_comparison.png")
    
    # 2. Parameter importance plot
    plt.figure(figsize=(10, 6))
    
    # Choose a metric to analyze importance for
    target_metric = 'cophenetic_corr' if 'cophenetic_corr' in results_df.columns else 'runtime'
    
    # Calculate mean performance for each parameter value
    param_importance = {}
    for param in param_names:
        param_values = results_df[param].unique()
        importance = {}
        for value in param_values:
            if target_metric == 'runtime':
                # For runtime, lower is better
                importance[value] = -results_df[results_df[param] == value][target_metric].mean()
            else:
                # For other metrics, higher is better
                importance[value] = results_df[results_df[param] == value][target_metric].mean()
        param_importance[param] = importance
    
    # Plot parameter importance
    for i, (param, values) in enumerate(param_importance.items()):
        plt.subplot(2, 2, i+1)
        value_labels = list(values.keys())
        value_scores = list(values.values())
        
        # Normalize scores for visualization
        if target_metric == 'runtime':
            # Convert back to positive values for runtime
            value_scores = [-x for x in value_scores]
        
        # Sort by performance
        sorted_indices = np.argsort(value_scores)
        sorted_labels = [value_labels[i] for i in sorted_indices]
        sorted_scores = [value_scores[i] for i in sorted_indices]
        
        # Plot as horizontal bars
        bars = plt.barh(sorted_labels, sorted_scores)
        
        # Add value annotations
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                    va='center')
        
        plt.title(f'Impact of {param} on {target_metric}')
        plt.tight_layout()
    
    plt.savefig(f"{base_dir}/parameter_importance.png")
    
    # 3. Top performers table
    plt.figure(figsize=(15, 6))
    metrics_to_show = ['runtime', 'cophenetic_corr']
    
    # Find top 5 parameter combinations for each metric
    for i, metric_name in enumerate(metrics_to_show):
        if metric_name in results_df.columns:
            plt.subplot(1, len(metrics_to_show), i+1)
            
            # Sort based on metric (ascending for runtime, descending for others)
            ascending = metric_name == 'runtime'
            top_df = results_df.sort_values(metric_name, ascending=ascending).head(5)
            
            # Create a summary of top performers
            param_cols = param_names.copy()
            metric_cols = [col for col in top_df.columns if col not in param_cols and col != 'run_id']
            
            # Visualize as a table
            cell_text = []
            for _, row in top_df.iterrows():
                cell_text.append([row[p] for p in param_cols] + [f"{row[m]:.4f}" for m in metric_cols if m in row])
            
            plt.table(cellText=cell_text, 
                     colLabels=param_cols + metric_cols,
                     loc='center', 
                     cellLoc='center')
            plt.axis('off')
            plt.title(f'Top 5 Parameter Combinations by {metric_name}')
    
    plt.tight_layout()
    plt.savefig(f"{base_dir}/top_performers.png")
    
    print(f"Saved visualizations to {base_dir}/")
    
except Exception as e:
    print(f"Error creating visualizations: {e}")

# Determine best parameters
print("\n===== BEST PARAMETER COMBINATIONS =====")

# Function to find best parameters based on a metric
def find_best_parameters(df, metric_name, ascending=True):
    if metric_name not in df.columns:
        return None
    
    # Sort by the metric
    sorted_df = df.sort_values(metric_name, ascending=ascending)
    if len(sorted_df) == 0:
        return None
        
    # Get the best row
    best_row = sorted_df.iloc[0]
    
    # Extract parameters
    best_params = {param: best_row[param] for param in param_names}
    best_value = best_row[metric_name]
    
    return best_params, best_value

# Find best parameters for different metrics
best_by_runtime = find_best_parameters(results_df, 'runtime', ascending=True)
best_by_coph = find_best_parameters(results_df, 'cophenetic_corr', ascending=False)
best_by_rf = find_best_parameters(results_df, 'tree_robinson_foulds', ascending=True)

# Print recommendations
if best_by_runtime:
    params, value = best_by_runtime
    print(f"\nFastest pipeline configuration ({value:.2f}s):")
    for param, val in params.items():
        print(f"  {param}: {val}")

if best_by_coph:
    params, value = best_by_coph
    print(f"\nBest clustering quality (cophenetic correlation: {value:.4f}):")
    for param, val in params.items():
        print(f"  {param}: {val}")

if best_by_rf:
    params, value = best_by_rf
    print(f"\nClosest to reference tree (Robinson-Foulds: {value}):")
    for param, val in params.items():
        print(f"  {param}: {val}")

# Overall recommendation based on available metrics
print("\n===== FINAL RECOMMENDATION =====")
if best_by_rf:
    print("Based on reference tree similarity (most important for phylogenetic accuracy):")
    for param, val in best_by_rf[0].items():
        print(f"  {param}: {val}")
elif best_by_coph:
    print("Based on clustering quality (best for structural validity):")
    for param, val in best_by_coph[0].items():
        print(f"  {param}: {val}")
else:
    print("Based on computational efficiency:")
    for param, val in best_by_runtime[0].items():
        print(f"  {param}: {val}")

print("\nGrid search complete! Use these parameters for your production pipeline.")