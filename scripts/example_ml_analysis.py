#!/usr/bin/env python3
"""
Example script demonstrating kmer-ml machine learning pipeline
with taxonomic validation and optimization
"""

import os
from pathlib import Path
from kmerml.ml import cluster_organisms, optimize_clustering

def main():
    """Run comprehensive clustering analysis example"""
    
    # Set up directories
    stats_dir = "data/processed/features"
    output_dir = "data/results/example_clustering"
    taxonomy_file = "data/ncbi_dataset.tsv"  # Optional taxonomy file
    
    # Check if k-mer statistics exist
    if not Path(stats_dir).exists():
        print(f"Error: K-mer statistics directory not found: {stats_dir}")
        print("Please run k-mer generation pipeline first.")
        return
    
    print("Running kmer-ml comprehensive clustering analysis...")
    
    # Define analysis parameters optimized for organism clustering
    metrics = ['shannon_entropy', 'gc_percent', 'relative_freq', 'at_skew']
    agg_funcs = ['mean', 'std', 'min', 'max']
    
    print(f"Using metrics: {metrics}")
    print(f"Using aggregation functions: {agg_funcs}")
    
    try:
        # Step 1: Find optimal number of clusters
        print("\n=== STEP 1: Optimizing cluster number ===")
        from kmerml.ml.features import KmerFeatureBuilderAgg
        
        builder = KmerFeatureBuilderAgg(stats_dir)
        feature_matrix = builder.build_aggregated_features(metrics=metrics, agg_funcs=agg_funcs)
        feature_matrix_std = builder.standardize_features()
        
        optimization_results = optimize_clustering(
            feature_matrix_std, 
            cluster_range=range(2, 8),
            method="hierarchical"
        )
        
        optimal_clusters = optimization_results['optimal_n_clusters']
        print(f"Optimal number of clusters: {optimal_clusters}")
        print(f"Best silhouette score: {optimization_results['best_silhouette_score']:.3f}")
        
        # Step 2: Perform clustering with optimal parameters
        print(f"\n=== STEP 2: Clustering with {optimal_clusters} clusters ===")
        
        # Check if taxonomy file exists
        tax_file = taxonomy_file if Path(taxonomy_file).exists() else None
        if tax_file:
            print(f"Using taxonomy file for validation: {tax_file}")
        else:
            print("No taxonomy file found - skipping taxonomic validation")
        
        results = cluster_organisms(
            stats_dir=stats_dir,
            metrics=metrics,
            agg_funcs=agg_funcs,
            dim_reduction="umap",
            clustering="hierarchical",
            n_clusters=optimal_clusters,
            output_dir=output_dir,
            taxonomy_file=tax_file
        )
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Feature matrix shape: {results['feature_matrix'].shape}")
        print(f"Silhouette score: {results['clustering_results'].get('silhouette_score', 'N/A'):.3f}")
        
        # Show cluster assignments
        if 'labels' in results['clustering_results']:
            labels = results['clustering_results']['labels']
            organisms = results['feature_matrix'].index
            
            print(f"\n=== CLUSTER ASSIGNMENTS ===")
            cluster_counts = {}
            for org, cluster in zip(organisms, labels):
                if cluster not in cluster_counts:
                    cluster_counts[cluster] = []
                cluster_counts[cluster].append(org)
            
            for cluster_id, organisms_in_cluster in cluster_counts.items():
                print(f"\nCluster {cluster_id} ({len(organisms_in_cluster)} organisms):")
                for org in organisms_in_cluster[:5]:  # Show first 5
                    print(f"  - {org}")
                if len(organisms_in_cluster) > 5:
                    print(f"  ... and {len(organisms_in_cluster) - 5} more")
        
        # Show taxonomic validation if available
        if 'taxonomy_validation' in results['clustering_results']:
            tax_val = results['clustering_results']['taxonomy_validation']
            print(f"\n=== TAXONOMIC VALIDATION ===")
            for level, metrics in tax_val.items():
                if isinstance(metrics, dict) and 'adjusted_rand_score' in metrics:
                    print(f"{level.upper()}:")
                    print(f"  Adjusted Rand Score: {metrics['adjusted_rand_score']:.3f}")
                    print(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.3f}")
                    print(f"  True groups: {metrics['n_true_groups']}, Predicted clusters: {metrics['n_predicted_clusters']}")
        
        # Step 3: Try different clustering methods for comparison
        print(f"\n=== STEP 3: Comparing clustering methods ===")
        
        methods = ['hierarchical', 'kmeans', 'dbscan']
        comparison_results = {}
        
        for method in methods:
            try:
                if method == 'dbscan':
                    method_results = cluster_organisms(
                        stats_dir=stats_dir,
                        metrics=metrics,
                        agg_funcs=agg_funcs,
                        dim_reduction="umap",
                        clustering=method,
                        eps=0.5,
                        min_samples=3,
                        output_dir=f"{output_dir}_{method}"
                    )
                else:
                    method_results = cluster_organisms(
                        stats_dir=stats_dir,
                        metrics=metrics,
                        agg_funcs=agg_funcs,
                        dim_reduction="umap",
                        clustering=method,
                        n_clusters=optimal_clusters,
                        output_dir=f"{output_dir}_{method}"
                    )
                
                silhouette = method_results['clustering_results'].get('silhouette_score', 'N/A')
                n_clusters_found = len(set(method_results['clustering_results'].get('labels', [])))
                comparison_results[method] = {
                    'silhouette_score': silhouette,
                    'n_clusters': n_clusters_found
                }
                print(f"{method}: Silhouette={silhouette:.3f}, Clusters={n_clusters_found}")
                
            except Exception as e:
                print(f"{method}: Failed - {e}")
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print("Check the output directories for detailed results and visualizations!")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
