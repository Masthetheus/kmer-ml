#!/usr/bin/env python3
"""
Example script demonstrating kmer-ml machine learning pipeline
"""

import os
from pathlib import Path
from kmerml.ml import cluster_organisms, optimize_clustering

def main():
    """Run example clustering analysis"""
    
    # Set up directories
    stats_dir = "data/processed/features"
    output_dir = "data/results/example_clustering"
    
    # Check if k-mer statistics exist
    if not Path(stats_dir).exists():
        print(f"Error: K-mer statistics directory not found: {stats_dir}")
        print("Please run k-mer generation pipeline first.")
        return
    
    print("Running kmer-ml clustering analysis example...")
    
    # Define analysis parameters
    metrics = ['shannon_entropy', 'gc_percent', 'relative_freq']
    agg_funcs = ['mean', 'std']
    
    print(f"Using metrics: {metrics}")
    print(f"Using aggregation functions: {agg_funcs}")
    
    try:
        # Perform clustering analysis
        results = cluster_organisms(
            stats_dir=stats_dir,
            metrics=metrics,
            agg_funcs=agg_funcs,
            dim_reduction="umap",
            clustering="hierarchical",
            n_clusters=3,
            output_dir=output_dir
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Feature matrix shape: {results['feature_matrix'].shape}")
        print(f"Silhouette score: {results['clustering_results'].get('silhouette_score', 'N/A')}")
        
        # Show cluster assignments
        if 'labels' in results['clustering_results']:
            labels = results['clustering_results']['labels']
            organisms = results['feature_matrix'].index
            print(f"\nCluster assignments:")
            for org, cluster in zip(organisms, labels):
                print(f"  {org}: Cluster {cluster}")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
