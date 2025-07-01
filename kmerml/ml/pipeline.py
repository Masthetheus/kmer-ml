"""
End-to-end pipeline for k-mer based machine learning workflows
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional

from .features import KmerFeatureBuilderAgg
from .dimensionality import reduce_dimensions, plot_reduced
from .clustering import hierarchical_clustering, kmeans_clustering, plot_dendrogram, evaluate_clustering

def cluster_organisms(stats_dir, metrics=None, agg_funcs=None, k_value=8, 
                     dim_reduction="umap", clustering="hierarchical", n_clusters=3,
                     output_dir=None):
    """
    Complete workflow from k-mer statistics files to clustering results
    
    Args:
        stats_dir: Directory containing k-mer statistics files
        metrics: List of k-mer metrics to use for clustering
        agg_funcs: List of aggregation functions
        k_value: K-mer length (for organization purposes)
        dim_reduction: Dimensionality reduction method ('pca', 'umap', 'tsne')
        clustering: Clustering method ('hierarchical', 'kmeans', 'dbscan')
        n_clusters: Number of clusters for methods that require it
        output_dir: Directory to save results
    
    Returns:
        Dictionary with results and generated plots
    """
    # Set up paths
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Build feature matrix
    print("Building aggregated feature matrix...")
    builder = KmerFeatureBuilderAgg(stats_dir)
    feature_matrix = builder.build_aggregated_features(
        metrics=metrics,
        agg_funcs=agg_funcs or ['mean', 'std']
    )
    
    # Standardize features
    feature_matrix_std = builder.standardize_features()
    
    # Dimensionality reduction for visualization
    print(f"Performing {dim_reduction} dimensionality reduction...")
    reduced_data = reduce_dimensions(
        feature_matrix_std, 
        method=dim_reduction, 
        n_components=2
    )
    
    # Clustering
    print(f"Performing {clustering} clustering...")
    if clustering == "hierarchical":
        clustering_results = hierarchical_clustering(
            feature_matrix_std, 
            n_clusters=n_clusters
        )
    elif clustering == "kmeans":
        clustering_results = kmeans_clustering(
            feature_matrix_std, 
            n_clusters=n_clusters
        )
    else:
        raise ValueError(f"Clustering method '{clustering}' not implemented")
    
    # Evaluate clustering
    if 'labels' in clustering_results:
        silhouette = evaluate_clustering(feature_matrix_std, clustering_results['labels'])
        clustering_results['silhouette_score'] = silhouette
    
    # Generate plots
    plots = {}
    
    # Dimensionality reduction plot
    plots['dim_reduction'] = plot_reduced(
        reduced_data, 
        labels=feature_matrix.index.tolist(),
        title=f"K-mer Feature Space ({dim_reduction.upper()})"
    )
    
    # Clustering visualization on reduced data
    if 'labels' in clustering_results:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_data[:, 0], 
            reduced_data[:, 1], 
            c=clustering_results['labels'], 
            cmap='tab10'
        )
        plt.title(f"{clustering.title()} Clustering ({dim_reduction.upper()})")
        plt.xlabel(f"{dim_reduction.upper()} 1")
        plt.ylabel(f"{dim_reduction.upper()} 2")
        plt.colorbar(scatter)
        plots['clustering'] = plt.gcf()
    
    # Dendrogram for hierarchical clustering
    if clustering == "hierarchical" and 'linkage_matrix' in clustering_results:
        plots['dendrogram'] = plot_dendrogram(
            clustering_results['linkage_matrix'],
            labels=feature_matrix.index.tolist(),
            title="Hierarchical Clustering Dendrogram"
        )
    
    # Save results if output directory specified
    if output_dir:
        # Save feature matrix
        feature_matrix.to_csv(output_path / "feature_matrix.csv")
        feature_matrix_std.to_csv(output_path / "feature_matrix_standardized.csv")
        
        # Save reduced coordinates
        reduced_df = pd.DataFrame(
            reduced_data, 
            index=feature_matrix.index,
            columns=[f"{dim_reduction.upper()}_{i+1}" for i in range(reduced_data.shape[1])]
        )
        reduced_df.to_csv(output_path / f"coordinates_{dim_reduction}.csv")
        
        # Save clustering results
        if 'labels' in clustering_results:
            labels_df = pd.DataFrame({
                'organism': feature_matrix.index,
                'cluster': clustering_results['labels']
            })
            labels_df.to_csv(output_path / "cluster_labels.csv", index=False)
        
        # Save plots
        for name, plot in plots.items():
            plot.savefig(output_path / f"{name}.png", dpi=300, bbox_inches='tight')
    
    return {
        'feature_matrix': feature_matrix,
        'feature_matrix_standardized': feature_matrix_std,
        'reduced_data': reduced_data,
        'clustering_results': clustering_results,
        'plots': plots
    }

def optimize_clustering(feature_matrix, cluster_range=range(2, 11), method="kmeans"):
    """
    Find optimal number of clusters using silhouette analysis
    
    Args:
        feature_matrix: Standardized feature matrix
        cluster_range: Range of cluster numbers to test
        method: Clustering method to optimize
    
    Returns:
        Dictionary with optimization results
    """
    silhouette_scores = []
    inertias = []
    
    for n_clusters in cluster_range:
        if method == "kmeans":
            results = kmeans_clustering(feature_matrix, n_clusters=n_clusters)
            silhouette = evaluate_clustering(feature_matrix, results['labels'])
            silhouette_scores.append(silhouette)
            inertias.append(results['inertia'])
        elif method == "hierarchical":
            results = hierarchical_clustering(feature_matrix, n_clusters=n_clusters)
            silhouette = evaluate_clustering(feature_matrix, results['labels'])
            silhouette_scores.append(silhouette)
            inertias.append(None)
    
    # Find optimal number of clusters
    optimal_idx = np.argmax(silhouette_scores)
    optimal_n_clusters = list(cluster_range)[optimal_idx]
    
    return {
        'cluster_range': list(cluster_range),
        'silhouette_scores': silhouette_scores,
        'inertias': inertias,
        'optimal_n_clusters': optimal_n_clusters,
        'best_silhouette_score': silhouette_scores[optimal_idx]
    }
