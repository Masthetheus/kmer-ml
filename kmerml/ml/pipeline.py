"""
End-to-end pipeline for k-mer based machine learning workflows
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional

from .features import KmerFeatureBuilderAgg
from .dimensionality import reduce_dimensions, plot_reduced
from .clustering import hierarchical_clustering, kmeans_clustering, dbscan_clustering, plot_dendrogram, evaluate_clustering

def cluster_organisms(stats_dir, metrics=None, agg_funcs=None, k_value=8, 
                     dim_reduction="umap", clustering="hierarchical", n_clusters=3,
                     output_dir=None, taxonomy_file=None, eps=0.5, min_samples=5):
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
        taxonomy_file: Optional path to taxonomy TSV file for validation
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
    
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
    elif clustering == "dbscan":
        clustering_results = dbscan_clustering(
            feature_matrix_std, 
            eps=eps,
            min_samples=min_samples
        )
    else:
        raise ValueError(f"Clustering method '{clustering}' not implemented")
    
    # Evaluate clustering
    if 'labels' in clustering_results:
        silhouette = evaluate_clustering(feature_matrix_std, clustering_results['labels'])
        clustering_results['silhouette_score'] = silhouette
        
        # Validate against taxonomy if provided
        if taxonomy_file:
            taxonomy_validation = validate_clustering_with_taxonomy(
                clustering_results['labels'], 
                feature_matrix.index.tolist(),
                taxonomy_file
            )
            clustering_results['taxonomy_validation'] = taxonomy_validation
    
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
        # Generate organism summary
        organism_summary = generate_organism_summary(feature_matrix, clustering_results, output_dir)
        
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
        'plots': plots,
        'organism_summary': generate_organism_summary(feature_matrix, clustering_results) if 'labels' in clustering_results else None
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
    calinski_harabasz_scores = []
    
    for n_clusters in cluster_range:
        if method == "kmeans":
            results = kmeans_clustering(feature_matrix, n_clusters=n_clusters)
            silhouette = evaluate_clustering(feature_matrix, results['labels'], method="silhouette")
            calinski = evaluate_clustering(feature_matrix, results['labels'], method="calinski_harabasz")
            silhouette_scores.append(silhouette)
            calinski_harabasz_scores.append(calinski)
            inertias.append(results['inertia'])
        elif method == "hierarchical":
            results = hierarchical_clustering(feature_matrix, n_clusters=n_clusters)
            silhouette = evaluate_clustering(feature_matrix, results['labels'], method="silhouette")
            calinski = evaluate_clustering(feature_matrix, results['labels'], method="calinski_harabasz")
            silhouette_scores.append(silhouette)
            calinski_harabasz_scores.append(calinski)
            inertias.append(None)
    
    # Find optimal number of clusters (best silhouette score)
    optimal_idx = np.argmax(silhouette_scores)
    optimal_n_clusters = list(cluster_range)[optimal_idx]
    
    return {
        'cluster_range': list(cluster_range),
        'silhouette_scores': silhouette_scores,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'inertias': inertias,
        'optimal_n_clusters': optimal_n_clusters,
        'best_silhouette_score': silhouette_scores[optimal_idx],
        'recommendation': f"Optimal number of clusters: {optimal_n_clusters} (Silhouette Score: {silhouette_scores[optimal_idx]:.3f})"
    }

def validate_clustering_with_taxonomy(clustering_labels, organism_ids, taxonomy_file=None):
    """
    Validate clustering results against known taxonomy
    
    Args:
        clustering_labels: Cluster assignments from ML
        organism_ids: List of organism identifiers
        taxonomy_file: Path to TSV file with taxonomy information
        
    Returns:
        Dictionary with validation metrics
    """
    if taxonomy_file is None:
        return None
        
    try:
        import pandas as pd
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Load taxonomy data
        tax_df = pd.read_csv(taxonomy_file, sep='\t')
        
        # Match organisms with their taxonomy
        organism_taxonomy = {}
        for org_id in organism_ids:
            # Extract accession from organism ID (e.g., GCF_000001)
            accession = org_id.split('_')[0] + '_' + org_id.split('_')[1]
            matching_row = tax_df[tax_df['accession'].str.contains(accession, na=False)]
            
            if not matching_row.empty:
                organism_taxonomy[org_id] = {
                    'phylum': matching_row.iloc[0].get('phylum', 'Unknown'),
                    'family': matching_row.iloc[0].get('family', 'Unknown'),
                    'species': matching_row.iloc[0].get('species', 'Unknown')
                }
        
        if not organism_taxonomy:
            return {'warning': 'No taxonomy matches found'}
        
        # Calculate validation metrics for different taxonomic levels
        validation_results = {}
        
        for tax_level in ['phylum', 'family', 'species']:
            # Create true labels based on taxonomy
            true_labels = []
            filtered_clustering_labels = []
            
            for i, org_id in enumerate(organism_ids):
                if org_id in organism_taxonomy:
                    true_labels.append(organism_taxonomy[org_id][tax_level])
                    filtered_clustering_labels.append(clustering_labels[i])
            
            if len(set(true_labels)) > 1:  # Need at least 2 different taxonomic groups
                # Convert taxonomic labels to numeric
                unique_tax = list(set(true_labels))
                true_numeric = [unique_tax.index(label) for label in true_labels]
                
                # Calculate metrics
                ari = adjusted_rand_score(true_numeric, filtered_clustering_labels)
                nmi = normalized_mutual_info_score(true_numeric, filtered_clustering_labels)
                
                validation_results[tax_level] = {
                    'adjusted_rand_score': ari,
                    'normalized_mutual_info': nmi,
                    'n_organisms': len(true_labels),
                    'n_true_groups': len(set(true_labels)),
                    'n_predicted_clusters': len(set(filtered_clustering_labels))
                }
        
        return validation_results
        
    except Exception as e:
        return {'error': f'Taxonomy validation failed: {str(e)}'}

def generate_organism_summary(feature_matrix, clustering_results, output_dir=None):
    """
    Generate detailed summary for each organism in the clustering
    
    Args:
        feature_matrix: Original feature matrix
        clustering_results: Results from clustering
        output_dir: Directory to save summary
        
    Returns:
        DataFrame with organism-level summary
    """
    organism_summary = pd.DataFrame({
        'organism_id': feature_matrix.index,
        'cluster': clustering_results.get('labels', ['Unknown'] * len(feature_matrix))
    })
    
    # Add feature statistics for each organism
    for col in feature_matrix.columns:
        organism_summary[f'{col}_value'] = feature_matrix[col].values
    
    # Add cluster statistics
    if 'labels' in clustering_results:
        cluster_sizes = pd.Series(clustering_results['labels']).value_counts()
        organism_summary['cluster_size'] = organism_summary['cluster'].map(cluster_sizes)
        
        # Calculate distance to cluster center for kmeans
        if 'cluster_centers' in clustering_results:
            distances_to_center = []
            for i, (_, row) in enumerate(organism_summary.iterrows()):
                cluster_id = row['cluster']
                if cluster_id >= 0:  # Valid cluster (not noise for DBSCAN)
                    center = clustering_results['cluster_centers'][cluster_id]
                    organism_features = feature_matrix.iloc[i].values
                    distance = np.linalg.norm(organism_features - center)
                    distances_to_center.append(distance)
                else:
                    distances_to_center.append(np.nan)
            organism_summary['distance_to_center'] = distances_to_center
    
    # Save if output directory provided
    if output_dir:
        organism_summary.to_csv(Path(output_dir) / "organism_summary.csv", index=False)
    
    return organism_summary
