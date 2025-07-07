from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from typing import Dict

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate clustering metrics
    
    Args:
        X: Original data
        labels: clusters labels
        
    Returns:
        Dict with calculated metrics:
            - silhouette: Silhouette score
            - calinski_harabasz: Calinski-Harabasz index
            - davies_bouldin: Davies-Bouldin index
            - n_clusters: Number of clusters
            - noise_points: Number of noise points (if applicable)
    """
    if len(set(labels)) < 2:
        return {
            'silhouette': -1.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf'),
            'n_clusters': len(set(labels)),
            'noise_points': sum(labels == -1) if -1 in labels else 0
        }
    
    # Remove noise points if data does not support it
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    if len(set(labels_clean)) < 2:
        return {
            'silhouette': -1.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf'),
            'n_clusters': len(set(labels)),
            'noise_points': sum(labels == -1)
        }
    
    return {
        'silhouette': silhouette_score(X_clean, labels_clean),
        'calinski_harabasz': calinski_harabasz_score(X_clean, labels_clean),
        'davies_bouldin': davies_bouldin_score(X_clean, labels_clean),
        'n_clusters': len(set(labels)),
        'noise_points': sum(labels == -1) if -1 in labels else 0
    }

def rank_results(results: list, primary_metric: str = 'silhouette') -> list:
    """
    Ranks results based on the primary metric.
    
    Args:
        results: Dict with results
        primary_metric: Primary metric to use for ranking
    Returns:
        Sorted array of results based on the primary metric.
    """
    if primary_metric == 'davies_bouldin':
        # For davies_bouldin, lower is better
        return sorted(results, key=lambda x: x['metrics'][primary_metric])
    else:
        return sorted(results, key=lambda x: x['metrics'][primary_metric], reverse=True)