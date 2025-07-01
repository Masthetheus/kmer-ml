"""
Clustering functions for k-mer based machine learning workflows
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def hierarchical_clustering(feature_matrix, n_clusters=None, method="ward", metric="euclidean"):
    """
    Perform hierarchical clustering on k-mer features
    
    Args:
        feature_matrix: DataFrame or array with organisms as rows
        n_clusters: Number of clusters (if None, returns linkage matrix)
        method: Linkage method ('ward', 'single', 'complete', 'average')
        metric: Distance metric
    
    Returns:
        Dictionary with clustering results
    """
    # Calculate distance matrix
    if method == "ward":
        distances = pdist(feature_matrix, metric="euclidean")
    else:
        distances = pdist(feature_matrix, metric=metric)
    
    # Perform linkage
    linkage_matrix = linkage(distances, method=method)
    
    result = {
        'linkage_matrix': linkage_matrix,
        'distance_matrix': squareform(distances)
    }
    
    if n_clusters:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = clustering.fit_predict(feature_matrix)
        result['labels'] = labels
        result['n_clusters'] = n_clusters
    
    return result

def kmeans_clustering(feature_matrix, n_clusters=3, random_state=42, **kwargs):
    """
    Perform k-means clustering on k-mer features
    
    Args:
        feature_matrix: DataFrame or array with organisms as rows
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for KMeans
    
    Returns:
        Dictionary with clustering results
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    labels = kmeans.fit_predict(feature_matrix)
    
    return {
        'labels': labels,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'n_clusters': n_clusters
    }

def dbscan_clustering(feature_matrix, eps=0.5, min_samples=5, **kwargs):
    """
    Perform DBSCAN clustering on k-mer features
    
    Args:
        feature_matrix: DataFrame or array with organisms as rows
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood
        **kwargs: Additional parameters for DBSCAN
    
    Returns:
        Dictionary with clustering results
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = dbscan.fit_predict(feature_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'core_sample_indices': dbscan.core_sample_indices_
    }

def gaussian_mixture_clustering(feature_matrix, n_components=3, random_state=42, **kwargs):
    """
    Perform Gaussian Mixture Model clustering
    
    Args:
        feature_matrix: DataFrame or array with organisms as rows
        n_components: Number of mixture components
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for GaussianMixture
    
    Returns:
        Dictionary with clustering results
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, **kwargs)
    gmm.fit(feature_matrix)
    labels = gmm.predict(feature_matrix)
    probabilities = gmm.predict_proba(feature_matrix)
    
    return {
        'labels': labels,
        'probabilities': probabilities,
        'n_components': n_components,
        'aic': gmm.aic(feature_matrix),
        'bic': gmm.bic(feature_matrix),
        'log_likelihood': gmm.score(feature_matrix)
    }

def plot_dendrogram(linkage_matrix, labels=None, title="Hierarchical Clustering"):
    """
    Plot dendrogram from linkage matrix
    
    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        labels: List of organism labels
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.xlabel("Organisms")
    plt.ylabel("Distance")
    plt.tight_layout()
    return plt.gcf()

def evaluate_clustering(feature_matrix, labels, method="silhouette"):
    """
    Evaluate clustering quality
    
    Args:
        feature_matrix: Original feature matrix
        labels: Cluster labels
        method: Evaluation method ('silhouette', 'calinski_harabasz', 'davies_bouldin')
    
    Returns:
        Evaluation score
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    if len(set(labels)) < 2:
        return None
    
    if method == "silhouette":
        return silhouette_score(feature_matrix, labels)
    elif method == "calinski_harabasz":
        return calinski_harabasz_score(feature_matrix, labels)
    elif method == "davies_bouldin":
        return davies_bouldin_score(feature_matrix, labels)
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
