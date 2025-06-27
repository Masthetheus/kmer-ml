"""
Methods for analyzing feature importance in phylogenetic trees.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Union, Optional

def calculate_feature_tree_correlation(
    feature_matrix: pd.DataFrame,
    tree_distances: np.ndarray
) -> Dict[str, float]:
    """
    Calculate correlation between each feature and the tree distances.
    
    Args:
        feature_matrix: K-mer feature matrix (organisms × k-mers)
        tree_distances: Pairwise distances from the phylogenetic tree
        
    Returns:
        Dictionary mapping feature names to correlation coefficients
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Get organism names
    organisms = feature_matrix.index.tolist()
    n_organisms = len(organisms)
    
    # Prepare flattened tree distances (upper triangle)
    flat_tree_dist = []
    for i in range(n_organisms):
        for j in range(i+1, n_organisms):
            flat_tree_dist.append(tree_distances[i, j])
    
    # Calculate correlation for each feature
    correlations = {}
    for col in feature_matrix.columns:
        # Calculate pairwise distances based on this feature
        feature_values = feature_matrix[col].values.reshape(-1, 1)
        feature_distances = squareform(pdist(feature_values, metric='euclidean'))
        
        # Flatten the feature distances (upper triangle)
        flat_feature_dist = []
        for i in range(n_organisms):
            for j in range(i+1, n_organisms):
                flat_feature_dist.append(feature_distances[i, j])
        
        # Calculate Spearman correlation
        corr, _ = spearmanr(flat_tree_dist, flat_feature_dist)
        correlations[col] = corr
    
    return correlations

def analyze_k_pattern_importance(
    top_features: List[str],
    importances: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze importance patterns by k-mer length and composition.
    
    Args:
        top_features: List of k-mer feature names
        importances: Dictionary of feature importances
        
    Returns:
        Dictionary with analysis results
    """
    # Extract k-values
    k_values = {}
    for feature in top_features:
        # Assuming feature names like "k6_ATCGAT" or "ATCGAT_k6"
        if '_' in feature:
            parts = feature.split('_')
            if parts[0].startswith('k') and parts[0][1:].isdigit():
                k = int(parts[0][1:])
                kmer = parts[1]
            elif parts[-1].startswith('k') and parts[-1][1:].isdigit():
                k = int(parts[-1][1:])
                kmer = parts[0]
            else:
                # Can't determine k, use the whole feature
                k = len(feature)
                kmer = feature
        else:
            # No explicit k, use length of the feature
            k = len(feature)
            kmer = feature
        
        if k not in k_values:
            k_values[k] = []
        
        k_values[k].append((kmer, importances[feature]))
    
    # Calculate statistics by k
    k_stats = {}
    for k, features in k_values.items():
        values = [f[1] for f in features]
        k_stats[k] = {
            'count': len(features),
            'mean_importance': np.mean(values),
            'max_importance': np.max(values),
            'top_kmer': features[np.argmax(values)][0]
        }
    
    # Analyze base composition
    base_stats = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'CpG': 0}
    for feature, importance in [(f, importances[f]) for f in top_features]:
        kmer = feature.split('_')[-1] if '_' in feature else feature
        for base in 'ATGC':
            base_stats[base] += kmer.count(base) * importance
        base_stats['CpG'] += kmer.count('CG') * importance
    
    # Normalize
    total = sum(importances.values())
    for base in base_stats:
        base_stats[base] /= total
    
    return {
        'k_value_stats': k_stats,
        'base_composition': base_stats
    }

def plot_feature_importance(
    importances: Dict[str, float],
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Top Phylogenetically Informative K-mers'
) -> None:
    """
    Plot top features by importance.
    
    Args:
        importances: Dictionary mapping features to importance scores
        top_n: Number of top features to show
        figsize: Figure size
        title: Plot title
    """
    # Sort features by importance
    sorted_features = sorted(
        importances.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:top_n]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract feature names and values
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    # Create barplot
    bars = ax.barh(
        range(len(features)), 
        [abs(v) for v in values], 
        color=['red' if v < 0 else 'blue' for v in values]
    )
    
    # Add feature names
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    
    # Add labels
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    
    # Show plot
    plt.tight_layout()
    plt.show()

def get_discriminative_kmers_by_clade(
    feature_matrix: pd.DataFrame,
    clade_labels: Dict[str, str]
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find k-mers that discriminate between clades.
    
    Args:
        feature_matrix: K-mer feature matrix (organisms × k-mers)
        clade_labels: Dictionary mapping organism names to clade labels
        
    Returns:
        Dictionary mapping clades to lists of (k-mer, score) tuples
    """
    from sklearn.feature_selection import mutual_info_classif
    
    # Prepare data
    X = feature_matrix.copy()
    
    # Create target vector
    y = np.array([clade_labels.get(org, 'unknown') for org in X.index])
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y)
    
    # Map scores to features
    feature_scores = dict(zip(X.columns, mi_scores))
    
    # Find discriminative k-mers for each clade
    clade_kmers = {}
    for clade in set(clade_labels.values()):
        # Create binary target: 1 for this clade, 0 for others
        binary_target = np.array([1 if c == clade else 0 for c in y])
        
        # Calculate mean value for each feature within and outside clade
        clade_means = {}
        for col in X.columns:
            in_clade_mean = X[col][binary_target == 1].mean()
            out_clade_mean = X[col][binary_target == 0].mean()
            
            # Positive score means higher in this clade
            clade_means[col] = in_clade_mean - out_clade_mean
        
        # Sort by absolute difference, weighted by mutual information
        weighted_scores = {
            col: clade_means[col] * feature_scores[col] 
            for col in X.columns
        }
        
        # Get top k-mers for this clade
        top_kmers = sorted(
            weighted_scores.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:50]  # Top 50 per clade
        
        clade_kmers[clade] = top_kmers
    
    return clade_kmers