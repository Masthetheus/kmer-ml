from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from typing import Dict, Any

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calcula múltiplas métricas de avaliação para clustering
    
    Args:
        X: Dados originais
        labels: Labels dos clusters
        
    Returns:
        Dict com métricas calculadas
    """
    if len(set(labels)) < 2:
        return {
            'silhouette': -1.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf'),
            'n_clusters': len(set(labels)),
            'noise_points': sum(labels == -1) if -1 in labels else 0
        }
    
    # Remove pontos de ruído para métricas que não suportam
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
    Rankeia resultados por métrica primária
    
    Args:
        results: Lista de dicionários com resultados
        primary_metric: Métrica para ranking
        
    Returns:
        Lista ordenada por melhor resultado
    """
    if primary_metric == 'davies_bouldin':
        # Para Davies-Bouldin, menor é melhor
        return sorted(results, key=lambda x: x['metrics'][primary_metric])
    else:
        # Para silhouette e calinski_harabasz, maior é melhor
        return sorted(results, key=lambda x: x['metrics'][primary_metric], reverse=True)