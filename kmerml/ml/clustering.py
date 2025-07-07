from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from typing import Dict, Any, List
import numpy as np

class ClusteringMethod:
    """Base class para métodos de clustering"""
    
    def __init__(self, name: str, estimator, param_grid: Dict[str, List]):
        self.name = name
        self.estimator = estimator
        self.param_grid = param_grid
    
    def fit_predict(self, X: np.ndarray, **params) -> np.ndarray:
        """Executa clustering com parâmetros específicos"""
        model = self.estimator(**params)
        return model.fit_predict(X)

def get_clustering_methods() -> Dict[str, ClusteringMethod]:
    """Retorna dicionário com todos os métodos de clustering disponíveis"""
    methods = {
        'kmeans': ClusteringMethod(
            name='KMeans',
            estimator=KMeans,
            param_grid={
                'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                'random_state': [42]
            }
        ),
        'gmm': ClusteringMethod(
            name='GMM',
            estimator=GaussianMixture,
            param_grid={
                'n_components': [2, 3, 4, 5, 6, 7, 8],
                'random_state': [42]
            }
        ),
        'dbscan': ClusteringMethod(
            name='DBSCAN',
            estimator=DBSCAN,
            param_grid={
                'eps': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                'min_samples': [3, 5, 7, 10]
            }
        ),
        'agglomerative': ClusteringMethod(
            name='Agglomerative',
            estimator=AgglomerativeClustering,
            param_grid={
                'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                'linkage': ['ward', 'complete', 'average']
            }
        )
    }
    return methods