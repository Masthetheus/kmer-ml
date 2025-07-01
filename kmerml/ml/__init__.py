"""Machine learning modules for k-mer analysis."""

from .features import KmerFeatureBuilderAgg
from .dimensionality import reduce_dimensions, plot_reduced
from .clustering import hierarchical_clustering, kmeans_clustering, evaluate_clustering
from .pipeline import cluster_organisms, optimize_clustering

__all__ = [
    'KmerFeatureBuilderAgg',
    'reduce_dimensions', 'plot_reduced',
    'hierarchical_clustering', 'kmeans_clustering', 'evaluate_clustering',
    'cluster_organisms', 'optimize_clustering'
]
