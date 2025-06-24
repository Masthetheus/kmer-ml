"""
Functions for k-mer based machine learning workflows
This module is designed to organize general pipeline functions for k-mer based machine learning tasks
"""
def cluster_organisms(kmer_dir, k_value=8, dim_reduction="umap", 
                     clustering="hierarchical", n_clusters=3):
    """Complete workflow from k-mer files to clustering results"""
    pass

def optimize_clustering(feature_matrix, cluster_range=range(2, 11)):
    """Find optimal number of clusters using silhouette method"""
    pass