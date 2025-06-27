"""
Machine learning clustering algorithms for phylogenetic analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Union, Optional

class PhylogeneticClustering:
    """Clustering methods for phylogenetic tree reconstruction."""
    
    def __init__(self, 
                method: str = 'upgma',
                metric: str = 'cosine'):
        """
        Initialize clustering model.
        
        Args:
            method: Linkage method ('upgma', 'wpgma', 'single', 'complete', 'ward')
            metric: Distance metric for k-mer features
        """
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.model = None
        self.tree = None

    def fit(self, X):
        """
        Fit clustering model to data.
        
        Args:
            X: Feature matrix (samples x features)
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Calculate condensed distance matrix directly
        condensed_dist = pdist(X, metric=self.metric)
        
        # Store square-form distance matrix for other uses
        self.distance_matrix = squareform(condensed_dist)
        
        # Ensure symmetry in the stored distance matrix
        np.fill_diagonal(self.distance_matrix, 0)
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        
        # Perform hierarchical clustering
        if self.method == 'upgma':
            self.linkage_matrix = linkage(condensed_dist, method='average')
        else:
            self.linkage_matrix = linkage(condensed_dist, method=self.method)
        
        # Convert linkage matrix to tree structure
        self.tree = to_tree(self.linkage_matrix)
        
        return self
    
    def get_tree(self) -> Tuple:
        """
        Get hierarchical clustering tree.
        
        Returns:
            Tree structure from scipy
        """
        if self.tree is None:
            raise ValueError("Model has not been fitted yet")
            
        return self.tree
    
    def get_linkage_matrix(self) -> np.ndarray:
        """
        Get linkage matrix.
        
        Returns:
            Linkage matrix from clustering
        """
        if self.linkage_matrix is None:
            raise ValueError("Model has not been fitted yet")
            
        return self.linkage_matrix
    
    def plot_dendrogram(self, 
                       labels: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (15, 10),
                       title: str = 'Phylogenetic Tree',
                       **kwargs) -> None:
        """
        Plot dendrogram from clustering result.
        
        Args:
            labels: Labels for leaf nodes (organisms)
            figsize: Figure size
            title: Plot title
            **kwargs: Additional arguments for dendrogram function
        """
        import matplotlib.pyplot as plt
        
        if self.linkage_matrix is None:
            raise ValueError("Model has not been fitted yet")
        
        plt.figure(figsize=figsize)
        dendrogram(
            self.linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
            **kwargs
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def save_newick(self, filepath: str, labels: Optional[List[str]] = None) -> None:
        """
        Save tree in Newick format.
        
        Args:
            filepath: Path to save file
            labels: Labels for leaf nodes (organisms)
        """
        if self.tree is None:
            raise ValueError("Model has not been fitted yet")
            
        # Convert tree to Newick format
        from Bio import Phylo
        from io import StringIO
        
        # If no labels provided, use indices
        if labels is None:
            n_leaves = len(self.linkage_matrix) + 1
            labels = [str(i) for i in range(n_leaves)]
        
        def get_newick(node, newick, parentdist, leaf_names):
            if node.is_leaf():
                return f"{leaf_names[node.id]}:{parentdist - node.dist:.10f}{newick}"
            else:
                if len(newick) > 0:
                    newick = f"):{parentdist - node.dist:.10f}{newick}"
                else:
                    newick = ");"
                newick = get_newick(node.get_right(), newick, node.dist, leaf_names)
                newick = get_newick(node.get_left(), f",{newick}", node.dist, leaf_names)
                newick = f"({newick}"
                return newick
            
        newick = get_newick(self.tree, "", self.tree.dist, labels)
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(newick)

    def _calculate_distance_matrix(self, X):
        """
        Calculate distance matrix from feature matrix.
        
        Args:
            X: Feature matrix (samples x features)
            
        Returns:
            Distance matrix with zero diagonal
        """
        from sklearn.metrics.pairwise import pairwise_distances
        
        # Calculate pairwise distances
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Use sklearn's pairwise_distances for flexibility
        dist_matrix = pairwise_distances(X, metric=self.metric)
        
        # Ensure diagonal is zero (should be already, but double-check)
        np.fill_diagonal(dist_matrix, 0)
        
        # Enforce perfect symmetry - this is the key fix
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        
        return dist_matrix
    def get_distance_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate distance matrix from feature matrix.
        
        Args:
            X: Feature matrix (organisms × k-mers)
            
        Returns:
            Distance matrix
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Calculate distances using pdist (this is more efficient)
        distances = pdist(X, metric=self.metric)
        dist_matrix = squareform(distances)
        
        # Ensure diagonal is zero and matrix is symmetric
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Enforce symmetry
        
        return dist_matrix