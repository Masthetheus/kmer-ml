"""
Dimensionality reduction methods for k-mer feature visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Union, Optional

class KmerDimensionalityReduction:
    """Reduce dimensionality of k-mer feature matrices for visualization."""
    
    def __init__(self, 
                method: str = 'umap',
                n_components: int = 2,
                random_state: int = 42):
        """
        Initialize dimensionality reduction.
        
        Args:
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of dimensions to reduce to
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit model and transform data.
        
        Args:
            X: Feature matrix to reduce
            
        Returns:
            Reduced data matrix
        """
        if self.method == 'pca':
            self.model = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'tsne':
            self.model = TSNE(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'umap':
            try:
                import umap
                self.model = umap.UMAP(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
            except ImportError:
                raise ImportError(
                    "UMAP requires the 'umap-learn' package. "
                    "Install it with: pip install umap-learn"
                )
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.method}")
        
        # Fit and transform
        result = self.model.fit_transform(X)
        self.is_fitted = True
        
        return result
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted model.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Reduced data matrix
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        if self.method == 'tsne' or self.method == 'umap':
            # t-SNE and UMAP don't support transform of new data
            # We need to fit_transform again
            return self.fit_transform(X)
        else:
            return self.model.transform(X)
    
    def plot(self, 
            X: pd.DataFrame,
            labels: Optional[Dict[str, str]] = None,
            figsize: Tuple[int, int] = (12, 10),
            title: str = 'K-mer Feature Space',
            cmap: str = 'viridis') -> None:
        """
        Reduce dimensions and plot data.
        
        Args:
            X: Feature matrix to reduce and plot
            labels: Optional dictionary mapping organism names to labels/groups
            figsize: Figure size
            title: Plot title
            cmap: Colormap for groups
        """
        # Reduce dimensions
        reduced_data = self.fit_transform(X)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # If we have labels/groups
        if labels is not None:
            # Get unique groups
            groups = set(labels.values())
            colors = plt.cm.get_cmap(cmap, len(groups))
            
            # Plot each group
            for i, group in enumerate(sorted(groups)):
                # Get indices for this group
                indices = [j for j, org in enumerate(X.index) if labels.get(org) == group]
                
                # Plot points
                ax.scatter(
                    reduced_data[indices, 0],
                    reduced_data[indices, 1],
                    label=group,
                    color=colors(i),
                    alpha=0.8
                )
                
            # Add legend
            ax.legend()
        else:
            # Plot all points with same color
            ax.scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                alpha=0.8
            )
            
            # Add organism labels
            for i, org in enumerate(X.index):
                ax.annotate(
                    org,
                    (reduced_data[i, 0], reduced_data[i, 1]),
                    fontsize=8
                )
        
        # Add labels
        ax.set_xlabel(f'{self.method.upper()} Component 1')
        ax.set_ylabel(f'{self.method.upper()} Component 2')
        ax.set_title(title)
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        return reduced_data