import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def reduce_dimensions(feature_matrix, method="pca", n_components=2, random_state=42, **kwargs):
    """
    Reduce dimensions of aggregated feature matrix (organism x aggregated features).
    Args:
        feature_matrix: pd.DataFrame or np.ndarray (organisms x features)
        method: "pca", "umap", or "tsne"
        n_components: int, number of dimensions
        random_state: int, for reproducibility
        **kwargs: extra args for the reducer
    Returns:
        np.ndarray of reduced dimensions (organisms x n_components)
    """
    if isinstance(feature_matrix, pd.DataFrame):
        X = feature_matrix.values
    else:
        X = feature_matrix

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, **kwargs)
    else:
        raise ValueError("method must be 'pca', 'umap', or 'tsne'")

    X_reduced = reducer.fit_transform(X)
    return X_reduced

def plot_reduced(X_reduced, labels=None, title="Dimensionality Reduction", figsize=(7,5)):
    """
    Simple scatter plot for 2D reduced data.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if labels is not None:
        for label in np.unique(labels):
            idx = np.array(labels) == label
            plt.scatter(X_reduced[idx,0], X_reduced[idx,1], label=str(label))
        plt.legend()
    else:
        plt.scatter(X_reduced[:,0], X_reduced[:,1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()