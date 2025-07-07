import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
try:
    import dendropy
    from dendropy.calculate import treecompare, treemeasure
    DENDROPY_AVAILABLE = True
except ImportError:
    DENDROPY_AVAILABLE = False

"""
Utility functions for working with phylogenetic trees.

This module provides functions for:
- Converting scipy linkage matrices to Newick format
- Saving and plotting dendrograms  
- Comparing phylogenetic trees
- Converting trees to distance matrices
"""


class TreeUtilityError(Exception):
    """Custom exception for tree utility operations."""
    pass


def distance_to_newick(linkage_matrix: np.ndarray, labels: List[str]) -> str:
    """
    Convert scipy linkage matrix to Newick tree format.
    
    Args:
        linkage_matrix: Hierarchical clustering linkage matrix from scipy
        labels: List of leaf node labels (organism names)
        
    Returns:
        Newick format tree string with proper branch lengths
        
    Raises:
        TreeUtilityError: If linkage matrix is invalid or labels don't match
    """
    if len(labels) != linkage_matrix.shape[0] + 1:
        raise TreeUtilityError(
            f"Labels length ({len(labels)}) must equal linkage matrix rows + 1 ({linkage_matrix.shape[0] + 1})"
        )
    
    # Convert linkage matrix to tree structure
    tree = to_tree(linkage_matrix)
    
    def _build_newick_recursive(node, leaf_names: List[str]) -> str:
        """Recursively build Newick string from tree nodes."""
        if node.is_leaf():
            return leaf_names[node.id]
        else:
            # Recurse on children
            left_subtree = _build_newick_recursive(node.get_left(), leaf_names)
            right_subtree = _build_newick_recursive(node.get_right(), leaf_names)
            
            # Format with proper branch lengths
            branch_length = node.dist / 2.0
            return f"({left_subtree},{right_subtree}):{branch_length:.6f}"
    
    # Build complete Newick string
    newick_str = _build_newick_recursive(tree, labels)
    
    # Remove root distance and add semicolon
    if ":" in newick_str:
        newick_str = newick_str.rsplit(":", 1)[0]
    
    return f"{newick_str};"


def save_tree_newick(
    linkage_matrix: np.ndarray, 
    labels: List[str],
    output_file: Union[str, Path]
) -> str:
    """
    Save hierarchical clustering result as Newick format tree file.
    
    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        labels: List of organism names corresponding to leaf nodes
        output_file: Path where to save the Newick tree file
        
    Returns:
        Absolute path to the saved file
        
    Raises:
        TreeUtilityError: If file cannot be written
    """
    try:
        # Convert to Newick format
        newick_str = distance_to_newick(linkage_matrix, labels)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(newick_str)
        
        return str(output_path.absolute())
        
    except (OSError, IOError) as e:
        raise TreeUtilityError(f"Failed to save tree file: {e}")


def plot_dendrogram(
    linkage_matrix: np.ndarray, 
    labels: List[str],
    output_file: Union[str, Path],
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = "Phylogenetic Tree",
    leaf_font_size: int = 8,
    orientation: str = 'right'
) -> str:
    """
    Plot and save dendrogram from hierarchical clustering.
    
    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        labels: List of organism names for leaf nodes
        output_file: Path to save the plot image
        figsize: Figure size (width, height) in inches
        title: Plot title (None to omit title)
        leaf_font_size: Font size for leaf labels
        orientation: Dendrogram orientation ('left', 'right', 'top', 'bottom')
        
    Returns:
        Absolute path to the saved plot file
        
    Raises:
        TreeUtilityError: If plot cannot be saved
    """
    try:
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create dendrogram with specified parameters
        dendrogram(
            linkage_matrix,
            labels=labels,
            orientation=orientation,
            leaf_font_size=leaf_font_size,
            color_threshold=0.7 * max(linkage_matrix[:, 2])  # Color threshold
        )
        
        # Add title and labels based on orientation
        if title:
            plt.title(title, fontsize=14, fontweight='bold')
            
        if orientation in ['right', 'left']:
            plt.xlabel('Distance', fontsize=12)
            plt.ylabel('Organisms', fontsize=12)
        else:
            plt.xlabel('Organisms', fontsize=12)
            plt.ylabel('Distance', fontsize=12)
        
        plt.tight_layout()
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with high DPI
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path.absolute())
        
    except Exception as e:
        plt.close()  # Ensure plot is closed even on error
        raise TreeUtilityError(f"Failed to save dendrogram plot: {e}")


def compare_trees(
    tree1_file: Union[str, Path],
    tree2_file: Union[str, Path]
) -> Dict[str, Union[float, str]]:
    """
    Compare two phylogenetic trees and calculate similarity metrics.
    
    Args:
        tree1_file: Path to first Newick tree file
        tree2_file: Path to second Newick tree file
        
    Returns:
        Dictionary containing comparison metrics:
        - robinson_foulds: Robinson-Foulds distance
        - weighted_robinson_foulds: Weighted RF distance  
        - euclidean_distance: Euclidean distance between trees
        
    Raises:
        TreeUtilityError: If dendropy is not available or trees cannot be compared
    """
    if not DENDROPY_AVAILABLE:
        raise TreeUtilityError(
            "dendropy package is required for tree comparison. "
            "Install with: pip install dendropy"
        )
    
    try:
        # Create shared taxon namespace
        taxon_namespace = dendropy.TaxonNamespace()
        
        # Load trees with shared namespace
        tree1 = dendropy.Tree.get(
            path=str(tree1_file), 
            schema="newick", 
            taxon_namespace=taxon_namespace
        )
        tree2 = dendropy.Tree.get(
            path=str(tree2_file), 
            schema="newick", 
            taxon_namespace=taxon_namespace
        )
        
        # Calculate comparison metrics
        rf_distance = treecompare.symmetric_difference(tree1, tree2)
        weighted_rf = treecompare.weighted_robinson_foulds_distance(tree1, tree2)
        euclidean_dist = treecompare.euclidean_distance(tree1, tree2)
        
        return {
            'robinson_foulds': rf_distance,
            'weighted_robinson_foulds': weighted_rf,
            'euclidean_distance': euclidean_dist,
            'n_taxa': len(taxon_namespace)
        }
        
    except Exception as e:
        raise TreeUtilityError(f"Failed to compare trees: {e}")


def load_distance_matrix_from_tree(tree_file: Union[str, Path]) -> Tuple[np.ndarray, List[str]]:
    """
    Load phylogenetic tree and convert to pairwise distance matrix.
    
    Args:
        tree_file: Path to tree file in Newick format
        
    Returns:
        Tuple of (distance_matrix, taxon_labels)
        - distance_matrix: Square numpy array of pairwise distances
        - taxon_labels: List of taxon names corresponding to matrix indices
        
    Raises:
        TreeUtilityError: If dendropy is not available or tree cannot be loaded
    """
    if not DENDROPY_AVAILABLE:
        raise TreeUtilityError(
            "dendropy package is required for tree operations. "
            "Install with: pip install dendropy"
        )
    
    try:
        # Load the tree
        tree = dendropy.Tree.get(path=str(tree_file), schema="newick")
        
        # Extract taxon information
        taxa = [taxon.label for taxon in tree.taxon_namespace]
        n_taxa = len(taxa)
        
        if n_taxa < 2:
            raise TreeUtilityError("Tree must contain at least 2 taxa")
        
        # Initialize distance matrix
        distance_matrix = np.zeros((n_taxa, n_taxa))
        
        # Compute pairwise patristic distances
        pdm = treemeasure.PatristicDistanceMatrix(tree)
        
        # Fill the distance matrix
        for i, taxon1 in enumerate(tree.taxon_namespace):
            for j, taxon2 in enumerate(tree.taxon_namespace):
                if i != j:
                    distance_matrix[i, j] = pdm.patristic_distance(taxon1, taxon2)
        
        return distance_matrix, taxa
        
    except Exception as e:
        raise TreeUtilityError(f"Failed to load distance matrix from tree: {e}")


def create_tree_from_features(
    feature_matrix: np.ndarray,
    labels: List[str],
    method: str = 'ward',
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Create hierarchical clustering tree from feature matrix.
    
    Args:
        feature_matrix: 2D array where rows are samples, columns are features
        labels: List of sample names
        method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric for linkage
        
    Returns:
        Linkage matrix suitable for use with other functions
        
    Raises:
        TreeUtilityError: If clustering fails
    """
    try:
        if feature_matrix.shape[0] != len(labels):
            raise TreeUtilityError(
                f"Feature matrix rows ({feature_matrix.shape[0]}) must match labels ({len(labels)})"
            )
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(feature_matrix, method=method, metric=metric)
        
        return linkage_matrix
        
    except Exception as e:
        raise TreeUtilityError(f"Failed to create tree from features: {e}")


# Utility function for quick tree generation and saving
def quick_tree_analysis(
    feature_matrix: np.ndarray,
    labels: List[str],
    output_dir: Union[str, Path],
    base_name: str = "tree",
    **kwargs
) -> Dict[str, str]:
    """
    Perform complete tree analysis: create tree, save Newick, and plot dendrogram.
    
    Args:
        feature_matrix: Feature matrix for clustering
        labels: Sample labels
        output_dir: Directory to save outputs
        base_name: Base name for output files
        **kwargs: Additional arguments passed to plotting/clustering functions
        
    Returns:
        Dictionary with paths to created files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create tree
    linkage_matrix = create_tree_from_features(feature_matrix, labels, **kwargs)
    
    # Save Newick file
    newick_file = save_tree_newick(
        linkage_matrix, 
        labels, 
        output_path / f"{base_name}.nwk"
    )
    
    # Save dendrogram plot
    plot_file = plot_dendrogram(
        linkage_matrix,
        labels,
        output_path / f"{base_name}_dendrogram.png",
        **{k: v for k, v in kwargs.items() if k in ['figsize', 'title', 'leaf_font_size']}
    )
    
    return {
        'newick_file': newick_file,
        'plot_file': plot_file,
        'linkage_matrix': linkage_matrix
    }