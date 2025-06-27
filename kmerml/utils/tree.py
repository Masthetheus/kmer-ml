"""
Utility functions for working with phylogenetic trees.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

def distance_to_newick(
    linkage_matrix: np.ndarray, 
    labels: List[str]
) -> str:
    """
    Convert scipy linkage matrix to Newick tree format.
    
    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        labels: List of leaf node labels (organism names)
        
    Returns:
        Newick format tree string
    """
    # Convert linkage matrix to tree
    tree = to_tree(linkage_matrix)
    
    def _build_newick(node, newick, leaf_names):
        """Recursively build Newick string from tree nodes."""
        if node.is_leaf():
            # If leaf, use the organism name
            return leaf_names[node.id]
        else:
            # If internal node, recurse on left and right children
            left = _build_newick(node.get_left(), newick, leaf_names)
            right = _build_newick(node.get_right(), newick, leaf_names)
            
            # Format distance to parent
            dist = node.dist / 2.0  # Divide by 2 for proper branch lengths
            return f"({left},{right}):{dist:.6f}"
    
    # Build Newick string, but remove final distance (root has no parent)
    newick_str = _build_newick(tree, "", labels)
    if ":" in newick_str:
        newick_str = newick_str.rsplit(":", 1)[0]
        
    # Add final semicolon
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
        labels: List of organism names
        output_file: Path to save Newick tree file
        
    Returns:
        Path to saved file
    """
    # Convert to Newick format
    newick_str = distance_to_newick(linkage_matrix, labels)
    
    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        f.write(newick_str)
    
    return str(output_path)

def plot_dendrogram(
    linkage_matrix: np.ndarray, 
    labels: List[str],
    output_file: Union[str, Path],
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = "Phylogenetic Tree",
    leaf_font_size: int = 8
) -> str:
    """
    Plot and save dendrogram from hierarchical clustering.
    
    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        labels: List of organism names
        output_file: Path to save plot image
        figsize: Figure size (width, height) in inches
        title: Plot title
        leaf_font_size: Font size for leaf labels
        
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=figsize)
    
    # Create dendrogram
    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation='right',
        leaf_font_size=leaf_font_size
    )
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Distance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return str(output_file)

def compare_trees(
    tree1_file: Union[str, Path],
    tree2_file: Union[str, Path]
) -> Dict[str, float]:
    """
    Compare two phylogenetic trees and calculate similarity metrics.
    
    Args:
        tree1_file: Path to first Newick tree file
        tree2_file: Path to second Newick tree file
        
    Returns:
        Dictionary of tree comparison metrics
    """
    try:
        # Use dendropy for tree comparison if available
        import dendropy
        from dendropy.calculate import treecompare
        
        # Load trees
        tree1 = dendropy.Tree.get(path=str(tree1_file), schema="newick")
        tree2 = dendropy.Tree.get(path=str(tree2_file), schema="newick")
        
        # Ensure trees have the same taxon namespace
        tns = dendropy.TaxonNamespace()
        tree1 = dendropy.Tree.get(path=str(tree1_file), schema="newick", taxon_namespace=tns)
        tree2 = dendropy.Tree.get(path=str(tree2_file), schema="newick", taxon_namespace=tns)
        
        # Calculate Robinson-Foulds distance
        rf_distance = treecompare.symmetric_difference(tree1, tree2)
        
        # Calculate weighted Robinson-Foulds distance
        weighted_rf = treecompare.weighted_robinson_foulds_distance(tree1, tree2)
        
        # Calculate euclidean distance
        euclidean_dist = treecompare.euclidean_distance(tree1, tree2)
        
        return {
            'robinson_foulds': rf_distance,
            'weighted_robinson_foulds': weighted_rf,
            'euclidean_distance': euclidean_dist
        }
    
    except ImportError:
        # Fallback if dendropy is not available
        return {
            'error': 'dendropy package not available for tree comparison'
        }

def load_distance_matrix_from_tree(tree_file: Union[str, Path]) -> np.ndarray:
    """
    Load a phylogenetic tree in Newick format and convert it to a distance matrix.
    
    Args:
        tree_file: Path to tree file in Newick format
        
    Returns:
        Distance matrix as numpy array
    """
    try:
        # Use dendropy to load the tree and compute distances
        import dendropy
        from dendropy.calculate import treemeasure
        
        # Load the tree
        tree = dendropy.Tree.get(path=str(tree_file), schema="newick")
        
        # Get all taxon labels
        taxa = [taxon.label for taxon in tree.taxon_namespace]
        n_taxa = len(taxa)
        
        # Initialize distance matrix
        dist_matrix = np.zeros((n_taxa, n_taxa))
        
        # Compute pairwise distances
        pdm = treemeasure.PatristicDistanceMatrix(tree)
        
        # Fill the distance matrix
        for i, taxon1 in enumerate(tree.taxon_namespace):
            for j, taxon2 in enumerate(tree.taxon_namespace):
                if i != j:  # Skip diagonal (self-distances)
                    dist_matrix[i, j] = pdm.patristic_distance(taxon1, taxon2)
        
        return dist_matrix
    
    except ImportError:
        # Fallback if dendropy is not available
        raise ImportError("dendropy package is required for tree operations. Install with: pip install dendropy")