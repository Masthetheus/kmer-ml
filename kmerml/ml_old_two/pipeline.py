"""
End-to-end machine learning pipelines for phylogenetic analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import gc
import os
import psutil

from kmerml.ml.feature_selection import PhylogeneticFeatureSelector
from kmerml.ml.clustering import PhylogeneticClustering
from kmerml.ml.feature_importance import (
    calculate_feature_tree_correlation,
    analyze_k_pattern_importance,
    plot_feature_importance
)
from kmerml.ml.dimensionality_reduction import KmerDimensionalityReduction

def check_memory_usage(threshold_gb=80):
    """Monitor memory usage and trigger garbage collection if needed"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    
    if memory_gb > threshold_gb:
        print(f"WARNING: High memory usage detected ({memory_gb:.2f} GB). Triggering garbage collection.")
        gc.collect()
        return True
    return False

class PhylogeneticPipeline:
    """End-to-end pipeline for k-mer-based phylogenetic analysis."""
    
    def __init__(self, 
                feature_dir: Union[str, Path],
                output_dir: Union[str, Path],
                reference_tree: Optional[Union[str, Path]] = None):
        """
        Initialize phylogenetic pipeline.
        
        Args:
            feature_dir: Directory containing k-mer feature files
            output_dir: Directory to save results
            reference_tree: Optional path to reference tree in Newick format
        """
        from kmerml.utils.tree import load_distance_matrix_from_tree
        
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path(output_dir)
        self.reference_tree = Path(reference_tree) if reference_tree else None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load reference distances if available
        self.reference_distances = None
        if self.reference_tree and self.reference_tree.exists():
            self.reference_distances = load_distance_matrix_from_tree(self.reference_tree)
    
    def _run_with_features(self,
                          feature_matrices,
                          n_features: int = 500,
                          feature_selection_method: str = 'mutual_info',
                          clustering_method: str = 'single',
                          distance_metric: str = 'cosine') -> Dict:
        """
        Run pipeline with pre-loaded feature matrices.
        
        Args:
            feature_matrices: List of pre-loaded feature matrices
            n_features: Number of features to select
            feature_selection_method: Method for feature selection
            clustering_method: Method for hierarchical clustering
            distance_metric: Distance metric for k-mer features
            
        Returns:
            Dictionary with results and paths to output files
        """
        # Combine matrices
        print("Combining features...")
        combined_matrix = pd.concat(feature_matrices, axis=1)
        
        # Save combined matrix
        combined_path = self.output_dir / "combined_features.csv"
        combined_matrix.to_csv(combined_path)
        
        # Step 2: Feature selection - rest of the pipeline as normal
        print("Selecting phylogenetically informative features...")
        
        if self.reference_distances is not None:
            # Supervised feature selection using reference tree
            selector = PhylogeneticFeatureSelector(
                n_features=n_features,
                method=feature_selection_method
            )
            reduced_matrix = selector.fit_transform(combined_matrix, self.reference_distances)
            
            # Save feature importances
            importances = selector.feature_importances
            importance_df = pd.DataFrame({
                'feature': list(importances.keys()),
                'importance': list(importances.values())
            }).sort_values('importance', ascending=False)
            
            importance_path = self.output_dir / "feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)
            
            # Plot top features
            plot_feature_importance(
                importances,
                top_n=20,
                title=f'Top Phylogenetically Informative K-mers (Reference-guided)'
            )
            
            # Analyze k-mer patterns
            pattern_analysis = analyze_k_pattern_importance(
                selector.get_top_features(n=100),
                importances
            )
            
            # Save pattern analysis
            pattern_path = self.output_dir / "kmer_pattern_analysis.json"
            import json
            with open(pattern_path, 'w') as f:
                json.dump(pattern_analysis, f, indent=2)
                
            # Clean up memory
            del combined_matrix
            check_memory_usage()
            
        else:
            # MEMORY-OPTIMIZED: Progressive feature selection
            print("Performing progressive feature selection...")
            
            # Initial rough filtering (removes zero/near-zero variance features)
            print("Phase 1: Initial variance filtering...")
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.0001)
            filtered = selector.fit_transform(combined_matrix)
            
            # Get column names after filtering
            filtered_cols = combined_matrix.columns[selector.get_support()]
            
            # Convert back to DataFrame
            filtered_df = pd.DataFrame(
                filtered, 
                index=combined_matrix.index,
                columns=filtered_cols
            )
            
            # Free memory
            del filtered, combined_matrix
            gc.collect()
            check_memory_usage()
            
            # Now select top N features in batches
            print(f"Phase 2: Selecting top {n_features} features...")
            batch_size = min(10000, filtered_df.shape[1])
            all_variances = []
            
            # Process in batches
            for i in range(0, filtered_df.shape[1], batch_size):
                end_idx = min(i + batch_size, filtered_df.shape[1])
                print(f"  Processing features {i} to {end_idx}...")
                batch_columns = filtered_df.columns[i:end_idx]
                batch_df = filtered_df[batch_columns]
                
                # Calculate variances for this batch
                batch_variances = batch_df.var()
                all_variances.append(batch_variances)
                
                # Free memory
                del batch_df
                gc.collect()
            
            # Combine variance results
            variances = pd.concat(all_variances)
            top_features = variances.nlargest(n_features).index.tolist()
            
            # Extract reduced matrix
            reduced_matrix = filtered_df[top_features]
            
            # Create importances dict based on variance
            importances = dict(zip(top_features, variances[top_features]))
            
            # Free memory
            del filtered_df, variances, all_variances
            gc.collect()
            check_memory_usage()
        
        # Save reduced matrix
        reduced_path = self.output_dir / "selected_features.csv"
        reduced_matrix.to_csv(reduced_path)
        
        # Step 3: Clustering and tree building
        print("Building phylogenetic tree...")
        clustering = PhylogeneticClustering(
            method=clustering_method,
            metric=distance_metric
        )
        
        # MEMORY-OPTIMIZED: Add batched distance calculation method
        def _calculate_distance_matrix_in_batches(self, X, batch_size=1000):
            """Calculate distance matrix in batches to save memory"""
            from sklearn.metrics.pairwise import pairwise_distances
            
            n_samples = X.shape[0]
            distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
            
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                X_batch_i = X[i:end_i]
                
                for j in range(0, n_samples, batch_size):
                    end_j = min(j + batch_size, n_samples)
                    X_batch_j = X[j:end_j]
                    
                    # Calculate pairwise distances for this batch
                    batch_distances = pairwise_distances(
                        X_batch_i, X_batch_j, metric=self.metric, n_jobs=-1
                    )
                    
                    # Fill the distance matrix
                    distance_matrix[i:end_i, j:end_j] = batch_distances
                    
                    # Mirror the matrix for symmetry
                    if i != j:
                        distance_matrix[j:end_j, i:end_i] = batch_distances.T
                        
                    # Check memory after each batch
                    check_memory_usage()
            
            return distance_matrix
        
        # Replace the distance calculation method with our batched version
        clustering._calculate_distance_matrix = _calculate_distance_matrix_in_batches.__get__(clustering, clustering.__class__)
        
        # Fit the clustering model
        clustering.fit(reduced_matrix)
        
        # Save tree in Newick format
        tree_path = self.output_dir / "phylogenetic_tree.nwk"
        clustering.save_newick(str(tree_path), labels=reduced_matrix.index.tolist())
        
        # Plot dendrogram
        clustering.plot_dendrogram(
            labels=reduced_matrix.index.tolist(),
            title=f'K-mer Based Phylogenetic Tree ({clustering_method.upper()})'
        )
        
        # Step 4: Dimensionality reduction for visualization
        print("Generating feature space visualization...")
        reducer = KmerDimensionalityReduction(method='umap')
        reduced_data = reducer.plot(
            reduced_matrix,
            title='K-mer Feature Space (UMAP)'
        )
        
        # Save reduced coordinates
        coords_df = pd.DataFrame(
            reduced_data,
            index=reduced_matrix.index,
            columns=['UMAP1', 'UMAP2']
        )
        coords_path = self.output_dir / "umap_coordinates.csv"
        coords_df.to_csv(coords_path)
        
        # Step 5: If we have a reference tree, compare the trees
        tree_comparison = None
        if self.reference_tree and self.reference_tree.exists():
            from kmerml.utils.tree import compare_trees
            
            print("Comparing with reference phylogeny...")
            tree_comparison = compare_trees(
                str(tree_path),
                str(self.reference_tree)
            )
            
            # Save comparison metrics
            comparison_path = self.output_dir / "tree_comparison.json"
            import json
            with open(comparison_path, 'w') as f:
                json.dump(tree_comparison, f, indent=2)
        
        # Return results
        return {
            'combined_matrix_path': str(combined_path),
            'reduced_matrix_path': str(reduced_path),
            'tree_path': str(tree_path),
            'coordinates_path': str(coords_path),
            'feature_count': reduced_matrix.shape[1],
            'organism_count': reduced_matrix.shape[0],
            'tree_comparison': tree_comparison,
            'linkage_matrix': clustering.linkage_matrix,
            'distance_matrix': clustering.distance_matrix
        }

    def run(self,
           k_values: List[int] = [6, 7, 8],
           metric: str = 'count',
           n_features: int = 500,
           feature_selection_method: str = 'random_forest',
           clustering_method: str = 'upgma',
           distance_metric: str = 'cosine') -> Dict:
        """
        Run complete phylogenetic analysis pipeline with memory optimization.
        
        Args:
            k_values: List of k-mer lengths to analyze
            metric: Feature metric to use ('count', 'gc_percent', etc.)
            n_features: Number of features to select
            feature_selection_method: Method for feature selection
            clustering_method: Method for hierarchical clustering
            distance_metric: Distance metric for k-mer features
            
        Returns:
            Dictionary with results and paths to output files
        """
        from kmerml.ml.features import KmerFeatureBuilder
        
        # Memory optimization: Use lower precision for numerical operations
        pd.set_option('precision', 5)
        
        # MEMORY-OPTIMIZED: Process K-values One at a Time
        print("Processing feature matrices one at a time...")
        feature_matrices = []

        for k in k_values:
            print(f"Processing k={k}...")
            builder = KmerFeatureBuilder(self.feature_dir)
            
            try:
                # Build matrix for this k value
                matrix = builder.build_from_statistics_files(metric=metric)
                
                # Convert to float32 to save memory
                matrix = matrix.astype(np.float32)
                
                # Check memory usage
                check_memory_usage()
                
                # Normalize
                normalized = matrix / matrix.sum(axis=1).values.reshape(-1, 1)
                normalized = normalized.astype(np.float32)  # Use lower precision
                
                # Add prefix to column names
                normalized.columns = [f"k{k}_{col}" for col in normalized.columns]
                
                feature_matrices.append(normalized)
                
                # Force cleanup of temporary data
                del matrix
                gc.collect()
                
                # Check memory again
                check_memory_usage()
                
            except Exception as e:
                print(f"Warning: Failed to process k={k}. Error: {e}")
                continue

        if not feature_matrices:
            raise ValueError(f"No valid feature matrices found for k values: {k_values}")
            
        # Continue with the rest of the pipeline
        return self._run_with_features(
            feature_matrices=feature_matrices,
            n_features=n_features,
            feature_selection_method=feature_selection_method,
            clustering_method=clustering_method,
            distance_metric=distance_metric
        )