import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from typing import List, Tuple, Union, Optional
from pathlib import Path

"""
Feature selection methods for identifying phylogenetically informative k-mers.
"""

class PhylogeneticFeatureSelector:
    """Select k-mer features that are most informative for phylogenetic reconstruction."""
    
    def __init__(self, 
                n_features: int = 500, 
                method: str = 'random_forest',
                random_state: int = 42):
        """
        Initialize feature selector.
        
        Args:
            n_features: Number of features to select
            method: Feature selection method ('random_forest', 'mutual_info', 'correlation')
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.method = method
        self.random_state = random_state
        self.selected_features = None
        self.feature_importances = None
        
    def fit(self, 
           X: pd.DataFrame, 
           reference_distances: np.ndarray) -> 'PhylogeneticFeatureSelector':
        """
        Fit feature selector to identify phylogenetically informative k-mers.
        
        Args:
            X: Feature matrix (organisms × k-mers)
            reference_distances: Pairwise evolutionary distances from reference phylogeny
            
        Returns:
            self: Fitted selector
        """
        if self.method == 'random_forest':
            return self._fit_random_forest(X, reference_distances)
        elif self.method == 'mutual_info':
            return self._fit_mutual_info(X, reference_distances)
        elif self.method == 'correlation':
            return self._fit_correlation(X, reference_distances)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
            
    def _fit_random_forest(self, 
                          X: pd.DataFrame, 
                          reference_distances: np.ndarray) -> 'PhylogeneticFeatureSelector':
        """Use Random Forest to select features most predictive of phylogenetic distances."""
        # Create distance prediction task
        X_pairs, y_pairs = self._create_distance_pairs(X, reference_distances)
        
        # Train a Random Forest regressor
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=self.random_state
        )
        model.fit(X_pairs, y_pairs)
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Store feature importances - FIX: Ensure indices are in bounds
        self.feature_importances = {}
        for i in range(min(len(importances), len(X.columns))):
            if i < len(X.columns):  # Extra safety check
                self.feature_importances[X.columns[i]] = importances[i]
        
        # Select top features - FIX: Ensure we don't exceed bounds
        n_to_select = min(self.n_features, len(indices), len(X.columns))
        top_indices = indices[:n_to_select]
        self.selected_features = [X.columns[i] for i in top_indices if i < len(X.columns)]
        
        return self
    
    def _fit_mutual_info(self, 
                        X: pd.DataFrame, 
                        reference_distances: np.ndarray) -> 'PhylogeneticFeatureSelector':
        """Use mutual information to select features most related to phylogenetic distances."""
        # Create distance prediction task
        X_pairs, y_pairs = self._create_distance_pairs(X, reference_distances)
        
        # Check if we have enough samples for MI calculation
        n_samples = X_pairs.shape[0]
        if n_samples < 4:  # Need more than 3 samples for default n_neighbors=3
            print(f"Warning: Only {n_samples} sample pairs available. Adjusting n_neighbors for MI calculation.")
            # Adjust n_neighbors parameter based on sample size
            n_neighbors = max(1, n_samples - 1)
            mi_scores = mutual_info_regression(
                X_pairs, y_pairs, 
                random_state=self.random_state,
                n_neighbors=n_neighbors
            )
        else:
            # Use default parameters
            mi_scores = mutual_info_regression(
                X_pairs, y_pairs, 
                random_state=self.random_state
            )
        
        indices = np.argsort(mi_scores)[::-1]
        
        # Store feature importances - FIX: Ensure indices are in bounds
        self.feature_importances = {}
        for i in range(min(len(mi_scores), len(X.columns))):
            if i < len(X.columns):  # Extra safety check
                self.feature_importances[X.columns[i]] = mi_scores[i]
        
        # Select top features - FIX: Ensure we don't exceed bounds
        n_to_select = min(self.n_features, len(indices), len(X.columns))
        top_indices = indices[:n_to_select]
        self.selected_features = [X.columns[i] for i in top_indices if i < len(X.columns)]
        
        return self
    
    def _fit_correlation(self, 
                        X: pd.DataFrame, 
                        reference_distances: np.ndarray) -> 'PhylogeneticFeatureSelector':
        """Use correlation with reference distances to select features."""
        from scipy.spatial.distance import pdist, squareform
        
        # Get organism names
        organisms = X.index.tolist()
        n_organisms = len(organisms)
        
        # Calculate correlation for each feature
        correlations = {}
        for col in X.columns:
            # Calculate pairwise distances based on this feature
            feature_values = X[col].values.reshape(-1, 1)
            feature_distances = squareform(pdist(feature_values, metric='euclidean'))
            
            # Flatten the distance matrices (excluding diagonal)
            flat_ref_dist = []
            flat_feature_dist = []
            for i in range(n_organisms):
                for j in range(i+1, n_organisms):
                    flat_ref_dist.append(reference_distances[i, j])
                    flat_feature_dist.append(feature_distances[i, j])
            
            # Calculate Spearman correlation
            corr, _ = spearmanr(flat_ref_dist, flat_feature_dist)
            correlations[col] = abs(corr)  # Use absolute correlation
        
        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # In the _fit_correlation method, when storing feature importances
        self.feature_importances = dict(sorted_features)
        
        # Replace with:
        self.feature_importances = {}
        for feature, importance in sorted_features:
            if feature in X.columns:  # Safety check
                self.feature_importances[feature] = importance
        
        # And when selecting top features:
        n_to_select = min(self.n_features, len(sorted_features))
        self.selected_features = [f[0] for f in sorted_features[:n_to_select] if f[0] in X.columns]
        return self
    
    def _create_distance_pairs(self, 
                             X: pd.DataFrame, 
                             reference_distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data for distance prediction.
        
        For each pair of organisms (i,j), create a feature vector combining their k-mer features,
        and use the reference distance as the target.
        """
        organisms = X.index.tolist()
        n_organisms = len(organisms)
        n_features = X.shape[1]
        
        # Calculate number of pairs
        n_pairs = n_organisms * (n_organisms - 1) // 2
        
        # Create arrays for pairs
        X_pairs = np.zeros((n_pairs, n_features * 2))
        y_pairs = np.zeros(n_pairs)
        
        # Fill arrays
        pair_idx = 0
        for i in range(n_organisms):
            for j in range(i+1, n_organisms):
                # Feature vector is concatenation of both organisms' features
                X_pairs[pair_idx, :n_features] = X.iloc[i].values
                X_pairs[pair_idx, n_features:] = X.iloc[j].values
                
                # Target is reference distance
                y_pairs[pair_idx] = reference_distances[i, j]
                
                pair_idx += 1
        
        return X_pairs, y_pairs
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return feature matrix with only selected features.
        
        Args:
            X: Feature matrix (organisms × k-mers)
            
        Returns:
            Reduced feature matrix with selected features
        """
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet")
            
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, reference_distances: np.ndarray) -> pd.DataFrame:
        """
        Fit selector and return reduced feature matrix.
        
        Args:
            X: Feature matrix (organisms × k-mers)
            reference_distances: Pairwise evolutionary distances from reference phylogeny
            
        Returns:
            Reduced feature matrix with selected features
        """
        self.fit(X, reference_distances)
        return self.transform(X)
    
    def get_top_features(self, n: int = 10) -> List[str]:
        """
        Get top n most important features.
        
        Args:
            n: Number of features to return
            
        Returns:
            List of feature names
        """
        if self.feature_importances is None:
            raise ValueError("Selector has not been fitted yet")
            
        sorted_features = sorted(
            self.feature_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [f[0] for f in sorted_features[:n]]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save feature selector state to file.
        
        Args:
            filepath: Path to save file
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_features': self.n_features,
                'method': self.method,
                'random_state': self.random_state,
                'selected_features': self.selected_features,
                'feature_importances': self.feature_importances
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'PhylogeneticFeatureSelector':
        """
        Load feature selector from file.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            Loaded feature selector
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        selector = cls(
            n_features=data['n_features'],
            method=data['method'],
            random_state=data['random_state']
        )
        selector.selected_features = data['selected_features']
        selector.feature_importances = data['feature_importances']
        
        return selector


def select_features_by_variance(
    X: pd.DataFrame, 
    threshold: float = 0.0, 
    n_features: Optional[int] = None
) -> List[str]:
    """
    Select features based on variance.
    
    Args:
        X: Feature matrix (organisms × k-mers)
        threshold: Minimum variance to keep feature
        n_features: Maximum number of features to select
        
    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import VarianceThreshold
    
    # Apply variance threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    
    # Get features that pass threshold
    features = X.columns[selector.get_support()]
    
    # If n_features specified, select top by variance
    if n_features is not None and len(features) > n_features:
        variances = X[features].var().sort_values(ascending=False)
        features = variances.index[:n_features]
    
    return features.tolist()