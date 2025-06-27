"""
Machine learning models for predicting phylogenetic distances from k-mer features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union, Optional

class PhylogeneticDistancePredictor(BaseEstimator, RegressorMixin):
    """Machine learning model to predict phylogenetic distances from k-mer features."""
    
    def __init__(self, 
                model_type: str = 'random_forest',
                n_estimators: int = 100,
                random_state: int = 42):
        """
        Initialize distance predictor.
        
        Args:
            model_type: ML algorithm ('random_forest', 'gradient_boosting')
            n_estimators: Number of estimators for ensemble methods
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = self._create_model()
        self.scaler = StandardScaler()
        
    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PhylogeneticDistancePredictor':
        """
        Train the distance predictor.
        
        Args:
            X: Feature matrix of organism pairs
            y: Target distances
            
        Returns:
            self: Trained model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict distances for organism pairs.
        
        Args:
            X: Feature matrix of organism pairs
            
        Returns:
            Predicted distances
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix of organism pairs
            y: True distances
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import spearmanr
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        spearman_corr, _ = spearmanr(y, y_pred)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'spearman_correlation': spearman_corr
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix of organism pairs
            y: True distances
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation metrics
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        cv_rmse = -cross_val_score(
            self.model, X_scaled, y, 
            cv=cv, scoring='neg_root_mean_squared_error'
        )
        
        cv_r2 = cross_val_score(
            self.model, X_scaled, y, 
            cv=cv, scoring='r2'
        )
        
        return {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save file
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'n_estimators': self.n_estimators,
                'random_state': self.random_state
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'PhylogeneticDistancePredictor':
        """
        Load model from file.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            Loaded model
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        predictor = cls(
            model_type=data['model_type'],
            n_estimators=data['n_estimators'],
            random_state=data['random_state']
        )
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        
        return predictor


def prepare_distance_data(
    X: pd.DataFrame, 
    reference_distances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pairwise data for distance prediction.
    
    Args:
        X: Feature matrix (organisms × k-mers)
        reference_distances: Pairwise reference distances
        
    Returns:
        X_pairs: Features for each organism pair
        y_pairs: Target distance for each pair
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