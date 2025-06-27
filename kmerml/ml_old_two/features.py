import pandas as pd
from pathlib import Path
from typing import Dict, Union
from kmerml.utils.path_utils import find_files

"""
Functions for k-mer based machine learning workflows
This module is designed to handle k-mer files and convert them into a format suitable for machine learning
"""

class KmerFeatureBuilder:
    """Convert k-mer statistics data into ML-ready feature matrices."""
    
    def __init__(self, stats_dir: Union[str, Path] = None):
        """
        Initialize the feature builder.
        
        Args:
            stats_dir: Directory containing k-mer statistics CSV files
        """
        self.stats_dir = Path(stats_dir) if stats_dir else None
        self.feature_matrix = None
        self.organisms = []
        self.kmers = []
    
    def build_from_statistics_files(self, 
        metric: str = "count",
        file_pattern: str = "*kmer_features.csv") -> pd.DataFrame:
        """
        Build feature matrix from k-mer statistics files.
        
        Args:
            metric: Which metric to use as feature value (count, gc_percent, etc.)
            file_pattern: Pattern to match statistics files
        """
        # Find statistics files
        if not self.stats_dir:
            raise ValueError("Statistics directory not set")
            
        stats_files = find_files(self.stats_dir, patterns=[file_pattern], recursive=True)
        
        if not stats_files:
            raise ValueError(f"No statistics files found matching pattern: {file_pattern}")
        
        # Process each statistics file
        organism_data = {}
        for file_path in stats_files:
            organism_id = self._extract_organism_id(file_path)
            
            try:
                df = pd.read_csv(file_path)
                
                
                # Verify required columns exist
                if 'kmer' not in df.columns or metric not in df.columns:
                    available_cols = ', '.join(df.columns)
                    raise ValueError(f"Required columns not found in {file_path}. Available: {available_cols}")
                
                # Create dictionary mapping k-mers to the selected metric
                kmer_values = dict(zip(df['kmer'], df[metric]))
                organism_data[organism_id] = kmer_values
                
            # Filter k-mers by length if specified             
            except Exception as e:
                import traceback
                print(f"Error processing {file_path}: {e}")
                print(traceback.format_exc())
        
        # Build feature matrix
        matrix = self._build_matrix(organism_data)
        
        return matrix
        
    def _extract_organism_id(self, file_path: Path) -> str:
        """Extract organism ID from statistics filename."""
        # For files named like "GCF_000001_kmer_features.csv"
        name_parts = file_path.stem.split('_')
        if len(name_parts) >= 2:
            # Return "GCF_000001" part
            return f"{name_parts[0]}_{name_parts[1]}"
        return file_path.stem
    
    def _build_matrix(self, organism_data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Build feature matrix from organism k-mer dictionaries.
        
        Args:
            organism_data: Dict mapping organism IDs to k-mer value dictionaries
            
        Returns:
            Feature matrix DataFrame
        """
        # Get unique set of all k-mers
        all_kmers = set()
        for kmer_dict in organism_data.values():
            all_kmers.update(kmer_dict.keys())
        
        # Sort k-mers for reproducibility
        all_kmers = sorted(all_kmers)
        
        # Build matrix rows
        rows = []
        self.organisms = []
        
        for organism, kmer_dict in organism_data.items():
            # Create row with values for each k-mer
            row = [kmer_dict.get(kmer, 0) for kmer in all_kmers]
            rows.append(row)
            self.organisms.append(organism)
        
        # Create DataFrame
        self.feature_matrix = pd.DataFrame(rows, index=self.organisms, columns=all_kmers)
        self.kmers = all_kmers
        
        return self.feature_matrix
    
