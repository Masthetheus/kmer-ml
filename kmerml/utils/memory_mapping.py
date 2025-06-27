import numpy as np
import pandas as pd
import os
from pathlib import Path

def create_shared_kmer_dataset(feature_dir, k_value=10, metric="shannon_entropy"):
    """Create memory-mapped files for k-mer data that can be shared between processes"""
    from kmerml.ml.features import KmerFeatureBuilder
    
    # Create directory for memory-mapped files
    mmap_dir = Path("data/mmap_features")
    mmap_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths for memory-mapped files
    feature_matrix_path = mmap_dir / f"k{k_value}_{metric}_features.npy"
    row_index_path = mmap_dir / f"k{k_value}_{metric}_row_index.txt"
    col_names_path = mmap_dir / f"k{k_value}_{metric}_col_names.txt"
    
    # Check if files already exist
    if feature_matrix_path.exists():
        print(f"Memory-mapped feature matrix already exists at {feature_matrix_path}")
    else:
        print(f"Creating memory-mapped feature matrix for k={k_value}, metric={metric}...")
        
        # Load feature matrix normally first
        builder = KmerFeatureBuilder(feature_dir)
        matrix = builder.build_from_statistics_files(metric=metric)
        
        # Save row and column metadata
        with open(row_index_path, 'w') as f:
            f.write('\n'.join(matrix.index.astype(str)))
        
        with open(col_names_path, 'w') as f:
            f.write('\n'.join(matrix.columns.astype(str)))
        
        # Convert to float32 to save memory
        np_array = matrix.values.astype(np.float32)
        
        # Save as memory-mapped file
        np.save(feature_matrix_path, np_array)
        
        # Clear memory
        del matrix, np_array
        import gc
        gc.collect()
    
    # Return paths to the memory-mapped files
    return {
        'data_path': str(feature_matrix_path),
        'row_index_path': str(row_index_path),
        'col_names_path': str(col_names_path)
    }

def load_shared_kmer_data(mmap_info):
    """Load data from memory-mapped files into a DataFrame"""
    # Load row and column metadata
    with open(mmap_info['row_index_path'], 'r') as f:
        row_index = f.read().splitlines()
    
    with open(mmap_info['col_names_path'], 'r') as f:
        col_names = f.read().splitlines()
    
    # Load data in memory-map mode
    data = np.load(mmap_info['data_path'], mmap_mode='r')
    
    # Create DataFrame (this doesn't copy the data)
    df = pd.DataFrame(
        data,  # This is a view, not a copy
        index=row_index,
        columns=col_names
    )
    
    return df