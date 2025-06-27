import os
from pathlib import Path
import pandas as pd

def find_files(directory, patterns=None, recursive=False):
    """
    Find files matching specified patterns in a directory.
    
    Args:
        directory (str or Path): Directory to search
        patterns (list): List of glob patterns to match (e.g., ["*.fa", "*.fasta"])
                         Defaults to ["*"] (all files)
        recursive (bool): Whether to search subdirectories
    
    Returns:
        list: List of Path objects to matching files
    """
    
    directory = Path(directory)
    
    if patterns is None:
        patterns = ["*"]
    
    # Determine search depth
    base_pattern = "**/" if recursive else ""
    
    matching_files = []
    for pattern in patterns:
        # Combine base pattern with file pattern
        search_pattern = f"{base_pattern}{pattern}"
        matching_files.extend(list(directory.glob(search_pattern)))
    
    return sorted(matching_files)

def ensure_directory_exists(directory_path):
    """
    Check if directory exists and create it if it doesn't.
    
    Args:
        directory_path (str or Path): Directory path to check/create
    
    Returns:
        Path: Path object of the directory
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def is_valid_file(file_path):
    """
    Check if a file exists and is readable.
    
    Args:
        file_path (str or Path): File path to check
    
    Returns:
        bool: True if file exists and is readable, False otherwise
    """
    path = Path(file_path)
    return path.exists() and path.is_file() and os.access(path, os.R_OK)

def get_accession_codes_from_tsv(tsv_path, accession_column="Assembly Accession"):
    """
    Reads a TSV file and returns a list of accession codes from the specified column,
    converting dots to underscores (e.g., GCF_035610405.1 -> GCF_035610405_1).

    Args:
        tsv_path (str or Path): Path to the TSV file.
        accession_column (str): Name of the column containing accession codes.

    Returns:
        list: List of accession codes as strings with dots replaced by underscores.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    # Replace dots with underscores in accession codes
    return df[accession_column].dropna().astype(str).str.replace('.', '_', regex=False).tolist()