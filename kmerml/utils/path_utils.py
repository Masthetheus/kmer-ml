import os
from pathlib import Path

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