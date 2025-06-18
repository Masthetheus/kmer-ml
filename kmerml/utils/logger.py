import os
import logging
from pathlib import Path

def setup_logging(log_dir="data/logs", log_level=logging.INFO):
    """Set up logging with automatic directory creation."""
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure file handler
    log_file = os.path.join(log_dir, "kmerml.log")
    
    # Set up logging configuration
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger("kmerml")

def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(f"kmerml.{name}")