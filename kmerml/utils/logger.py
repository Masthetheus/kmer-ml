import logging
import os
from pathlib import Path

def setup_logging(name=None, log_dir="data/logs", log_level=logging.INFO):
    """Set up logging with automatic directory creation and script-specific logging."""
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Use script name in the log file name if provided
    log_file = os.path.join(log_dir, f"kmerml{'-' + name if name else ''}.log")
    
    # Create a logger with the given name
    logger_name = f"kmerml{('.' + name) if name else ''}"
    logger = logging.getLogger(logger_name)
    
    # Only add handlers if they don't exist to prevent duplicates
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Create formatters and handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger

def get_logger(name):
    """Get a logger with the given name. If the logger doesn't exist, it creates a minimal one."""
    logger = logging.getLogger(f"kmerml.{name}")
    
    # If this logger hasn't been set up yet, at least set the level
    if not logger.handlers and not logger.parent.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
        logger.propagate = False
        
    return logger