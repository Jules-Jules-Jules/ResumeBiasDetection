"""
Utils: Logging

Consistent logging configuration across the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_logger(
    name: str,
    log_path: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        log_path: Optional path to log file. If None, logs only to console
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Format: timestamp - level - logger name - message
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always added)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamped_log_path(base_dir: str | Path, prefix: str = "log") -> Path:
    """
    Generate a timestamped log file path.
    
    Args:
        base_dir: Directory to store log file
        prefix: Prefix for log filename
        
    Returns:
        Path to timestamped log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.log"
    return Path(base_dir) / filename
