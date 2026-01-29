"""
Configuration: Paths

Centralized path definitions for the project.
All file I/O should reference these paths to avoid hardcoding.
"""

import os
from pathlib import Path

# Detect project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Docs
DOCS_DIR = PROJECT_ROOT / "docs"

# Tests
TESTS_DIR = PROJECT_ROOT / "tests"


def ensure_dirs():
    """
    Create all required directories if they don't exist.
    Should be called during initialization or smoke tests.
    """
    dirs = [
        DATA_DIR,
        DATA_RAW,
        DATA_PROCESSED,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        DOCS_DIR,
        TESTS_DIR,
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return True


def get_path_summary() -> dict:
    """
    Return a dictionary of all key paths for debugging/logging.
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "data_raw": str(DATA_RAW),
        "data_processed": str(DATA_PROCESSED),
        "models_dir": str(MODELS_DIR),
        "notebooks_dir": str(NOTEBOOKS_DIR),
        "docs_dir": str(DOCS_DIR),
    }
