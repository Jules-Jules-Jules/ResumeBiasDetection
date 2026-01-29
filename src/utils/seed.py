"""
Utils: Seed Management

Ensures reproducible experiments by setting seeds across all libraries.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon) operations
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
