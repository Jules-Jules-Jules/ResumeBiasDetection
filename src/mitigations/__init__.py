"""
Phase 5: Mitigation Experiments

M0 - No Mitigation (Baseline)
    Current Phase 4 results serve as the reference.

M1 - Name Masking
    Tests whether removing explicit identity tokens reduces bias.
    Masks first names with [NAME] token while preserving structure.

M2 - Name Frequency Normalization
    Tests whether bias is driven by token rarity/complexity.
    Replaces names with frequency-matched equivalents across groups.

Evaluated on three models:
    1. MiniLM-frozen (retrieval baseline)
    2. MiniLM-trained (classifier head)
    3. E5-trained (classifier head)
"""

from .name_masking import mask_resume_text, mask_first_name
from .name_frequency_sets import (
    get_frequency_matched_name,
    replace_name_with_frequency_matched,
    FREQUENCY_MATCHED_NAMES
)

__all__ = [
    'mask_resume_text',
    'mask_first_name',
    'get_frequency_matched_name',
    'replace_name_with_frequency_matched',
    'FREQUENCY_MATCHED_NAMES',
]
