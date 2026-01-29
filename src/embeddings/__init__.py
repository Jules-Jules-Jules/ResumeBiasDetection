"""
Embedding encoders for resume and job text.

Provides both neural (sentence transformers) and classical (TF-IDF) encoders
with a consistent interface for scoring.
"""

from .encoder import SentenceEncoder
from .tfidf import TfidfEncoder

__all__ = ['SentenceEncoder', 'TfidfEncoder']
