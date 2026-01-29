from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class TfidfEncoder:
    
    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        normalize: bool = True
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english',
            norm='l2' if normalize else None
        )
        self.is_fitted = False
        self.normalize = normalize
    
    def fit(self, texts: List[str]):
        print(f"Fitting TF-IDF on {len(texts)} documents")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"Vocabulary size: {vocab_size}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = False
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must call fit() before encode()")
        
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()
    
    def fit_encode(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.encode(texts)
    
    def cosine_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        return sklearn_cosine(embeddings_a, embeddings_b)
