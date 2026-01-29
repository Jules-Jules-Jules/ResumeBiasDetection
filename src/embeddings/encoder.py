import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config.paths import DATA_DIR


class SentenceEncoder:
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        normalize: bool = True
    ):
        print(f"Loading pretrained model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.normalize = normalize
        
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Embedding cache enabled: {cache_dir}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        if self.cache_dir:
            embeddings, missing_indices = self._load_from_cache(texts)
            if not missing_indices:
                return embeddings
            
            missing_texts = [texts[i] for i in missing_indices]
            missing_embeds = self._encode_batch(
                missing_texts,
                batch_size,
                show_progress
            )
            
            for idx, embed in zip(missing_indices, missing_embeds):
                self._save_to_cache(texts[idx], embed)
            
            for idx, embed in zip(missing_indices, missing_embeds):
                embeddings[idx] = embed
            
            return embeddings
        else:
            return self._encode_batch(texts, batch_size, show_progress)
    
    def _encode_batch(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _cache_path(self, text: str) -> Path:
        text_hash = self._text_hash(text)
        return self.cache_dir / f"{text_hash}.npy"
    
    def _load_from_cache(
        self,
        texts: List[str]
    ) -> tuple[np.ndarray, List[int]]:
        dim = self.model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        missing = []
        
        for i, text in enumerate(texts):
            cache_path = self._cache_path(text)
            if cache_path.exists():
                embeddings[i] = np.load(cache_path)
            else:
                missing.append(i)
        
        return embeddings, missing
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        cache_path = self._cache_path(text)
        np.save(cache_path, embedding)
    
    def cosine_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        if self.normalize:
            return embeddings_a @ embeddings_b.T
        else:
            norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
            norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)
            normalized_a = embeddings_a / (norms_a + 1e-8)
            normalized_b = embeddings_b / (norms_b + 1e-8)
            return normalized_a @ normalized_b.T
