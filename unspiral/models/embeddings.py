"""Shared embedding model singleton using sentence-transformers."""

from __future__ import annotations

import numpy as np
from typing import Optional

_embedder = None


def get_embedder():
    """Return a lazy-loaded SentenceTransformer('all-MiniLM-L6-v2') singleton."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed(text: str) -> np.ndarray:
    """Embed a single text string into a vector."""
    model = get_embedder()
    return model.encode(text, convert_to_numpy=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed multiple texts into a 2-D array of shape (N, dim)."""
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
