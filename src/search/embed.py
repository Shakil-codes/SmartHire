"""
Embedding module for SmartHire GenAI.
Provides unified embedding generation using Google Gemini Embeddings (text-embedding-004)
with a graceful local fallback for offline development or testing without API keys.
"""

from typing import List
import numpy as np

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.feature_extraction.text import HashingVectorizer

from src.config import GEMINI_API_KEY, EMBEDDING_MODEL, check_api_key


class LocalHashEmbeddings(Embeddings):
    """
    Lightweight deterministic local embedding fallback using hashed n-gram projections.
    Generates 768-dimensional normalized dense vectors without external API calls.
    Used when no API key is provided or for local testing.
    """
    def __init__(self, n_features: int = 768):
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2)
        )

    def _normalize(self, vec: np.ndarray) -> List[float]:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        sparse_mat = self.vectorizer.transform(texts)
        dense_mat = sparse_mat.toarray()
        return [self._normalize(row) for row in dense_mat]

    def embed_query(self, text: str) -> List[float]:
        sparse_vec = self.vectorizer.transform([text]).toarray()[0]
        return self._normalize(sparse_vec)


def get_embedding_function(use_local_fallback: bool = False) -> Embeddings:
    """
    Returns the configured embedding model.
    Uses Google text-embedding-004 if API key is valid; otherwise uses LocalHashEmbeddings.
    """
    if check_api_key() and not use_local_fallback:
        try:
            return GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GEMINI_API_KEY
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize Google Generative AI Embeddings: {e}. Using local fallback.")
            return LocalHashEmbeddings()
    else:
        return LocalHashEmbeddings()
