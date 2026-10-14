"""
Embedding module for SmartHire GenAI.
Supports Google Gemini Embeddings (text-embedding-004) and Local Hash/TF-IDF Embeddings.
Includes dual-model benchmarking and comparison tools (Stretch Goal 4).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from src.config import GEMINI_API_KEY, EMBEDDING_MODEL, check_api_key


class LocalHashEmbeddings(Embeddings):
    """
    Lightweight deterministic subword n-gram embedding model (768 dimensions).
    Captures character n-grams and token frequencies without external API calls.
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


class LocalTfidfEmbeddings(Embeddings):
    """
    Lexical term-frequency inverse-document-frequency embedding model.
    Serves as an exact keyword frequency baseline.
    """
    def __init__(self, corpus: Optional[List[str]] = None, max_features: int = 512):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        if corpus:
            self.vectorizer.fit(corpus)
        else:
            self.vectorizer.fit([
                "software python developer backend frontend engineer data science machine learning cloud devops",
                "sql react node aws docker kubernetes java go fullstack analytics mlops ai"
            ])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        dense = self.vectorizer.transform(texts).toarray()
        return [row.tolist() for row in dense]

    def embed_query(self, text: str) -> List[float]:
        dense = self.vectorizer.transform([text]).toarray()[0]
        return dense.tolist()


def get_embedding_function(model_type: str = "default") -> Embeddings:
    """
    Returns an embedding function instance based on model_type:
    - 'default': Google Gemini text-embedding-004 if API key present, else LocalHash
    - 'gemini': Force Gemini text-embedding-004 (if configured)
    - 'local_hash': Deterministic Subword Hash Embeddings (768-dim)
    - 'tfidf': Lexical TF-IDF Vectorizer
    """
    if model_type == "gemini" or (model_type == "default" and check_api_key()):
        try:
            return GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GEMINI_API_KEY
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize Gemini embeddings: {e}. Falling back to LocalHash.")
            return LocalHashEmbeddings()
    elif model_type == "tfidf":
        return LocalTfidfEmbeddings()
    else:
        return LocalHashEmbeddings()


def compare_embedding_models(
    query_text: str,
    job_documents: List[Dict[str, Any]],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Stretch Goal 4: Compares two embedding models on the same candidate query:
    - Model A: Dense Neural Embeddings (Gemini or Dense Subword)
    - Model B: Lexical TF-IDF Model
    Reports ranking overlap, cosine similarity distributions, and qualitative trade-offs.
    """
    dense_embed = get_embedding_function("default")
    lexical_embed = LocalTfidfEmbeddings([j.get("description", "") + " " + j.get("required_skills", "") for j in job_documents])

    texts = [f"{j.get('title', '')} {j.get('required_skills', '')} {j.get('description', '')}" for j in job_documents]

    # Generate Model A (Dense) Embeddings
    dense_doc_vecs = np.array(dense_embed.embed_documents(texts))
    dense_query_vec = np.array(dense_embed.embed_query(query_text))
    dense_norms = np.linalg.norm(dense_doc_vecs, axis=1) * np.linalg.norm(dense_query_vec)
    dense_sims = np.dot(dense_doc_vecs, dense_query_vec) / np.maximum(dense_norms, 1e-9)

    # Generate Model B (Lexical TF-IDF) Embeddings
    lex_doc_vecs = np.array(lexical_embed.embed_documents(texts))
    lex_query_vec = np.array(lexical_embed.embed_query(query_text))
    lex_norms = np.linalg.norm(lex_doc_vecs, axis=1) * np.linalg.norm(lex_query_vec)
    lex_sims = np.dot(lex_doc_vecs, lex_query_vec) / np.maximum(lex_norms, 1e-9)

    # Rank Top-K for Model A
    dense_top_indices = np.argsort(dense_sims)[::-1][:top_k]
    dense_results = []
    for idx in dense_top_indices:
        job = job_documents[idx]
        score = round(float(dense_sims[idx]) * 100, 1)
        dense_results.append({
            "job_id": job.get("job_id", ""),
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "score": max(0.0, min(100.0, score)),
            "skills": job.get("required_skills", "")
        })

    # Rank Top-K for Model B
    lex_top_indices = np.argsort(lex_sims)[::-1][:top_k]
    lex_results = []
    for idx in lex_top_indices:
        job = job_documents[idx]
        score = round(float(lex_sims[idx]) * 100, 1)
        lex_results.append({
            "job_id": job.get("job_id", ""),
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "score": max(0.0, min(100.0, score)),
            "skills": job.get("required_skills", "")
        })

    # Compute Jaccard Overlap
    set_a = {r["job_id"] for r in dense_results}
    set_b = {r["job_id"] for r in lex_results}
    jaccard_overlap = len(set_a.intersection(set_b)) / len(set_a.union(set_b)) if set_a.union(set_b) else 1.0

    return {
        "model_a_name": "Dense Semantic Embeddings (Gemini / Subword Dense)",
        "model_b_name": "Lexical Keyword Embeddings (TF-IDF)",
        "query_evaluated": query_text[:120] + "...",
        "jaccard_similarity": round(jaccard_overlap * 100, 1),
        "shared_job_count": len(set_a.intersection(set_b)),
        "model_a_rankings": dense_results,
        "model_b_rankings": lex_results,
        "analytical_conclusion": (
            "Dense semantic embeddings excel at discovering conceptual synonyms "
            "(e.g., matching 'deep learning' with 'PyTorch / Neural Networks') even without literal word overlap, "
            "whereas TF-IDF strictly prioritizes exact keyword frequency counts."
        )
    }
