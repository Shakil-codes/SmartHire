"""
Job Search Module for SmartHire GenAI.
Builds and queries FAISS vector indices for semantic job recommendation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import (
    JOBS_DATA_PATH,
    JOB_VECTORSTORE_PATH,
    TOP_K_JOBS,
)
from src.search.embed import get_embedding_function


def prepare_job_documents(df: pd.DataFrame) -> List[Document]:
    """
    Transforms DataFrame job rows into LangChain Document objects with rich metadata
    and unified semantic text representation.
    """
    docs = []
    for _, row in df.iterrows():
        job_id = str(row.get("job_id", ""))
        title = str(row.get("title", ""))
        company = str(row.get("company", ""))
        location = str(row.get("location", ""))
        exp_level = str(row.get("experience_level", ""))
        skills = str(row.get("required_skills", ""))
        desc = str(row.get("description", ""))
        salary = str(row.get("salary_range", "Competitive"))
        emp_type = str(row.get("employment_type", "Full-time"))

        # Semantic content combines all relevant facets
        content = (
            f"Job Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Experience Level: {exp_level}\n"
            f"Required Skills: {skills}\n"
            f"Role Description: {desc}"
        )

        metadata = {
            "job_id": job_id,
            "title": title,
            "company": company,
            "location": location,
            "experience_level": exp_level,
            "required_skills": skills,
            "description": desc,
            "salary_range": salary,
            "employment_type": emp_type
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_job_vectorstore(
    csv_path: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False
) -> FAISS:
    """
    Builds and persists a FAISS vector index from the jobs dataset CSV.
    """
    csv_file = Path(csv_path or JOBS_DATA_PATH)
    target_path = Path(save_path or JOB_VECTORSTORE_PATH)

    if not csv_file.exists():
        raise FileNotFoundError(f"Jobs CSV not found at: {csv_file}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_function()

    print(f"[SmartHire] Reading job listings from {csv_file}...")
    df = pd.read_csv(csv_file)
    docs = prepare_job_documents(df)

    print(f"[SmartHire] Generating embeddings and building FAISS vectorstore for {len(docs)} jobs...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print(f"[SmartHire] Saving vectorstore index to {target_path}...")
    vectorstore.save_local(str(target_path))
    print("[SmartHire] Job vectorstore successfully built and saved!")
    return vectorstore


def load_job_vectorstore(save_path: Optional[Union[str, Path]] = None) -> FAISS:
    """
    Loads an existing FAISS vectorstore from disk.
    If the index does not exist, it builds it automatically.
    """
    target_path = Path(save_path or JOB_VECTORSTORE_PATH)
    index_file = target_path / "index.faiss"

    if not index_file.exists():
        print(f"[SmartHire] Index not found at {target_path}. Building now...")
        return build_job_vectorstore(save_path=target_path)

    embeddings = get_embedding_function()
    return FAISS.load_local(str(target_path), embeddings, allow_dangerous_deserialization=True)


def calculate_match_percentage(score: float, is_distance: bool = True) -> float:
    """
    Converts raw vector similarity/distance scores into an intuitive 0-100% match score.
    For L2 distance, smaller is closer. Score formula: 1 / (1 + distance).
    """
    if is_distance:
        # Distance >= 0
        similarity = 1.0 / (1.0 + max(0.0, score))
    else:
        # Cosine / dot product [-1, 1] or [0, 1]
        similarity = max(0.0, min(1.0, score))
    return round(similarity * 100, 1)


def search_jobs(
    query_text: str,
    top_k: int = TOP_K_JOBS,
    vectorstore_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search over the jobs index.
    Returns ranked jobs with normalized percentage match scores and full metadata.
    """
    vs = load_job_vectorstore(vectorstore_path)
    # Search with score (LangChain FAISS returns (doc, L2_distance))
    results = vs.similarity_search_with_score(query_text, k=top_k)

    ranked_jobs = []
    for rank, (doc, raw_score) in enumerate(results, start=1):
        meta = doc.metadata
        match_pct = calculate_match_percentage(raw_score, is_distance=True)

        ranked_jobs.append({
            "rank": rank,
            "job_id": meta.get("job_id", ""),
            "title": meta.get("title", ""),
            "company": meta.get("company", ""),
            "location": meta.get("location", ""),
            "experience_level": meta.get("experience_level", ""),
            "required_skills": meta.get("required_skills", ""),
            "description": meta.get("description", ""),
            "salary_range": meta.get("salary_range", "Competitive"),
            "employment_type": meta.get("employment_type", "Full-time"),
            "match_score": match_pct,
            "raw_distance": round(float(raw_score), 4)
        })

    return ranked_jobs
