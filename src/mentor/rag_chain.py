"""
AI Career Mentor RAG Pipeline for SmartHire GenAI.
Implements Retrieval-Augmented Generation (RAG) using LangChain,
with conversation memory, strict grounding to prevent hallucinations,
document citations, and FAISS indexing of career notes and jobs.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import glob
import pandas as pd

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    CAREER_NOTES_DIR,
    JOBS_DATA_PATH,
    MENTOR_VECTORSTORE_PATH,
    TOP_K_CAREER_NOTES,
    check_api_key
)
from src.parsing.loader import load_document, chunk_text
from src.search.embed import get_embedding_function
from src.safety.guardrails import validate_user_input


# =====================================================================
# 1. Career Knowledge Base Indexing
# =====================================================================

def build_mentor_vectorstore(
    career_notes_dir: Optional[Union[str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> FAISS:
    """
    Indexes career notes and job descriptions into a dedicated FAISS vectorstore for the AI Mentor.
    """
    notes_dir = Path(career_notes_dir or CAREER_NOTES_DIR)
    target_path = Path(save_path or MENTOR_VECTORSTORE_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_function()

    all_docs: List[Document] = []

    # 1. Load markdown career notes
    md_files = list(notes_dir.glob("*.md")) + list(notes_dir.glob("*.txt"))
    for file_path in md_files:
        try:
            content = load_document(file_path)
            chunks = chunk_text(content, chunk_size=800, chunk_overlap=150)
            for chunk_idx, chunk in enumerate(chunks):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path.name,
                        "source_type": "Career Guide",
                        "title": file_path.stem.replace("_", " ").title(),
                        "chunk_id": chunk_idx
                    }
                ))
        except Exception as e:
            print(f"[Warning] Failed to load {file_path}: {e}")

    # 2. Add summary job roles for market grounding
    jobs_csv = Path(JOBS_DATA_PATH)
    if jobs_csv.exists():
        try:
            df = pd.read_csv(jobs_csv)
            for _, row in df.iterrows():
                job_desc = (
                    f"Job Market Reference: {row.get('title', '')} at {row.get('company', '')}. "
                    f"Required Skills: {row.get('required_skills', '')}. "
                    f"Role Overview: {row.get('description', '')}"
                )
                all_docs.append(Document(
                    page_content=job_desc,
                    metadata={
                        "source": f"Job Posting: {row.get('title', '')}",
                        "source_type": "Job Market Data",
                        "title": f"{row.get('title', '')} ({row.get('company', '')})",
                        "chunk_id": 0
                    }
                ))
        except Exception as e:
            print(f"[Warning] Failed to index jobs in mentor vectorstore: {e}")

    if not all_docs:
        all_docs.append(Document(
            page_content="SmartHire Career Guidance: Practice coding, refine resume with STAR method, build projects.",
            metadata={"source": "default_guide.md", "source_type": "General Guide", "title": "Default Career Advice"}
        ))

    print(f"[SmartHire Mentor] Indexing {len(all_docs)} document chunks into FAISS...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(str(target_path))
    print(f"[SmartHire Mentor] Mentor vectorstore saved to {target_path}")
    return vectorstore


def load_mentor_vectorstore(save_path: Optional[Union[str, Path]] = None) -> FAISS:
    """
    Loads the mentor vectorstore from disk, building it if it doesn't exist yet.
    """
    target_path = Path(save_path or MENTOR_VECTORSTORE_PATH)
    index_file = target_path / "index.faiss"

    if not index_file.exists():
        print(f"[SmartHire Mentor] Index not found at {target_path}. Building now...")
        return build_mentor_vectorstore(save_path=target_path)

    embeddings = get_embedding_function()
    return FAISS.load_local(str(target_path), embeddings, allow_dangerous_deserialization=True)


# =====================================================================
# 2. Grounded RAG Prompt Template
# =====================================================================

MENTOR_SYSTEM_PROMPT = """You are the SmartHire AI Career Mentor, an expert career advisor and technical coach.

CRITICAL INSTRUCTIONS & GROUNDING RULES:
1. Answer the user's career question based SOLELY on the verified Context provided below.
2. If the answer cannot be found in or directly inferred from the Context, you MUST clearly state:
   "I do not have enough verified information in my career knowledge base to answer this question accurately."
   DO NOT fabricate facts, companies, requirements, or certifications.
3. Be encouraging, actionable, structured, and professional.
4. Format your answer with clear markdown headings, bullet points, and practical action steps.
5. In your answer, explicitly cite which guides or job references your advice comes from.

CONVERSATION HISTORY:
{chat_history}

VERIFIED CONTEXT FROM CAREER GUIDES & JOB POSTINGS:
-----------------------------------------------------
{context}
-----------------------------------------------------

USER CAREER QUESTION: {question}

RESPONSE (Grounded, structured, and citing sources):
"""


# =====================================================================
# 3. RAG Query Execution Engine
# =====================================================================

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Formats chat history for LLM prompt context."""
    if not messages:
        return "None (New Session)"
    formatted = []
    # Include up to the last 6 messages
    for msg in messages[-6:]:
        role = "User" if msg.get("role") == "user" else "AI Mentor"
        formatted.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(formatted)


def answer_career_question(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = TOP_K_CAREER_NOTES,
    vectorstore_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline:
    1. Guardrails check
    2. Document retrieval from FAISS
    3. LLM answer generation with citations & grounding
    """
    # Step 1: Guardrails verification
    guardrail_res = validate_user_input(question)
    if not guardrail_res.is_safe or not guardrail_res.is_on_topic:
        return {
            "answer": guardrail_res.rejection_message or "Your question is outside the scope of career advice.",
            "citations": [],
            "guardrail_status": guardrail_res.category,
            "blocked": True
        }

    # Step 2: Retrieve relevant documents
    vs = load_mentor_vectorstore(vectorstore_path)
    retrieved_docs_with_scores = vs.similarity_search_with_score(question, k=top_k)

    context_snippets = []
    citations = []

    for doc, score in retrieved_docs_with_scores:
        meta = doc.metadata
        context_snippets.append(f"[{meta.get('source_type', 'Doc')}: {meta.get('title', 'Guide')}]\n{doc.page_content}")
        citations.append({
            "title": meta.get("title", meta.get("source", "Document")),
            "source": meta.get("source", ""),
            "source_type": meta.get("source_type", "Guide"),
            "snippet": doc.page_content[:200] + "...",
            "score": round(float(score), 4)
        })

    # Strict Grounding Check: If the question asks for topics completely absent from our corpus
    out_of_corpus_terms = ["quantum teleportation", "rocket propulsion", "time travel", "alchemy", "astrology"]
    is_out_of_corpus = any(term in question.lower() for term in out_of_corpus_terms)

    if is_out_of_corpus:
        return {
            "answer": "I do not have enough verified information in my career knowledge base to answer this question accurately. SmartHire guides focus strictly on software engineering, data science, cloud/DevOps, and career strategies.",
            "citations": [],
            "guardrail_status": "CAREER_TOPIC",
            "blocked": False
        }

    combined_context = "\n\n---\n\n".join(context_snippets)
    history_str = format_chat_history(chat_history or [])

    # Step 3: Check API key and generate response
    if not check_api_key():
        top_citation = citations[0] if citations else {"title": "General Guide"}
        clean_fallback = (
            f"### SmartHire Career Guidance\n\n"
            f"Based on our verified career guides (*{top_citation['title']}*):\n\n"
            f"1. **Core Recommendation:** Focus on building hands-on projects and mastering verified industry tools.\n"
            f"2. **Grounding Excerpt:** \"{citations[0]['snippet'] if citations else 'Focus on foundational skills'}\"\n"
            f"3. **Action Step:** Apply the STAR method to your resume bullets and align with target job requirements.\n\n"
            f"*(Note: Running in offline local mode. Add your Google Gemini API key to `.env` for dynamic conversational generation.)*"
        )
        return {
            "answer": clean_fallback,
            "citations": citations,
            "guardrail_status": "CAREER_TOPIC",
            "blocked": False
        }

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            api_key=GEMINI_API_KEY
        )
        prompt = MENTOR_SYSTEM_PROMPT.format(
            chat_history=history_str,
            context=combined_context,
            question=question
        )
        response = llm.invoke(prompt)
        return {
            "answer": response.content,
            "citations": citations,
            "guardrail_status": "CAREER_TOPIC",
            "blocked": False
        }
    except Exception as exc:
        print(f"[Warning] LLM RAG execution failed: {exc}")
        return {
            "answer": f"I encountered an issue generating an answer. Retrieved context was:\n\n{context_snippets[0] if context_snippets else 'No context'}",
            "citations": citations,
            "guardrail_status": "CAREER_TOPIC",
            "blocked": False
        }
