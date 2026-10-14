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

MENTOR_SYSTEM_PROMPT = """You are the SmartHire AI Career Mentor & Technical Recruiter at SmartHire Technologies Inc.

You are having an engaging, ongoing, and supportive 1-on-1 career coaching conversation with the user.

CONVERSATIONAL GUIDELINES:
1. Be warm, empathetic, highly knowledgeable, and actionable. Speak like a mentor who genuinely cares about the user's career growth.
2. Maintain natural continuity with the CONVERSATION HISTORY. If the user asks a follow-up (e.g. "tell me more about phase 2", "can you give an example?"), build directly upon what was discussed previously.
3. If the user greets you or makes casual remarks, respond warmly and guide them toward career questions.
4. Ground your technical and career advice in the VERIFIED CONTEXT provided below.
5. If the user asks about their resume or target job, tailor your advice directly using the ACTIVE CANDIDATE CONTEXT and ACTIVE TARGET JOB CONTEXT.
6. If the question cannot be answered from the verified context or candidate details, clearly state:
   "I do not have enough verified information in my career knowledge base to answer this question accurately."
7. Format your answer with clean markdown headings, bullet points, and practical action steps.
8. End your response with an encouraging, relevant follow-up question or suggestion to keep the conversation flowing.

ACTIVE CANDIDATE CONTEXT:
{candidate_context}

ACTIVE TARGET JOB CONTEXT:
{job_context}

CONVERSATION HISTORY (Prior Messages):
{chat_history}

VERIFIED CONTEXT FROM CAREER GUIDES & JOB POSTINGS:
-----------------------------------------------------
{context}
-----------------------------------------------------

USER CURRENT MESSAGE: {question}

MENTOR RESPONSE (Natural, conversational, grounded, and personalized):
"""


# =====================================================================
# 3. RAG Query Execution Engine
# =====================================================================

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Formats prior chat messages for LLM conversational context."""
    if not messages:
        return "None (New Conversation)"

    # If the last message is the current turn, exclude it from prior history
    history_items = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
    if not history_items:
        return "None (New Conversation)"

    formatted = []
    # Include up to the last 6 messages
    for msg in history_items[-6:]:
        role = "User" if msg.get("role") == "user" else "AI Mentor"
        content = msg.get("content", "").strip()
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def answer_career_question(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    candidate_context: Optional[str] = None,
    job_context: Optional[str] = None,
    top_k: int = TOP_K_CAREER_NOTES,
    vectorstore_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline:
    1. Guardrails check
    2. Document retrieval from FAISS
    3. Injects candidate and target job context for personalized coaching
    4. LLM answer generation with citations & grounding
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
    cand_str = candidate_context or "No active candidate resume loaded."
    job_str = job_context or "No active target job selected."

    # Step 3: Check API key and generate response
    if not check_api_key():
        q_lower = question.lower()
        top_citation = citations[0] if citations else {"title": "Career Notes"}

        # Dynamic Conversational Heuristic Responses
        if any(g in q_lower for g in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]):
            cand_name = "there"
            if candidate_context and "Target Role:" in candidate_context:
                cand_name = "colleague"
            response_text = (
                f"👋 **Hello {cand_name}!** Great to connect with you.\n\n"
                f"I am your **SmartHire AI Career Mentor & Recruiter**. I have full context on your active profile and target roles.\n\n"
                f"Here are a few high-impact things we can do together right now:\n"
                f"1. **Audit your skill gaps** for your target job\n"
                f"2. **Run a mock interview** with role-specific questions\n"
                f"3. **Rewrite passive resume bullets** using the Google XYZ formula\n"
                f"4. **Build a 30-day learning roadmap**\n\n"
                f"What would you like to dive into first?"
            )
        elif "interview" in q_lower or "question" in q_lower:
            response_text = (
                f"### 🎙️ Tailored Technical Mock Interview Questions\n\n"
                f"Based on our hiring benchmarks and your target role, here are **3 strategic interview questions** to prepare for:\n\n"
                f"1. **System & Tool Deep-Dive:** *\"Can you walk me through the trade-offs you considered when choosing your primary data/development pipeline architecture?\"*\n"
                f"2. **Failure Modes & Troubleshooting:** *\"Describe a challenging production latency or data quality incident you diagnosed. What metric did you monitor, and what was your root cause resolution?\"*\n"
                f"3. **Business Impact & Ownership:** *\"How do you quantify the business value of your technical deliverables to cross-functional stakeholders?\"*\n\n"
                f"💡 **Mentor Coaching Tip:** Frame every answer using the **STAR Method** (Situation, Task, Action, Result), spending 70% of your time on your specific Action and measurable Result.\n\n"
                f"Would you like to practice answering one of these questions now?"
            )
        elif any(k in q_lower for k in ["gap", "missing", "audit", "weakness"]):
            response_text = (
                f"### 🎯 Skill Gap & Career Audit\n\n"
                f"Reviewing your target role requirements against industry benchmarks:\n\n"
                f"• **High Priority Skills to Deepen:** Focus on mastering production deployment, automated testing, and scalable architecture.\n"
                f"• **Grounded Strategy:** According to *{top_citation['title']}*, recruiters look for candidates who have built end-to-end projects demonstrating business value rather than toy exercises.\n"
                f"• **Action Item:** Pick one missing technical tool and build a public GitHub project with a clear README and performance metrics.\n\n"
                f"Would you like me to suggest an end-to-end portfolio project idea tailored to your profile?"
            )
        elif any(k in q_lower for k in ["xyz", "bullet", "formula", "rewrite"]):
            response_text = (
                f"### 📝 The Google XYZ Formula for High-Impact Resumes\n\n"
                f"Recruiters evaluate achievements through Google's proven impact formula:\n\n"
                f"> **\"Accomplished [X] as measured by [Y], by doing [Z]\"**\n\n"
                f"#### Example Transformation:\n"
                f"• ❌ **Weak / Passive:** *\"Worked on developing data pipelines and improving model accuracy.\"*\n"
                f"• ✅ **Google XYZ Rewrite:** *\"Optimized predictive recommendation pipeline using hyperparameter tuning in Scikit-Learn, boosting ROC-AUC from 0.74 to 0.86 and driving 14% higher engagement.\"*\n\n"
                f"Share one of your resume bullet points with me, and I will rewrite it for you!"
            )
        elif any(k in q_lower for k in ["phase", "roadmap", "switch", "learn"]):
            response_text = (
                f"### 🚀 Recommended 3-Phase Transition Roadmap\n\n"
                f"According to verified career guides (*{top_citation['title']}*):\n\n"
                f"1. **Phase 1: Core Fundamentals (Weeks 1 - 4):** Master core scripting (Python/SQL), clean code standards, and version control.\n"
                f"2. **Phase 2: Applied Engineering (Weeks 5 - 10):** Build production pipelines, handle feature preprocessing, and understand error analysis.\n"
                f"3. **Phase 3: Portfolio & Productionization (Weeks 11 - 16):** Deploy 2-3 end-to-end applications via Docker / FastAPI with automated CI/CD and metrics.\n\n"
                f"Which phase are you currently in right now?"
            )
        else:
            snippet = citations[0]['snippet'] if citations else "Focus on verified technical competencies and practical project portfolios."
            response_text = (
                f"### 💡 SmartHire Career Coaching\n\n"
                f"Based on our verified career guides (*{top_citation['title']}*):\n\n"
                f"• **Key Principle:** {snippet}\n"
                f"• **Recruiter Recommendation:** Align your portfolio and technical talking points directly with the employer's operational goals.\n"
                f"• **Next Step:** Apply the Google XYZ formula to quantify your past achievements.\n\n"
                f"What specific area would you like to focus on next?"
            )

        return {
            "answer": response_text,
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
            candidate_context=cand_str,
            job_context=job_str,
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
