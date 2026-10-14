"""
SmartHire GenAI — Resume Matching & AI Career Mentor
Streamlit Web Portal Application
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import json

from src.config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    JOBS_DATA_PATH,
    CAREER_NOTES_DIR,
    RESUMES_DIR,
    REPORTS_DIR,
    TOP_K_JOBS,
    check_api_key
)
from src.parsing.loader import load_document
from src.parsing.resume_parser import parse_resume, CandidateProfile
from src.search.job_search import search_jobs, load_job_vectorstore
from src.generate.cv_suggestions import (
    generate_cv_improvements,
    rewrite_tailored_resume,
    run_prompt_comparison
)
from src.mentor.rag_chain import answer_career_question, load_mentor_vectorstore
from src.safety.guardrails import validate_user_input

# ==========================================
# 1. Page Configuration & Custom CSS Styling
# ==========================================
st.set_page_config(
    page_title="SmartHire GenAI | Resume Matching & Career Mentor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Glassmorphic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 16px;
        padding: 28px 36px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #e0e7ff;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .match-score-badge {
        font-size: 1.25rem;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 12px;
        display: inline-block;
    }
    .score-high {
        background: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    .score-mid {
        background: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }

    .source-citation {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6366f1;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Session State Initialization
# ==========================================
if "current_profile" not in st.session_state:
    st.session_state.current_profile = None

if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = []

if "selected_job" not in st.session_state:
    st.session_state.selected_job = None

if "cv_suggestions" not in st.session_state:
    st.session_state.cv_suggestions = None

if "mentor_chat_history" not in st.session_state:
    st.session_state.mentor_chat_history = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am your **SmartHire AI Career Mentor**. I can help guide your career transitions, answer questions about required industry skills, and evaluate your resume using verified career roadmaps and job market data.\n\nAsk me anything like *'How do I switch to a Data Scientist role?'* or *'What is the Google XYZ formula for resume bullet points?'*!"
        }
    ]


# ==========================================
# 3. Sidebar Configuration & Sample Loader
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("SmartHire GenAI")
    st.caption("Capstone Project · Resume Matching & AI Career Mentor")
    st.markdown("---")

    # API Key Configuration
    st.subheader("🔑 LLM Settings")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=GEMINI_API_KEY if GEMINI_API_KEY else "",
        type="password",
        help="Enter your Google AI Studio API key. If empty, local deterministic heuristics are used."
    )
    if api_key_input and api_key_input != os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input

    model_choice = st.selectbox(
        "Model Selection",
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )

    st.markdown("---")
    st.subheader("📂 Quick Sample Resumes")
    st.caption("Load a pre-configured test resume to test without uploading:")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📊 Alex (Data Science)", use_container_width=True):
            alex_path = PROJECT_ROOT / "data/resumes/resume_alex_data_scientist.pdf"
            if alex_path.exists():
                text = load_document(alex_path)
                with st.spinner("Parsing Alex's Resume..."):
                    st.session_state.current_profile = parse_resume(text, model_name=model_choice)
                    st.session_state.matched_jobs = search_jobs(st.session_state.current_profile.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                st.success("Loaded Alex Rivera!")
                st.rerun()

        if st.button("🌐 Sarah (Full Stack)", use_container_width=True):
            sarah_path = PROJECT_ROOT / "data/resumes/resume_sarah_fullstack_developer.pdf"
            if sarah_path.exists():
                text = load_document(sarah_path)
                with st.spinner("Parsing Sarah's Resume..."):
                    st.session_state.current_profile = parse_resume(text, model_name=model_choice)
                    st.session_state.matched_jobs = search_jobs(st.session_state.current_profile.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                st.success("Loaded Sarah Chen!")
                st.rerun()

    with col_s2:
        if st.button("☁️ Rahul (DevOps)", use_container_width=True):
            rahul_path = PROJECT_ROOT / "data/resumes/resume_rahul_devops_engineer.docx"
            if rahul_path.exists():
                text = load_document(rahul_path)
                with st.spinner("Parsing Rahul's Resume..."):
                    st.session_state.current_profile = parse_resume(text, model_name=model_choice)
                    st.session_state.matched_jobs = search_jobs(st.session_state.current_profile.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                st.success("Loaded Rahul Verma!")
                st.rerun()

        if st.button("📈 Priya (Fresher)", use_container_width=True):
            priya_path = PROJECT_ROOT / "data/resumes/resume_priya_fresher_analyst.pdf"
            if priya_path.exists():
                text = load_document(priya_path)
                with st.spinner("Parsing Priya's Resume..."):
                    st.session_state.current_profile = parse_resume(text, model_name=model_choice)
                    st.session_state.matched_jobs = search_jobs(st.session_state.current_profile.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                st.success("Loaded Priya Sharma!")
                st.rerun()

    st.markdown("---")
    st.subheader("🛡️ Responsible AI Status")
    st.markdown("✅ **Guardrails Layer:** Active")
    st.markdown("✅ **Semantic FAISS Index:** Ready")
    st.markdown("✅ **Document Grounding:** Enforced")


# ==========================================
# 4. Main App Header
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>SmartHire GenAI Portal</h1>
    <p>Semantic Resume Matching, AI CV Optimization & Retrieval-Augmented Career Mentorship</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. Core Navigation Tabs
# ==========================================
tab_parse, tab_jobs, tab_improve, tab_mentor, tab_eval = st.tabs([
    "📄 Resume Parser",
    "🎯 Semantic Job Matching",
    "💡 CV Optimization & Rewrite",
    "🤖 AI Career Mentor (RAG)",
    "📊 System Evaluation"
])


# ----------------------------------------------------
# TAB 1: RESUME PARSER & PROFILE EXTRACTION
# ----------------------------------------------------
with tab_parse:
    st.subheader("📤 Resume Ingestion & Structured Profile Extraction")
    st.write("Upload a candidate CV in PDF or DOCX format. The system extracts a clean structured JSON profile.")

    uploaded_file = st.file_uploader(
        "Drag and drop your CV (PDF or DOCX)",
        type=["pdf", "docx", "doc", "txt"],
        help="Files are processed directly in memory."
    )

    if uploaded_file is not None:
        if st.button("🚀 Parse & Match Candidate CV", type="primary"):
            with st.spinner("Reading document and generating structured profile..."):
                raw_text = load_document(uploaded_file, filename=uploaded_file.name)
                profile = parse_resume(raw_text, model_name=model_choice)
                st.session_state.current_profile = profile

                # Trigger semantic search
                st.session_state.matched_jobs = search_jobs(profile.to_search_text(), top_k=TOP_K_JOBS)
                st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                st.session_state.cv_suggestions = None
                st.success(f"Parsed candidate: **{profile.name}**!")

    # Display Parsed Profile if available
    profile = st.session_state.current_profile
    if profile:
        st.markdown("---")
        st.markdown("### 👤 Candidate Structured Profile")

        col_p1, col_p2, col_p3 = st.columns([1.5, 1, 1])
        with col_p1:
            st.markdown(f"**Name:** {profile.name}")
            st.markdown(f"**Target Role:** `{profile.target_role}`")
            if profile.email:
                st.markdown(f"**Email:** {profile.email}")
            if profile.phone:
                st.markdown(f"**Phone:** {profile.phone}")
            if profile.location:
                st.markdown(f"**Location:** {profile.location}")

        with col_p2:
            st.metric("Estimated Experience", f"{profile.years_of_experience:.1f} yrs")

        with col_p3:
            st.metric("Total Skills Extracted", len(profile.skills))

        st.markdown("#### 🛠️ Extracted Technical Skills")
        skills_html = "".join([f'<span class="badge-pill">{s}</span>' for s in profile.skills])
        st.markdown(skills_html, unsafe_allow_html=True)

        st.markdown("#### 📝 Professional Summary")
        st.info(profile.summary)

        st.markdown("#### 💼 Professional Experience")
        for exp in profile.experience:
            with st.expander(f"📌 {exp.role} @ {exp.company} ({exp.duration})", expanded=True):
                for bullet in exp.description:
                    st.markdown(f"- {bullet}")

        if profile.education:
            st.markdown("#### 🎓 Education")
            for edu in profile.education:
                st.markdown(f"- **{edu.degree}** — {edu.institution} {f'({edu.graduation_year})' if edu.graduation_year else ''}")

        st.markdown("---")
        st.button("👉 Proceed to View Semantic Job Matches", on_click=lambda: st.toast("Switch to Tab 2 to see matching jobs!"))

    else:
        st.info("👈 Upload a resume above or click one of the quick sample buttons in the left sidebar to get started!")


# ----------------------------------------------------
# TAB 2: SEMANTIC JOB SEARCH (FAISS)
# ----------------------------------------------------
with tab_jobs:
    st.subheader("🎯 Semantic Job Matching (FAISS Vector Search)")
    st.write("Jobs ranked by semantic relevance to candidate profile embeddings.")

    matched_jobs = st.session_state.matched_jobs
    if not matched_jobs:
        if st.session_state.current_profile:
            st.session_state.matched_jobs = search_jobs(st.session_state.current_profile.to_search_text(), top_k=TOP_K_JOBS)
            matched_jobs = st.session_state.matched_jobs

    if matched_jobs:
        st.write(f"Displaying top **{len(matched_jobs)}** matching positions:")

        for job in matched_jobs:
            score = job.get("match_score", 0.0)
            score_class = "score-high" if score >= 45 else "score-mid"

            with st.container():
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <h3 style="margin: 0; color: #f8fafc;">{job['title']}</h3>
                            <p style="margin: 4px 0; color: #94a3b8; font-weight: 500;">
                                🏢 {job['company']} &nbsp;•&nbsp; 📍 {job['location']} &nbsp;•&nbsp; 💰 {job['salary_range']}
                            </p>
                        </div>
                        <div class="match-score-badge {score_class}">
                            {score}% Match
                        </div>
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{job['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                col_btn1, col_btn2 = st.columns([1.5, 4])
                with col_btn1:
                    if st.button(f"🎯 Optimize CV for Job #{job['rank']}", key=f"btn_opt_{job['job_id']}"):
                        st.session_state.selected_job = job
                        st.session_state.cv_suggestions = None
                        st.toast(f"Selected {job['title']}! Switch to Tab 3.")
                with col_btn2:
                    st.caption(f"Required Skills: {job['required_skills']}")
                st.markdown("---")

    else:
        st.info("Please upload a resume in Tab 1 to see matching jobs.")


# ----------------------------------------------------
# TAB 3: CV IMPROVEMENT GENERATOR & REWRITE
# ----------------------------------------------------
with tab_improve:
    st.subheader("💡 Actionable CV Improvement & Tailoring")
    profile = st.session_state.current_profile
    selected_job = st.session_state.selected_job

    if not profile:
        st.warning("Please upload a resume in Tab 1 first.")
    elif not selected_job:
        st.warning("Please select a target job from Tab 2.")
    else:
        st.markdown(f"**Target Role:** `{selected_job['title']}` at **{selected_job['company']}**")

        if st.session_state.cv_suggestions is None:
            if st.button("✨ Generate AI CV Improvement Analysis", type="primary"):
                with st.spinner("Analyzing candidate gaps, bullet strengths, and XYZ metrics..."):
                    res = generate_cv_improvements(profile, selected_job, model_name=model_choice)
                    st.session_state.cv_suggestions = res
                    st.rerun()

        suggestions = st.session_state.cv_suggestions
        if suggestions:
            col_fit1, col_fit2 = st.columns([1, 2])
            with col_fit1:
                st.metric("Estimated Role Fit Score", f"{suggestions.overall_fit_score}%")
                st.progress(suggestions.overall_fit_score / 100.0)

            with col_fit2:
                st.markdown("#### 🎯 Skill Overlap & Gap Analysis")
                st.markdown("**Overlapping Skills:** " + "".join([f'<span class="badge-pill badge-green">✓ {s}</span>' for s in suggestions.matching_skills]), unsafe_allow_html=True)
                st.markdown("**Missing Key Skills to Learn/Add:** " + "".join([f'<span class="badge-pill badge-amber">+ {s}</span>' for s in suggestions.missing_hard_skills]), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Bullet Point Critiques & Google XYZ Rewrites")
            st.caption("Google XYZ Formula: 'Accomplished [X] as measured by [Y], by doing [Z]'")

            for idx, critique in enumerate(suggestions.bullet_point_improvements, 1):
                with st.expander(f"Critique #{idx}: {critique.original_bullet[:60]}...", expanded=True):
                    st.markdown(f"**❌ Original Weak Bullet:** `{critique.original_bullet}`")
                    st.markdown(f"**⚠️ Recruiter Critique:** {critique.critique_weakness}")
                    st.success(f"**✅ High-Impact XYZ Rewrite:**\n\n{critique.rewritten_bullet_with_metrics}")

            st.markdown("---")
            st.markdown("### 📝 Tailored Executive Summary")
            st.info(suggestions.tailored_summary)

            st.markdown("### 🎙️ Strategic Interview Talking Points")
            for tp in suggestions.interview_talking_points:
                st.markdown(f"- 🗣️ {tp}")

            # Stretch Goal: Full Resume Tailored Rewrite
            st.markdown("---")
            st.markdown("### 🚀 Stretch Goal: Full Tailored Resume Generation")
            if st.button("📄 Generate Full Tailored Resume for this Position"):
                with st.spinner("Crafting tailored resume in markdown..."):
                    tailored_md = rewrite_tailored_resume(profile, selected_job, model_name=model_choice)
                    st.markdown(tailored_md)
                    st.download_button("💾 Download Tailored Resume (.md)", tailored_md, file_name="tailored_resume.md")


# ----------------------------------------------------
# TAB 4: AI CAREER MENTOR (RAG)
# ----------------------------------------------------
with tab_mentor:
    st.subheader("🤖 AI Career Mentor (RAG)")
    st.caption("Grounded in verified career roadmaps, interview strategies, and real job postings.")

    # Render Conversation History
    for msg in st.session_state.mentor_chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Sources & References Cited"):
                    for c in msg["citations"]:
                        st.markdown(f"**• {c['title']}** ({c['source_type']})\n*\"{c['snippet']}\"*")

    # Prompt Suggestions
    st.markdown("**Suggested Questions:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    suggested_q = None
    with col_q1:
        if st.button("🚀 How do I switch to Data Science?"):
            suggested_q = "How do I switch into Data Science and what are the phases?"
    with col_q2:
        if st.button("🎯 Google XYZ Resume Formula"):
            suggested_q = "Explain the Google XYZ formula for resume bullet points with examples."
    with col_q3:
        if st.button("☁️ DevOps Tools Roadmap"):
            suggested_q = "What core tools should I master for DevOps and Cloud infrastructure?"

    # Chat Input
    user_query = st.chat_input("Ask your career mentor a question...")
    query_to_run = user_query or suggested_q

    if query_to_run:
        # Add user message
        st.session_state.mentor_chat_history.append({"role": "user", "content": query_to_run})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query_to_run)

        # Process with RAG & Guardrails
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Consulting career knowledge base and verifying safety guardrails..."):
                resp = answer_career_question(
                    query_to_run,
                    chat_history=st.session_state.mentor_chat_history
                )
                st.markdown(resp["answer"])

                if resp.get("citations"):
                    with st.expander("📚 Sources & References Cited", expanded=False):
                        for c in resp["citations"]:
                            st.markdown(f"**• {c['title']}** ({c['source_type']})\n*\"{c['snippet']}\"*")

                st.session_state.mentor_chat_history.append({
                    "role": "assistant",
                    "content": resp["answer"],
                    "citations": resp.get("citations", [])
                })

    if st.button("🗑️ Clear Chat History"):
        st.session_state.mentor_chat_history = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I help your career journey today?"
            }
        ]
        st.rerun()


# ----------------------------------------------------
# TAB 5: SYSTEM EVALUATION & BENCHMARKS
# ----------------------------------------------------
with tab_eval:
    st.subheader("📊 System Evaluation & Answer Quality Benchmarks")
    st.write("Quantitative metrics for semantic retrieval, grounding, hallucination prevention, and prompt optimization.")

    eval_file = REPORTS_DIR / "answer_quality.md"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_content = f.read()
        st.markdown(eval_content)
    else:
        st.warning("Evaluation report not yet generated. Run `python -m src.evaluate` to generate it.")
        if st.button("▶️ Run Full Evaluation Suite Now"):
            with st.spinner("Running benchmarks..."):
                from src.evaluate import run_full_evaluation
                content = run_full_evaluation()
                st.success("Evaluation complete!")
                st.markdown(content)
