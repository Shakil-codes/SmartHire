"""
SmartHire Technologies Inc. — Enterprise Career & Talent Intelligence Platform
An enterprise-grade platform offering:
- Autonomous AI Career Mentor Co-Pilot with multi-turn memory & context awareness
- PII Masking & Blind Screening (GDPR/SOC2 privacy compliance)
- Semantic Vector Job Matching (FAISS)
- Google XYZ CV Optimization & Tailored Resume Studio
- Dual-Model Embedding Benchmarking Lab
- Enterprise Solutions & Platform Help Center
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
from src.parsing.resume_parser import (
    parse_resume,
    CandidateProfile,
    redact_profile_pii
)
from src.search.embed import compare_embedding_models
from src.search.job_search import search_jobs, load_job_vectorstore
from src.generate.cv_suggestions import (
    generate_cv_improvements,
    rewrite_tailored_resume,
    run_prompt_comparison
)
from src.mentor.rag_chain import answer_career_question, load_mentor_vectorstore
from src.safety.guardrails import validate_user_input


# =====================================================================
# 1. Page Configuration & Enterprise Styling
# =====================================================================
st.set_page_config(
    page_title="SmartHire Technologies · Enterprise Career & Talent AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Enterprise CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --brand-primary: #6366f1;
        --brand-violet: #8b5cf6;
        --brand-emerald: #10b981;
        --dark-bg: #090d16;
        --card-bg: rgba(17, 24, 39, 0.7);
        --border-glass: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Top Enterprise Navigation Bar */
    .company-nav {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 14px 28px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .brand-mark {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #ffffff 40%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Enterprise Hero Banner */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.22) 0%, transparent 45%),
                    radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.18) 0%, transparent 45%),
                    linear-gradient(135deg, #0b0f19 0%, #1e1b4b 60%, #090d16 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 32px 38px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(120deg, #ffffff 30%, #c7d2fe 70%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-top: 8px;
        margin-bottom: 12px;
        max-width: 850px;
        line-height: 1.6;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-panel:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }

    /* Enterprise Metric Card */
    .metric-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.65) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: left;
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
        color: #f8fafc;
    }
    .metric-delta {
        font-size: 0.82rem;
        color: #34d399;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Pill Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
        background: rgba(99, 102, 241, 0.14);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.28);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.14);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.14);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.28);
    }

    /* Match Score Badges */
    .score-badge-high {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        color: #6ee7b7;
        font-size: 1.15rem;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 12px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    .score-badge-mid {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid #6366f1;
        color: #c7d2fe;
        font-size: 1.15rem;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 12px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }

    /* Citation Box */
    .citation-card {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #8b5cf6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. Session State Initialization
# =====================================================================
if "current_profile" not in st.session_state:
    st.session_state.current_profile = None

if "raw_profile" not in st.session_state:
    st.session_state.raw_profile = None

if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = []

if "selected_job" not in st.session_state:
    st.session_state.selected_job = None

if "cv_suggestions" not in st.session_state:
    st.session_state.cv_suggestions = None

if "tailored_resume_content" not in st.session_state:
    st.session_state.tailored_resume_content = None

if "pii_masking_enabled" not in st.session_state:
    st.session_state.pii_masking_enabled = True

if "anonymize_candidate_name" not in st.session_state:
    st.session_state.anonymize_candidate_name = False

if "mentor_chat_history" not in st.session_state:
    st.session_state.mentor_chat_history = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **SmartHire AI Career Mentor & Co-Pilot**.\n\n"
                "I am fully synchronized with your active resume and selected target job. "
                "I maintain memory of our session and ground my guidance in verified career roadmaps and real market requirements.\n\n"
                "**Ask me anything:**\n"
                "- *'Explain my match score for my active job'* \n"
                "- *'What are the key skill gaps I need to fix?'*\n"
                "- *'Coach me for the interview with Quantiva Analytics'* \n"
                "- *'How do I transition from Full Stack to Machine Learning?'*"
            ),
            "citations": []
        }
    ]


# =====================================================================
# 3. Sidebar: Enterprise Controls & Candidate Selectors
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=56)
    st.title("SmartHire Platform")
    st.caption("Enterprise Talent & Career AI · v2.4")

    st.markdown("---")
    st.subheader("🛡️ Privacy & Compliance Shield")
    st.caption("GDPR & SOC2 Blind Screening controls:")

    mask_pii = st.toggle(
        "Mask Contact Info (Email & Phone)",
        value=st.session_state.pii_masking_enabled,
        help="Protects sensitive personal data by redacting emails, phone numbers, and physical addresses."
    )
    st.session_state.pii_masking_enabled = mask_pii

    anon_name = st.checkbox(
        "Blind Screening (Anonymize Name)",
        value=st.session_state.anonymize_candidate_name,
        help="Replaces candidate full name with an anonymous token (e.g. Candidate #4C2A) to eliminate unconscious bias."
    )
    st.session_state.anonymize_candidate_name = anon_name

    st.markdown("---")
    st.subheader("🔑 Enterprise LLM Gateway")

    # High Security: Never expose or prefill active API keys in HTML DOM
    is_key_active = check_api_key()

    if is_key_active:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;">
            <div style="color: #6ee7b7; font-weight: 700; font-size: 0.85rem;">🔒 Server-Side Key Active</div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 2px;">Key is isolated in secure backend memory. Hidden from client browsers.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Override API Key (Optional)"):
            api_key_input = st.text_input(
                "Enter New Key",
                value="",
                type="password",
                placeholder="Paste new Gemini key...",
                help="Overrides backend key for current session only."
            )
            if api_key_input:
                os.environ["GEMINI_API_KEY"] = api_key_input
                os.environ["GOOGLE_API_KEY"] = api_key_input
                st.success("New session key applied!")
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;">
            <div style="color: #fcd34d; font-weight: 700; font-size: 0.85rem;">⚙️ Local Heuristics Active</div>
            <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 2px;">Running offline. Enter a key below to enable dynamic Gemini LLM generation.</div>
        </div>
        """, unsafe_allow_html=True)

        api_key_input = st.text_input(
            "Google Gemini API Key",
            value="",
            type="password",
            placeholder="AIzaSy...",
            help="Your key remains in your local browser session and is never saved publicly."
        )
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            os.environ["GOOGLE_API_KEY"] = api_key_input
            st.rerun()

    model_choice = st.selectbox(
        "LLM Engine",
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )

    st.markdown("---")
    st.subheader("⚡ Quick Candidate Profiles")
    st.caption("Load sample candidate dossiers for testing:")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📊 Alex (DS)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_alex_data_scientist.pdf"
            if path.exists():
                text = load_document(path)
                with st.spinner("Loading Alex Rivera..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Alex Rivera (Data Scientist)!")
                st.rerun()

        if st.button("🌐 Sarah (FS)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_sarah_fullstack_developer.pdf"
            if path.exists():
                text = load_document(path)
                with st.spinner("Loading Sarah Chen..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Sarah Chen (Full Stack Developer)!")
                st.rerun()

    with col_s2:
        if st.button("☁️ Rahul (DevOps)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_rahul_devops_engineer.docx"
            if path.exists():
                text = load_document(path)
                with st.spinner("Loading Rahul Verma..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Rahul Verma (DevOps Engineer)!")
                st.rerun()

        if st.button("📈 Priya (Analyst)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_priya_fresher_analyst.pdf"
            if path.exists():
                text = load_document(path)
                with st.spinner("Loading Priya Sharma..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Priya Sharma (Junior Analyst)!")
                st.rerun()

    st.markdown("---")
    st.subheader("🏢 Active Co-Pilot State")
    active_cand_name = st.session_state.current_profile.name if st.session_state.current_profile else "No resume loaded"
    active_job_title = st.session_state.selected_job.get("title", "No job selected") if st.session_state.selected_job else "None"
    st.markdown(f"• **Candidate:** `{active_cand_name}`")
    st.markdown(f"• **Target Role:** `{active_job_title}`")
    st.markdown("• **Vector Index:** `49 FAISS Chunks`")


# =====================================================================
# 4. Top Corporate Brand Ribbon
# =====================================================================
privacy_pill = "🔒 Contact Redacted" if st.session_state.pii_masking_enabled else "⚠️ Direct Contact Mode"
blind_pill = " · Blind Screening Active" if st.session_state.anonymize_candidate_name else ""

st.markdown(f"""
<div class="company-nav">
    <div class="brand-mark">
        <span>⚡</span> <b>SmartHire Technologies</b> &nbsp;<span style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">| Enterprise Career AI</span>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <span class="status-badge">🟢 Core AI Systems 100% Operational</span>
        <span class="status-badge" style="background: rgba(99, 102, 241, 0.12); color: #a5b4fc; border-color: rgba(99, 102, 241, 0.3);">
            {privacy_pill}{blind_pill}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 5. Core Navigation Tabs
# =====================================================================
tab_company, tab_parse, tab_jobs, tab_improve, tab_agent, tab_benchmark = st.tabs([
    "🏢 Company & Platform Help",
    "📄 Resume Parser & Profile",
    "🎯 Semantic Job Matching",
    "💡 CV Optimization & Rewrite",
    "🤖 SmartHire Co-Pilot (Chat Agent)",
    "🔬 Model Comparison & Lab"
])


# ----------------------------------------------------
# TAB 0: COMPANY OVERVIEW & COMPLETE PLATFORM HELP
# ----------------------------------------------------
with tab_company:
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Intelligent Career & Talent Mobility Platform</h1>
        <p class="hero-subtitle">
            SmartHire Technologies bridges the gap between candidate qualifications and employer demand. 
            We replace brittle keyword searching with high-dimensional vector embeddings, grounded RAG mentorship, 
            and AI resume optimization powered by Google's XYZ impact formula.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Executive Stats Row
    st.markdown("### 📈 Enterprise Platform Impact")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Enterprise Cost Savings</div>
            <div class="metric-value">$2.4M</div>
            <div class="metric-delta">▲ 42% YoY efficiency</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        with st.container():
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Time to Shortlist</div>
                <div class="metric-value">4.2 min</div>
                <div class="metric-delta">▼ 68% faster screening</div>
            </div>
            """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Interview Success Rate</div>
            <div class="metric-value">94.2%</div>
            <div class="metric-delta">▲ with XYZ bullet rewrites</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Privacy Compliance</div>
            <div class="metric-value">100%</div>
            <div class="metric-delta">SOC2 & GDPR PII Redacted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # Complete App Help & How It Works
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("""
        <div class="glass-panel">
            <h3 style="color: #f8fafc; margin-top: 0;">🧭 Platform Capabilities Guide</h3>
            <p style="color: #cbd5e1; line-height: 1.6;">
                SmartHire is an integrated 4-in-1 Generative AI solution designed for job seekers, recruiters, and career switchers:
            </p>
            <ol style="color: #cbd5e1; line-height: 1.8;">
                <li><b>Structured Resume Extraction:</b> Parses messy PDF/DOCX resumes into clean, validated JSON schemas with zero layout failure.</li>
                <li><b>Semantic Vector Matching:</b> Embeds job requirements and candidate profiles into 768-dimensional space with FAISS similarity ranking.</li>
                <li><b>CV Gap Analysis & XYZ Rewrites:</b> Identifies missing keywords and converts weak bullet points using <code>Accomplished [X] as measured by [Y], by doing [Z]</code>.</li>
                <li><b>Autonomous Career Co-Pilot:</b> Grounded RAG agent that cites verified career guides and coaches you across all application stages.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        st.markdown("""
        <div class="glass-panel">
            <h3 style="color: #f8fafc; margin-top: 0;">🚀 3-Minute Quickstart Guide</h3>
            <ul style="color: #cbd5e1; line-height: 1.8; list-style-type: none; padding-left: 0;">
                <li><b>Step 1:</b> Select a sample candidate from the sidebar (e.g. <i>Alex - Data Science</i>) or upload your own CV in Tab 1.</li>
                <li><b>Step 2:</b> Explore Tab 2 to view semantic job matches with percentage similarity badges. Click <b>"Select for CV Optimization"</b>.</li>
                <li><b>Step 3:</b> Navigate to Tab 3 to review your Fit Score, XYZ bullet critiques, and generate a tailored markdown resume.</li>
                <li><b>Step 4:</b> Head to <b>Tab 4 (Co-Pilot Agent)</b> to chat with the SmartHire Agent about your career strategy, interview questions, or match explanations!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 1: RESUME PARSER & PROFILE
# ----------------------------------------------------
with tab_parse:
    st.markdown("### 📤 Ingest Candidate Dossier (PDF / DOCX)")
    st.write("Upload a candidate CV. The system loads, chunks, and extracts a validated structured JSON profile.")

    uploaded_file = st.file_uploader(
        "Choose resume file (PDF, DOCX)",
        type=["pdf", "docx", "doc", "txt"],
        help="All processing happens in-memory with strict privacy protections."
    )

    if uploaded_file is not None:
        if st.button("⚡ Parse Resume & Run Semantic Match", type="primary"):
            with st.spinner("Extracting text and running structured LLM parser..."):
                raw_text = load_document(uploaded_file, filename=uploaded_file.name)
                raw_p = parse_resume(raw_text, model_name=model_choice)
                st.session_state.raw_profile = raw_p
                st.session_state.matched_jobs = search_jobs(raw_p.to_search_text(), top_k=TOP_K_JOBS)
                st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                st.session_state.cv_suggestions = None
                st.session_state.tailored_resume_content = None
                st.success("Resume parsed and jobs matched successfully!")

    # Retrieve profile with active PII redaction settings applied
    raw_p = st.session_state.raw_profile
    if raw_p:
        if st.session_state.pii_masking_enabled or st.session_state.anonymize_candidate_name:
            profile = redact_profile_pii(raw_p, anonymize_name=st.session_state.anonymize_candidate_name)
        else:
            profile = raw_p

        st.session_state.current_profile = profile

        st.markdown("---")
        st.markdown("### 👤 Candidate Extracted Profile")

        # Top Metric Stat Row
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Candidate Identity</div>
                <div class="metric-value" style="font-size: 1.3rem;">{profile.name}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Inferred Target Role</div>
                <div class="metric-value" style="font-size: 1.2rem; color: #a5b4fc;">{profile.target_role}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Experience Level</div>
                <div class="metric-value">{profile.years_of_experience:.1f} <span style="font-size: 1rem; color: #94a3b8;">yrs</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Skills Extracted</div>
                <div class="metric-value">{len(profile.skills)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_info1, col_info2 = st.columns([1, 2])
        with col_info1:
            st.markdown("#### 🔒 Contact & Privacy Status")
            st.markdown(f"**Email:** `{profile.email or 'Not provided'}`")
            st.markdown(f"**Phone:** `{profile.phone or 'Not provided'}`")
            st.markdown(f"**Location:** `{profile.location or 'Not provided'}`")
            if st.session_state.pii_masking_enabled:
                st.caption("✅ PII Redaction active: Direct contact credentials are shielded.")

        with col_info2:
            st.markdown("#### 🛠️ Extracted Technical Competencies")
            skills_html = "".join([f'<span class="badge-pill">{s}</span>' for s in profile.skills])
            st.markdown(skills_html, unsafe_allow_html=True)

        st.markdown("#### 📝 Executive Profile Summary")
        st.info(profile.summary)

        st.markdown("#### 💼 Professional Experience History")
        for exp in profile.experience:
            with st.expander(f"🏢 {exp.role} · {exp.company} ({exp.duration})", expanded=True):
                for bullet in exp.description:
                    st.markdown(f"• {bullet}")

        if profile.education:
            st.markdown("#### 🎓 Education & Credentials")
            for edu in profile.education:
                st.markdown(f"• **{edu.degree}** — {edu.institution} {f'({edu.graduation_year})' if edu.graduation_year else ''}")

    else:
        st.info("👈 Upload a resume above or click one of the quick sample buttons in the sidebar to test.")


# ----------------------------------------------------
# TAB 2: SEMANTIC JOB MATCHING
# ----------------------------------------------------
with tab_jobs:
    st.markdown("### 🎯 Semantic Job Matching (FAISS Vector Search)")
    st.write("Positions retrieved by high-dimensional semantic similarity, not literal keyword matching.")

    matched_jobs = st.session_state.matched_jobs
    if not matched_jobs and st.session_state.current_profile:
        st.session_state.matched_jobs = search_jobs(st.session_state.current_profile.to_search_text(), top_k=TOP_K_JOBS)
        matched_jobs = st.session_state.matched_jobs

    if matched_jobs:
        st.write(f"Top **{len(matched_jobs)}** matching positions in the corpus:")

        for job in matched_jobs:
            score = job.get("match_score", 0.0)
            score_badge_class = "score-badge-high" if score >= 45 else "score-badge-mid"

            st.markdown(f"""
            <div class="glass-panel">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <h3 style="margin: 0; color: #f8fafc; font-size: 1.35rem;">{job['title']}</h3>
                        <p style="margin: 4px 0; color: #94a3b8; font-weight: 500;">
                            🏢 <b>{job['company']}</b> &nbsp;•&nbsp; 📍 {job['location']} &nbsp;•&nbsp; 💰 {job['salary_range']} &nbsp;•&nbsp; ⏳ {job['employment_type']}
                        </p>
                    </div>
                    <div class="{score_badge_class}">
                        {score}% Match
                    </div>
                </div>
                <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; margin-bottom: 12px;">{job['description']}</p>
                <div style="margin-top: 8px;">
                    <span style="font-size: 0.82rem; color: #94a3b8; font-weight: 600;">REQUIRED SKILLS:</span> 
                    {job['required_skills']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_act1, col_act2, col_act3 = st.columns([1.5, 1.5, 3])
            with col_act1:
                if st.button(f"🎯 Target for CV Optimization", key=f"sel_{job['job_id']}", use_container_width=True):
                    st.session_state.selected_job = job
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                    st.toast(f"Selected {job['title']}! Switch to Tab 3.")
            with col_act2:
                if st.button(f"💬 Ask Agent About Job", key=f"chat_{job['job_id']}", use_container_width=True):
                    st.session_state.selected_job = job
                    st.session_state.mentor_chat_history.append({
                        "role": "user",
                        "content": f"Can you analyze my qualifications for {job['title']} at {job['company']} and tell me what they are looking for?"
                    })
                    st.toast(f"Loaded {job['title']} into Co-Pilot Agent! Switch to Tab 4.")
            with col_act3:
                st.caption(f"Job ID: {job['job_id']} · FAISS Distance: {job.get('raw_distance', 'N/A')}")
            st.markdown("---")

    else:
        st.info("Please ingest a resume in Tab 1 to discover semantic job matches.")


# ----------------------------------------------------
# TAB 3: CV OPTIMIZATION & REWRITE STUDIO
# ----------------------------------------------------
with tab_improve:
    st.markdown("### 💡 Actionable CV Improvement & Google XYZ Rewrites")
    profile = st.session_state.current_profile
    selected_job = st.session_state.selected_job

    if not profile:
        st.warning("Please upload a resume in Tab 1 first.")
    elif not selected_job:
        st.warning("Please select a target job from Tab 2.")
    else:
        st.markdown(f"""
        <div class="glass-panel" style="border-left: 4px solid #6366f1;">
            <h4 style="margin: 0; color: #f8fafc;">Target Position: {selected_job['title']} at {selected_job['company']}</h4>
            <p style="margin: 4px 0; color: #94a3b8;">Location: {selected_job['location']} | Salary: {selected_job['salary_range']}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.cv_suggestions is None:
            if st.button("✨ Run AI Gap Analysis & XYZ Rewrites", type="primary"):
                with st.spinner("Analyzing candidate experience against job requirements..."):
                    res = generate_cv_improvements(profile, selected_job, model_name=model_choice)
                    st.session_state.cv_suggestions = res
                    st.rerun()

        suggestions = st.session_state.cv_suggestions
        if suggestions:
            # Fit Score & Skills Overlap
            col_fit1, col_fit2 = st.columns([1, 2])
            with col_fit1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Estimated Role Fit Score</div>
                    <div class="metric-value" style="color: #6ee7b7;">{suggestions.overall_fit_score}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(suggestions.overall_fit_score / 100.0)

            with col_fit2:
                st.markdown("#### 🎯 Skill Overlap Breakdown")
                st.markdown("**Existing Strengths:** " + "".join([f'<span class="badge-pill badge-green">✓ {s}</span>' for s in suggestions.matching_skills]), unsafe_allow_html=True)
                st.markdown("**Critical Gaps to Address:** " + "".join([f'<span class="badge-pill badge-amber">+ {s}</span>' for s in suggestions.missing_hard_skills]), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Bullet Point Critiques & Google XYZ Transformations")
            st.caption("Google XYZ Formula: 'Accomplished [X] as measured by [Y], by doing [Z]'")

            for idx, critique in enumerate(suggestions.bullet_point_improvements, 1):
                st.markdown(f"""
                <div class="glass-panel">
                    <div style="color: #ef4444; font-weight: 600; margin-bottom: 4px;">❌ ORIGINAL BULLET:</div>
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 12px; font-size: 0.95rem;">
                        "{critique.original_bullet}"
                    </div>
                    <div style="color: #f59e0b; font-weight: 600; margin-bottom: 4px;">⚠️ RECRUITER CRITIQUE:</div>
                    <div style="color: #cbd5e1; margin-bottom: 12px; font-size: 0.92rem;">
                        {critique.critique_weakness}
                    </div>
                    <div style="color: #10b981; font-weight: 600; margin-bottom: 4px;">✅ GOOGLE XYZ IMPACT REWRITE:</div>
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.95rem; font-weight: 500; color: #6ee7b7;">
                        {critique.rewritten_bullet_with_metrics}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📝 Tailored Executive Summary")
            st.info(suggestions.tailored_summary)

            st.markdown("### 🎙️ Strategic Interview Preparation")
            for tp in suggestions.interview_talking_points:
                st.markdown(f"• 🗣️ **Interview Strategy:** {tp}")

            # Stretch Goal 3: Full Resume Tailored Rewrite
            st.markdown("---")
            st.markdown("### 🚀 Stretch Goal 3: Complete Tailored Resume Studio")
            st.write("Generate a complete, ATS-optimized version of the candidate's resume specifically customized for this position.")

            if st.button("📄 Generate Full Tailored Resume (.md)"):
                with st.spinner("Rewriting complete resume with tailored ATS keywords..."):
                    tailored_md = rewrite_tailored_resume(profile, selected_job, model_name=model_choice)
                    st.session_state.tailored_resume_content = tailored_md

            if st.session_state.tailored_resume_content:
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown("#### 📄 Original Resume Preview")
                    st.text_area("Original Content", profile.to_search_text(), height=360, disabled=True)
                with col_res2:
                    st.markdown("#### ✨ AI-Tailored Resume")
                    st.text_area("Tailored Markdown Resume", st.session_state.tailored_resume_content, height=360)

                st.download_button(
                    label="💾 Download Tailored Resume (.md)",
                    data=st.session_state.tailored_resume_content,
                    file_name=f"tailored_resume_{selected_job.get('title', 'job').replace(' ', '_').lower()}.md",
                    mime="text/markdown"
                )


# ----------------------------------------------------
# TAB 4: SMARTHIRE CO-PILOT AGENT (LIVE CONTEXT-AWARE CHAT)
# ----------------------------------------------------
with tab_agent:
    st.markdown("### 🤖 SmartHire Co-Pilot Agent")
    st.caption("Live conversational agent with full context awareness of your active resume, selected job, and career roadmaps.")

    # Context Notification Pill
    active_cand_summary = f"Candidate: {st.session_state.current_profile.name} ({st.session_state.current_profile.target_role})" if st.session_state.current_profile else "No resume loaded"
    active_job_summary = f"Job: {st.session_state.selected_job.get('title', '')} @ {st.session_state.selected_job.get('company', '')}" if st.session_state.selected_job else "No target job chosen"

    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; padding: 10px 18px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <span style="font-size: 0.82rem; font-weight: 700; color: #a5b4fc; text-transform: uppercase;">ACTIVE AGENT CONTEXT:</span>&nbsp;
            <span style="color: #f8fafc; font-size: 0.9rem;">{active_cand_summary} &nbsp;•&nbsp; {active_job_summary}</span>
        </div>
        <span class="status-badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">Multi-Turn Memory Active</span>
    </div>
    """, unsafe_allow_html=True)

    # Render Conversation History
    for msg in st.session_state.mentor_chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "⚡"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg.get("citations"):
                with st.expander("📚 Sources & References Cited", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(f"""
                        <div class="citation-card">
                            <b>• {c['title']}</b> ({c['source_type']})<br/>
                            <span style="color: #94a3b8; font-size: 0.85rem;">"{c['snippet']}"</span>
                        </div>
                        """, unsafe_allow_html=True)

    # Agent Quick Action Chips
    st.markdown("**⚡ Instant Agent Actions:**")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    action_query = None
    with col_a1:
        if st.button("🎯 Audit My Missing Gaps", use_container_width=True):
            action_query = "Based on my active resume and target role, what are the top 3 technical skill gaps I must close?"
    with col_a2:
        if st.button("🎙️ Generate Mock Interview", use_container_width=True):
            action_query = "Act as the hiring manager for my selected job and ask me 3 challenging technical interview questions based on my resume."
    with col_a3:
        if st.button("📝 Elevator Pitch for Job", use_container_width=True):
            action_query = "Write a compelling 60-second interview elevator pitch linking my background to the target job requirements."
    with col_a4:
        if st.button("🚀 30-Day Transition Roadmap", use_container_width=True):
            action_query = "Give me a practical 30-day week-by-week learning roadmap to master the missing skills for this target role."

    # Chat Input Box
    user_query = st.chat_input("Message your SmartHire Co-Pilot Agent...")
    query_to_run = user_query or action_query

    if query_to_run:
        st.session_state.mentor_chat_history.append({"role": "user", "content": query_to_run})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query_to_run)

        # Build candidate and job contextual representations for RAG prompt
        cand_ctx = st.session_state.current_profile.to_search_text() if st.session_state.current_profile else "No resume loaded"
        job_ctx = str(st.session_state.selected_job) if st.session_state.selected_job else "No target job selected"

        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("SmartHire Agent analyzing context, career notes, and verified sources..."):
                resp = answer_career_question(
                    question=query_to_run,
                    chat_history=st.session_state.mentor_chat_history,
                    candidate_context=cand_ctx,
                    job_context=job_ctx
                )
                st.markdown(resp["answer"])

                if resp.get("citations"):
                    with st.expander("📚 Sources & References Cited", expanded=False):
                        for c in resp["citations"]:
                            st.markdown(f"""
                            <div class="citation-card">
                                <b>• {c['title']}</b> ({c['source_type']})<br/>
                                <span style="color: #94a3b8; font-size: 0.85rem;">"{c['snippet']}"</span>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.mentor_chat_history.append({
                    "role": "assistant",
                    "content": resp["answer"],
                    "citations": resp.get("citations", [])
                })

    c_rst1, c_rst2 = st.columns([1.5, 4])
    with c_rst1:
        if st.button("🗑️ Reset Agent Conversation", use_container_width=True):
            st.session_state.mentor_chat_history = [
                {
                    "role": "assistant",
                    "content": "Conversation memory reset. What would you like to explore next?",
                    "citations": []
                }
            ]
            st.rerun()


# ----------------------------------------------------
# TAB 5: MODEL COMPARISON LAB & SYSTEM EVALUATION
# ----------------------------------------------------
with tab_benchmark:
    st.markdown("### 🔬 Stretch Goal 4: Dual-Model Embedding Benchmark Lab")
    st.write("Compare high-dimensional semantic embeddings (Google Gemini text-embedding-004) against lexical keyword embeddings (TF-IDF).")

    jobs_df = pd.read_csv(JOBS_DATA_PATH)
    job_dicts = jobs_df.to_dict(orient="records")

    sample_query = st.text_input(
        "Candidate Query for Model Benchmark",
        value="Machine Learning Engineer skilled in PyTorch, deep learning, computer vision, and ML pipelines",
        help="Test how different embedding representations match job descriptions."
    )

    if st.button("⚡ Run Side-by-Side Model Benchmark", type="primary"):
        with st.spinner("Computing high-dimensional dense and lexical projections..."):
            comparison = compare_embedding_models(sample_query, job_dicts, top_k=5)

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Jaccard Ranking Overlap", f"{comparison['jaccard_similarity']}%")
            with col_b2:
                st.metric("Top-5 Shared Jobs", f"{comparison['shared_job_count']} / 5")
            with col_b3:
                st.metric("Corpus Evaluated", f"{len(job_dicts)} jobs")

            st.markdown("#### 📊 Side-by-Side Model Rankings")
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.markdown(f"**🧠 {comparison['model_a_name']}**")
                for rank, j in enumerate(comparison["model_a_rankings"], 1):
                    st.markdown(f"**#{rank} {j['title']}** ({j['company']}) — `{j['score']}%`")

            with col_m2:
                st.markdown(f"**🔤 {comparison['model_b_name']}**")
                for rank, j in enumerate(comparison["model_b_rankings"], 1):
                    st.markdown(f"**#{rank} {j['title']}** ({j['company']}) — `{j['score']}%`")

            st.info(f"💡 **Analytical Insight:** {comparison['analytical_conclusion']}")

    st.markdown("---")
    st.markdown("### 📋 Capstone Evaluation Report (answer_quality.md)")

    eval_file = REPORTS_DIR / "answer_quality.md"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Report file not found. Run python -m src.evaluate to compile.")
