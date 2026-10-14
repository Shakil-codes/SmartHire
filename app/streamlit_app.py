"""
SmartHire GenAI — Resume Matching & AI Career Mentor
Premium High-Aesthetic Web Portal Application
Includes:
- PII & Sensitive Data Redaction / Blind Screening Mode
- All Stretch Goals:
  1. Multi-turn Session Memory
  2. Grounded Document & Job Citations
  3. Side-by-Side Tailored Resume Rewriter (.md download)
  4. Dual-Model Embedding Benchmark (Gemini text-embedding-004 vs. TF-IDF)
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
# 1. Page Configuration & Ultra-Premium Styling
# =====================================================================
st.set_page_config(
    page_title="SmartHire GenAI · Next-Gen Career Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# World-Class Dark Glassmorphism CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #090d16;
        --card-bg: rgba(17, 24, 39, 0.7);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-primary: #6366f1;
        --accent-violet: #8b5cf6;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-primary);
    }

    /* Ambient Header with Glowing Radial Accent */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.25) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.2) 0%, transparent 40%),
                    linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #090d16 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(120deg, #ffffff 30%, #a5b4fc 70%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-top: 8px;
        margin-bottom: 12px;
        max-width: 820px;
        line-height: 1.6;
    }

    /* Privacy & Security Ribbon */
    .privacy-ribbon {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-top: 4px;
    }

    /* Glassmorphic Cards */
    .glass-panel {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-panel:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 16px 32px -8px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }

    /* Stat Metric Boxes */
    .stat-box {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: left;
    }
    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .stat-val {
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 4px;
        color: #f8fafc;
    }

    /* Badges & Tags */
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
    .badge-purple {
        background: rgba(168, 85, 247, 0.14);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.28);
    }

    /* Match Score Badges */
    .score-badge-high {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        color: #6ee7b7;
        font-size: 1.2rem;
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
        font-size: 1.2rem;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 12px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }

    /* Citation Expandable Box */
    .citation-card {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #8b5cf6;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    /* Code & Diffs */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
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
                "👋 Hello! I am your **SmartHire AI Career Mentor**.\n\n"
                "I am grounded in verified tech career roadmaps, job descriptions, and interview standards. "
                "I maintain memory of our session so you can ask multi-turn questions like:\n"
                "- *'What are the 3 phases to transition into Data Science?'*\n"
                "- *'Explain the Google XYZ formula for my resume bullets with an example.'*\n"
                "- *'Can you recommend a certification path for DevOps and Cloud?'*\n\n"
                "How can I help accelerate your career today?"
            ),
            "citations": []
        }
    ]


# =====================================================================
# 3. Sidebar: Configuration, Security & Sample Loaders
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=56)
    st.title("SmartHire GenAI")
    st.caption("v2.0 · Privacy-First Career Intelligence")

    st.markdown("---")
    st.subheader("🛡️ Privacy & PII Protection")
    st.caption("Prevent sensitive personal details from being displayed on screen:")

    mask_pii = st.toggle(
        "Mask Contact Info (Email & Phone)",
        value=st.session_state.pii_masking_enabled,
        help="Redacts email, phone numbers, and physical addresses to protect candidate privacy."
    )
    st.session_state.pii_masking_enabled = mask_pii

    anon_name = st.checkbox(
        "Blind Screening (Anonymize Name)",
        value=st.session_state.anonymize_candidate_name,
        help="Replaces candidate name with anonymous token (e.g. Candidate #4C2A) to eliminate hiring bias."
    )
    st.session_state.anonymize_candidate_name = anon_name

    st.markdown("---")
    st.subheader("🔑 LLM Credentials")
    
    # Masked API Key Input - Never show sensitive key in plaintext
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=GEMINI_API_KEY if GEMINI_API_KEY else "",
        type="password",
        help="Your key is kept securely in memory and never shared or persisted to remote servers."
    )
    if api_key_input and api_key_input != os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input

    if check_api_key():
        st.markdown('<span class="badge-pill badge-green">🔒 API Key Configured</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-pill badge-amber">⚙️ Offline Heuristics Mode</span>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Active Model",
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )

    st.markdown("---")
    st.subheader("⚡ Quick Sample Resumes")
    st.caption("Load verified test profiles to explore all features instantly:")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📊 Alex (DS)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_alex_data_scientist.pdf"
            if path.exists():
                text = load_document(path)
                with st.spinner("Analyzing Alex Rivera..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Alex Rivera!")
                st.rerun()

        if st.button("🌐 Sarah (FS)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_sarah_fullstack_developer.pdf"
            if path.exists():
                text = load_document(path)
                with st.spinner("Analyzing Sarah Chen..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Sarah Chen!")
                st.rerun()

    with col_s2:
        if st.button("☁️ Rahul (DevOps)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_rahul_devops_engineer.docx"
            if path.exists():
                text = load_document(path)
                with st.spinner("Analyzing Rahul Verma..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Rahul Verma!")
                st.rerun()

        if st.button("📈 Priya (Analyst)", use_container_width=True):
            path = PROJECT_ROOT / "data/resumes/resume_priya_fresher_analyst.pdf"
            if path.exists():
                text = load_document(path)
                with st.spinner("Analyzing Priya Sharma..."):
                    raw = parse_resume(text, model_name=model_choice)
                    st.session_state.raw_profile = raw
                    st.session_state.matched_jobs = search_jobs(raw.to_search_text(), top_k=TOP_K_JOBS)
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                st.toast("Loaded Priya Sharma!")
                st.rerun()

    st.markdown("---")
    st.subheader("🛡️ Enterprise Guardrails")
    st.markdown("• **Jailbreak Defense:** Active")
    st.markdown("• **Off-Topic Filter:** Active")
    st.markdown("• **Grounding Enforcement:** Strict")
    st.markdown("• **FAISS Vector Store:** 49 Chunks Ready")


# =====================================================================
# 4. Main Hero Banner with Privacy Shield Indicators
# =====================================================================
privacy_status = "🔒 PII Redaction: Active (Contact Masked)" if st.session_state.pii_masking_enabled else "⚠️ PII Redaction: Disabled (Live Contact Visible)"
blind_status = " · Blind Screening Active" if st.session_state.anonymize_candidate_name else ""

st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">SmartHire GenAI Portal</h1>
            <p class="hero-subtitle">
                Enterprise Semantic Resume Matching, Google XYZ Impact Optimization & Grounded AI Career Mentorship
            </p>
            <div class="privacy-ribbon">
                {privacy_status}{blind_status}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 5. Core Navigation Tabs
# =====================================================================
tab_parse, tab_jobs, tab_improve, tab_mentor, tab_benchmark = st.tabs([
    "📄 Resume Parser & Profile",
    "🎯 Semantic Job Matching",
    "💡 CV Optimization & Tailored Rewrite",
    "🤖 AI Career Mentor (RAG)",
    "🔬 Model Comparison & Evaluation"
])


# ----------------------------------------------------
# TAB 1: RESUME PARSER & STRUCTURED PROFILE
# ----------------------------------------------------
with tab_parse:
    st.markdown("### 📤 Ingest Candidate Resume (PDF / DOCX)")
    st.write("Upload a candidate CV. The system loads, chunks, and extracts a structured JSON profile using schema validation.")

    uploaded_file = st.file_uploader(
        "Choose resume file (PDF, DOCX)",
        type=["pdf", "docx", "doc", "txt"],
        help="All processing happens in-memory with strict privacy protections."
    )

    if uploaded_file is not None:
        if st.button("⚡ Parse Resume & Match Jobs", type="primary"):
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
        # Apply PII redaction according to user privacy preference
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
            <div class="stat-box">
                <div class="stat-label">Candidate Name</div>
                <div class="stat-val" style="font-size: 1.3rem;">{profile.name}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Inferred Target Role</div>
                <div class="stat-val" style="font-size: 1.2rem; color: #a5b4fc;">{profile.target_role}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Experience</div>
                <div class="stat-val">{profile.years_of_experience:.1f} <span style="font-size: 1rem; color: #94a3b8;">yrs</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c_m4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Skills Extracted</div>
                <div class="stat-val">{len(profile.skills)}</div>
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
                st.caption("✅ PII Masking active: Real email and phone are shielded.")

        with col_info2:
            st.markdown("#### 🛠️ Technical Competency Cloud")
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
# TAB 2: SEMANTIC JOB MATCHING (FAISS)
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

            col_act1, col_act2 = st.columns([1.5, 4])
            with col_act1:
                if st.button(f"🎯 Select for CV Optimization", key=f"sel_{job['job_id']}", use_container_width=True):
                    st.session_state.selected_job = job
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                    st.toast(f"Selected {job['title']}! Switch to Tab 3.")
            with col_act2:
                st.caption(f"Job ID: {job['job_id']} · FAISS Distance: {job.get('raw_distance', 'N/A')}")
            st.markdown("---")

    else:
        st.info("Please ingest a resume in Tab 1 to discover semantic job matches.")


# ----------------------------------------------------
# TAB 3: CV OPTIMIZATION & TAILORED REWRITER (STRETCH GOAL 3)
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
                <div class="stat-box">
                    <div class="stat-label">Estimated Fit Score</div>
                    <div class="stat-val" style="color: #6ee7b7;">{suggestions.overall_fit_score}%</div>
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

            # =========================================================
            # STRETCH GOAL 3: FULL TAILORED RESUME REWRITE & DOWNLOAD
            # =========================================================
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
# TAB 4: AI CAREER MENTOR (RAG) WITH MEMORY & CITATIONS
# ----------------------------------------------------
with tab_mentor:
    st.markdown("### 🤖 AI Career Mentor (RAG Pipeline)")
    st.caption("Grounded in verified career roadmaps and live job postings with conversational memory & document citations.")

    # Render Chat History (Stretch Goal 1: Multi-Turn Memory)
    for msg in st.session_state.mentor_chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Stretch Goal 2: Grounded Citations
            if msg.get("citations"):
                with st.expander("📚 Sources & References Cited", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(f"""
                        <div class="citation-card">
                            <b>• {c['title']}</b> ({c['source_type']})<br/>
                            <span style="color: #94a3b8; font-size: 0.85rem;">"{c['snippet']}"</span>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Suggested Career Questions:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    suggested_q = None
    with col_q1:
        if st.button("🚀 Data Science 3-Phase Roadmap"):
            suggested_q = "What is the recommended 3-phase roadmap to transition into Data Science?"
    with col_q2:
        if st.button("🎯 Google XYZ Resume Formula"):
            suggested_q = "What is the Google XYZ formula for writing strong resume bullet points?"
    with col_q3:
        if st.button("☁️ DevOps & Cloud Tools"):
            suggested_q = "What core tools should a DevOps engineer master for infrastructure as code and containers?"

    # Chat Input Box
    user_query = st.chat_input("Ask your Career Mentor a question...")
    query_to_run = user_query or suggested_q

    if query_to_run:
        # Add user query to conversation history
        st.session_state.mentor_chat_history.append({"role": "user", "content": query_to_run})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query_to_run)

        # Process with RAG & Guardrails
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Consulting career guides and applying safety guardrails..."):
                resp = answer_career_question(
                    query_to_run,
                    chat_history=st.session_state.mentor_chat_history
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

    c_rst1, c_rst2 = st.columns([1, 4])
    with c_rst1:
        if st.button("🗑️ Reset Chat History"):
            st.session_state.mentor_chat_history = [
                {
                    "role": "assistant",
                    "content": "Chat memory reset. What career topic would you like to explore next?",
                    "citations": []
                }
            ]
            st.rerun()


# ----------------------------------------------------
# TAB 5: MODEL COMPARISON & SYSTEM EVALUATION (STRETCH GOAL 4)
# ----------------------------------------------------
with tab_benchmark:
    st.markdown("### 🔬 Stretch Goal 4: Dual-Model Embedding Benchmark & Evaluation")
    st.write("Compare high-dimensional semantic embeddings (Google Gemini text-embedding-004) against lexical keyword embeddings (TF-IDF).")

    # Benchmark Test Runner
    jobs_df = pd.read_csv(JOBS_DATA_PATH)
    job_dicts = jobs_df.to_dict(orient="records")

    sample_query = st.text_input(
        "Candidate Search Query for Embedding Benchmark",
        value="Machine Learning Engineer skilled in PyTorch, deep learning, computer vision, and ML pipelines",
        help="Test how different embedding representations match job descriptions."
    )

    if st.button("⚡ Run Side-by-Side Embedding Benchmark", type="primary"):
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
