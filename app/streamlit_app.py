"""
SmartHire Technologies — AI-Powered Career Intelligence Platform
Reference benchmark workflow:
- ⊞ Overview: 5-step workflow progress track, career workflow cards, instant sample loaders
- 👤 Profile: Document ingestion (PDF/DOCX), structured profile verification, PII redaction
- 💼 Jobs: Semantic FAISS vector job matching, Kaggle corpus, match scores, gating guards
- 📄 Resume Studio: Target role alignment, Google XYZ impact formula rewrites, tailored resume export
- 💬 Career Mentor: Conversational AI coaching with multi-turn memory & grounded citations
- 🔬 Model Lab: Stretch Goal 4 dual-model embedding benchmarking & capstone evaluation
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
# 1. Page Configuration & Aesthetic SaaS Styling
# =====================================================================
st.set_page_config(
    page_title="SmartHire · Career Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic SaaS CSS matching Reference App
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --primary-blue: #1d4ed8;
        --primary-blue-hover: #1e40af;
        --brand-accent: #2563eb;
        --bg-main: #f8fafc;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }

    /* Streamlit Main Container styling */
    .stApp {
        background-color: #f8fafc;
    }

    /* Sidebar Brand Styling */
    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #2563eb;
        text-transform: uppercase;
        margin-top: 4px;
        margin-bottom: 18px;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #64748b;
        text-transform: uppercase;
        margin-top: 14px;
        margin-bottom: 8px;
    }

    /* Clean Radio Group Styling for Sidebar Navigation */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 8px 14px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        border: 1px solid transparent !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
    section[data-testid="stSidebar"] div[role="radiogroup"] label input,
    section[data-testid="stSidebar"] div[role="radiogroup"] label svg {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-color: #1d4ed8 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div {
        color: #ffffff !important;
    }

    /* Primary button matching reference app dark slate */
    button[data-testid="baseButton-primary"] {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #0f172a !important;
        border-color: #0f172a !important;
    }

    /* Workflow Status Box in Sidebar */
    .sidebar-status-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        margin-top: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .status-capsule {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 10px;
    }
    .status-capsule-pending {
        background: #f1f5f9;
        color: #64748b;
        border: 1px solid #e2e8f0;
    }
    .status-capsule-active {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }

    /* Workflow Progress Step Cards */
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        position: relative;
        min-height: 110px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .step-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .step-card-active {
        border: 1.5px solid #2563eb;
        background: #ffffff;
        box-shadow: 0 4px 12px -2px rgba(37,99,235,0.12);
    }
    .step-num {
        font-size: 1.05rem;
        font-weight: 800;
        color: #1e293b;
    }
    .step-num-active {
        color: #2563eb;
    }
    .step-badge {
        position: absolute;
        top: 14px;
        right: 14px;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .step-badge-current {
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
    }
    .step-badge-locked {
        background: #f8fafc;
        color: #94a3b8;
        border: 1px solid #e2e8f0;
    }
    .step-badge-done {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .step-badge-ready {
        background: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 8px;
        margin-bottom: 2px;
    }
    .step-sub {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* 3-Column Career Workflow Action Cards */
    .action-workflow-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 210px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .action-workflow-card:hover {
        border-color: #cbd5e1;
        transform: translateY(-2px);
    }
    .card-tracker {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .card-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 8px 0;
    }
    .card-body-text {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.55;
        margin-bottom: 18px;
    }

    /* Gating Action Needed Card */
    .gating-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 26px;
        margin-top: 14px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .badge-action-needed {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #d97706;
        background: #fffbeb;
        border: 1px solid #fde68a;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    /* Clean Job Cards */
    .job-listing-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: border-color 0.15s ease;
    }
    .job-listing-card:hover {
        border-color: #93c5fd;
    }
    .match-pill-high {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-size: 0.88rem;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .match-pill-mid {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        font-size: 0.88rem;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 9999px;
    }

    /* Competency Tags */
    .skill-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #dbeafe;
    }
    .skill-tag-match {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .skill-tag-missing {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    /* Google XYZ Box */
    .xyz-critique-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    /* Citation Box in Chat */
    .mentor-citation-box {
        background: #f8fafc;
        border-left: 3px solid #2563eb;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        margin-top: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }

    /* Buttons override */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.15s ease;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. Session State Initialization
# =====================================================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Overview"

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
                "I am fully synchronized with your active resume and selected target job. "
                "I maintain memory of our session and ground my guidance in verified career roadmaps and real market requirements.\n\n"
                "**Ask me anything:**\n"
                "- *'Explain my match score for my active job'* \n"
                "- *'What are the key skill gaps I need to fix?'*\n"
                "- *'Coach me for the interview with this company'* \n"
                "- *'How do I transition to Generative AI Engineer?'*"
            ),
            "citations": []
        }
    ]


# Helper to navigate between pages
def navigate_to(page_name: str):
    st.session_state.active_tab = page_name
    st.rerun()


# =====================================================================
# 3. Sidebar: Brand, Navigation & Workflow Status
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div>
        <div class="sidebar-brand-title">SMART HIRE</div>
        <div class="sidebar-brand-sub">CAREER INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">WORKSPACE</div>', unsafe_allow_html=True)

    nav_options = [
        "⊞ Overview",
        "👤 Profile",
        "💼 Jobs",
        "📄 Resume Studio",
        "💬 Career Mentor",
        "🔬 Model Lab"
    ]
    nav_keys = {
        "⊞ Overview": "Overview",
        "👤 Profile": "Profile",
        "💼 Jobs": "Jobs",
        "📄 Resume Studio": "Resume Studio",
        "💬 Career Mentor": "Career Mentor",
        "🔬 Model Lab": "Model Lab"
    }
    rev_keys = {v: k for k, v in nav_keys.items()}

    current_idx = 0
    if st.session_state.active_tab in rev_keys:
        current_idx = nav_options.index(rev_keys[st.session_state.active_tab])

    selected_nav = st.radio(
        label="Workspace Navigation",
        options=nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )
    new_tab = nav_keys[selected_nav]
    if new_tab != st.session_state.active_tab:
        st.session_state.active_tab = new_tab
        st.rerun()

    # Workflow Status Widget
    st.markdown('<div class="sidebar-section-title">WORKFLOW STATUS</div>', unsafe_allow_html=True)

    has_resume = st.session_state.raw_profile is not None
    resume_pill_html = (
        f'<span class="status-capsule status-capsule-active">● Active: {st.session_state.current_profile.name if st.session_state.current_profile else "Uploaded"}</span>'
        if has_resume else
        '<span class="status-capsule status-capsule-pending">○ No resume</span>'
    )
    target_role_text = (
        st.session_state.selected_job.get("title", "Not specified")
        if st.session_state.selected_job else
        (st.session_state.current_profile.target_role if (st.session_state.current_profile and st.session_state.current_profile.target_role) else "Not specified")
    )
    match_count_text = f"{len(st.session_state.matched_jobs)} roles" if st.session_state.matched_jobs else "0 roles"

    st.markdown(f"""
    <div class="sidebar-status-box">
        {resume_pill_html}
        <div style="font-size: 0.85rem; color: #0f172a; margin-top: 4px;">
            <b>Target:</b> <span style="color: #475569;">{target_role_text}</span>
        </div>
        <div style="font-size: 0.85rem; color: #0f172a; margin-top: 4px;">
            <b>Matches:</b> <span style="color: #475569;">{match_count_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if has_resume:
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        if st.button("✕ Clear Active Resume", use_container_width=True, help="Clear candidate dossier"):
            st.session_state.raw_profile = None
            st.session_state.current_profile = None
            st.session_state.matched_jobs = []
            st.session_state.selected_job = None
            st.session_state.cv_suggestions = None
            st.session_state.tailored_resume_content = None
            st.toast("Active resume cleared!")
            st.rerun()

    # Privacy & Controls Expander
    with st.expander("🛡️ Privacy & Settings", expanded=False):
        mask_pii = st.toggle(
            "Mask Contact PII",
            value=st.session_state.pii_masking_enabled,
            help="Redacts emails, phone numbers, and street addresses."
        )
        st.session_state.pii_masking_enabled = mask_pii

        anon_name = st.checkbox(
            "Blind Screening (Mask Name)",
            value=st.session_state.anonymize_candidate_name,
            help="Replaces full name with anonymous ID to remove unconscious bias."
        )
        st.session_state.anonymize_candidate_name = anon_name

        model_choice = st.selectbox(
            "LLM Engine",
            ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0
        )

        is_key_active = check_api_key()
        if is_key_active:
            st.success("🔒 Server API Key Active")
        else:
            api_key_input = st.text_input("Gemini API Key", type="password")
            if api_key_input:
                os.environ["GEMINI_API_KEY"] = api_key_input
                os.environ["GOOGLE_API_KEY"] = api_key_input
                st.rerun()


# Update current profile with active PII settings
if st.session_state.raw_profile:
    if st.session_state.pii_masking_enabled or st.session_state.anonymize_candidate_name:
        st.session_state.current_profile = redact_profile_pii(
            st.session_state.raw_profile,
            anonymize_name=st.session_state.anonymize_candidate_name
        )
    else:
        st.session_state.current_profile = st.session_state.raw_profile


# =====================================================================
# 4. VIEW: ⊞ OVERVIEW
# =====================================================================
if st.session_state.active_tab == "Overview":
    has_profile = st.session_state.raw_profile is not None

    # Step header
    step_num = "2" if has_profile else "1"
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <span style="font-size: 0.8rem; font-weight: 700; color: #64748b; background: #ffffff; border: 1px solid #e2e8f0; padding: 4px 12px; border-radius: 9999px;">
            ↻ Step {step_num} of 5
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_h_left, col_h_right = st.columns([3.5, 1.2])
    with col_h_left:
        if not has_profile:
            st.markdown("""
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-top: 0; margin-bottom: 8px; letter-spacing: -0.02em;">
                Add your resume to begin
            </h1>
            <p style="font-size: 1.05rem; color: #475569; line-height: 1.5; margin: 0;">
                Upload your resume document to extract verified competencies, evaluate matching roles, and tailor applications.
            </p>
            """, unsafe_allow_html=True)
        else:
            cand_name = st.session_state.current_profile.name if st.session_state.current_profile else "Candidate"
            st.markdown(f"""
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-top: 0; margin-bottom: 8px; letter-spacing: -0.02em;">
                Profile verified: {cand_name}
            </h1>
            <p style="font-size: 1.05rem; color: #475569; line-height: 1.5; margin: 0;">
                Competencies extracted and {len(st.session_state.matched_jobs)} semantic job matches ready for evaluation and career coaching.
            </p>
            """, unsafe_allow_html=True)

    with col_h_right:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if not has_profile:
            if st.button("Add your resume →", type="primary", use_container_width=True):
                navigate_to("Profile")
        else:
            if st.button("Explore Matching Jobs →", type="primary", use_container_width=True):
                navigate_to("Jobs")

    st.markdown("<br>", unsafe_allow_html=True)

    # Workflow progress track
    st.markdown('<h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 14px;">Workflow progress</h3>', unsafe_allow_html=True)

    col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)

    with col_w1:
        c1_class = "step-card step-card-active" if not has_profile else "step-card"
        c1_badge = '<span class="step-badge step-badge-current">Current</span>' if not has_profile else '<span class="step-badge step-badge-done">Done</span>'
        c1_sub = "Pending upload" if not has_profile else "Uploaded & Verified"
        st.markdown(f"""
        <div class="{c1_class}">
            <div class="step-num step-num-active">01</div>
            {c1_badge}
            <div class="step-title">Resume</div>
            <div class="step-sub">{c1_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_w2:
        c2_badge = '<span class="step-badge step-badge-locked">Locked</span>' if not has_profile else '<span class="step-badge step-badge-ready">Ready</span>'
        c2_sub = "Locked" if not has_profile else "Skills Extracted"
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">02</div>
            {c2_badge}
            <div class="step-title">Profile</div>
            <div class="step-sub">{c2_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_w3:
        c3_badge = '<span class="step-badge step-badge-locked">Locked</span>' if not has_profile else '<span class="step-badge step-badge-ready">Ready</span>'
        c3_sub = "Locked" if not has_profile else f"{len(st.session_state.matched_jobs)} Roles Matched"
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">03</div>
            {c3_badge}
            <div class="step-title">Jobs</div>
            <div class="step-sub">{c3_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_w4:
        c4_badge = '<span class="step-badge step-badge-locked">Locked</span>' if not has_profile else '<span class="step-badge step-badge-ready">Ready</span>'
        c4_sub = "Locked" if not has_profile else "XYZ Optimization"
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">04</div>
            {c4_badge}
            <div class="step-title">Resume Studio</div>
            <div class="step-sub">{c4_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_w5:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">05</div>
            <span class="step-badge step-badge-ready">Available</span>
            <div class="step-title">Career Mentor</div>
            <div class="step-sub">Coaching Ready</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Career workflow 3 cards
    st.markdown('<h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 14px;">Career workflow</h3>', unsafe_allow_html=True)

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.markdown("""
        <div class="action-workflow-card">
            <div>
                <div class="card-tracker">01 · PROFILE</div>
                <div class="card-heading">Review profile</div>
                <div class="card-body-text">
                    Ensure your skills, target role, and summary accurately represent your background before AI matching.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open profile →", key="btn_open_profile", use_container_width=True):
            navigate_to("Profile")

    with col_a2:
        st.markdown("""
        <div class="action-workflow-card">
            <div>
                <div class="card-tracker">02 · MATCHING</div>
                <div class="card-heading">Explore roles</div>
                <div class="card-body-text">
                    Discover positions matching your verified competencies with clear explanations of why they match.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore jobs →", key="btn_explore_jobs", use_container_width=True):
            navigate_to("Jobs")

    with col_a3:
        st.markdown("""
        <div class="action-workflow-card">
            <div>
                <div class="card-tracker">03 · OPTIMIZATION</div>
                <div class="card-heading">Improve & prepare</div>
                <div class="card-body-text">
                    Tailor your experience bullets for a specific position and consult the grounded career mentor.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Resume Studio →", key="btn_open_studio", use_container_width=True):
            navigate_to("Resume Studio")




# =====================================================================
# 5. VIEW: 👤 PROFILE
# =====================================================================
elif st.session_state.active_tab == "Profile":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 800; color: #2563eb; letter-spacing: 0.08em; text-transform: uppercase;">PROFILE SETUP</div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0f172a; margin-top: 4px; margin-bottom: 6px; letter-spacing: -0.02em;">
            Build your verified career profile
        </h1>
        <p style="font-size: 1.05rem; color: #475569; margin: 0;">
            Upload your professional resume document to extract structured competencies, verified skills, and career orientation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload box container matching reference app
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
        <div style="font-size: 1.2rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;">Upload your resume document</div>
        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 18px;">
            Our extraction engine parses competencies, verified experience, and career direction for your explicit review.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose resume file (PDF, DOCX, TXT)",
        type=["pdf", "docx", "doc", "txt"],
        help="All parsing executes in-memory. Sensitive contact information is protected.",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("⚡ Parse Resume & Verify Profile", type="primary", use_container_width=True):
            with st.spinner("Extracting text and running structured profile parser..."):
                raw_text = load_document(uploaded_file, filename=uploaded_file.name)
                raw_p = parse_resume(raw_text)
                st.session_state.raw_profile = raw_p
                st.session_state.matched_jobs = search_jobs(raw_p.to_search_text(), top_k=TOP_K_JOBS)
                st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                st.session_state.cv_suggestions = None
                st.session_state.tailored_resume_content = None
                st.toast("Profile extracted successfully!")
                st.rerun()

    profile = st.session_state.current_profile
    if profile:
        st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 28px 0;'>", unsafe_allow_html=True)
        st.markdown('<h3 style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-bottom: 16px;">Verified Candidate Profile</h3>', unsafe_allow_html=True)

        # Overview Stats
        cp1, cp2, cp3, cp4 = st.columns(4)
        with cp1:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Candidate</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-top: 2px;">{profile.name}</div>
            </div>
            """, unsafe_allow_html=True)
        with cp2:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Target Orientation</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #2563eb; margin-top: 2px;">{profile.target_role}</div>
            </div>
            """, unsafe_allow_html=True)
        with cp3:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Experience</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-top: 2px;">{profile.years_of_experience:.1f} yrs</div>
            </div>
            """, unsafe_allow_html=True)
        with cp4:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Skills Extracted</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-top: 2px;">{len(profile.skills)} skills</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_pr1, col_pr2 = st.columns([1, 2])
        with col_pr1:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #0f172a;">Contact & Privacy Shield</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"**Email:** `{profile.email or 'Redacted'}`")
            st.markdown(f"**Phone:** `{profile.phone or 'Redacted'}`")
            st.markdown(f"**Location:** `{profile.location or 'Not specified'}`")
            if st.session_state.pii_masking_enabled:
                st.caption("🔒 PII redaction active: Direct contact information is protected.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_pr2:
            st.markdown("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #0f172a;">Verified Skills</h4>
            """, unsafe_allow_html=True)
            skills_html = "".join([f'<span class="skill-tag">{s}</span>' for s in profile.skills])
            st.markdown(skills_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 18px;">
            <h4 style="margin: 0 0 8px 0; color: #0f172a;">Executive Summary</h4>
        """, unsafe_allow_html=True)
        st.write(profile.summary)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 💼 Professional Experience History")
        for exp in profile.experience:
            with st.expander(f"🏢 {exp.role} · {exp.company} ({exp.duration})", expanded=True):
                for bullet in exp.description:
                    st.markdown(f"• {bullet}")

        if profile.education:
            st.markdown("#### 🎓 Education & Credentials")
            for edu in profile.education:
                st.markdown(f"• **{edu.degree}** — {edu.institution} {f'({edu.graduation_year})' if edu.graduation_year else ''}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next: Explore Matching Roles in Jobs →", type="primary"):
            navigate_to("Jobs")

    else:
        st.info("Upload your professional resume above to inspect your verified competencies, experience history, and job matches.")


# =====================================================================
# 6. VIEW: 💼 JOBS
# =====================================================================
elif st.session_state.active_tab == "Jobs":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 800; color: #2563eb; letter-spacing: 0.08em; text-transform: uppercase;">OPPORTUNITIES</div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0f172a; margin-top: 4px; margin-bottom: 6px; letter-spacing: -0.02em;">
            Matching opportunities
        </h1>
        <p style="font-size: 1.05rem; color: #475569; margin: 0 0 12px 0;">
            Explore verified roles matched against your confirmed profile using semantic vector search.
        </p>
        <div style="font-size: 0.85rem; color: #64748b;">
            🗄️ <b>Job corpus:</b> Kaggle Naukri (21,739 postings) · <b>Search:</b> Semantic FAISS Vector Index
        </div>
    </div>
    """, unsafe_allow_html=True)

    has_profile = st.session_state.raw_profile is not None

    if not has_profile:
        st.markdown("""
        <div class="gating-card">
            <div class="badge-action-needed">● Action needed</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">
                Candidate profile confirmation required
            </div>
            <div style="font-size: 0.92rem; color: #475569; margin-bottom: 20px;">
                Semantic matching requires an approved candidate profile with verified skills and target role.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2, col_g3 = st.columns([1.5, 1.5, 3])
        with col_g1:
            if st.button("Upload resume in Profile →", type="primary", use_container_width=True):
                navigate_to("Profile")
        with col_g2:
            if st.button("Review profile →", use_container_width=True):
                navigate_to("Profile")

    else:
        # Auto-compute matches if empty
        if not st.session_state.matched_jobs:
            st.session_state.matched_jobs = search_jobs(st.session_state.current_profile.to_search_text(), top_k=TOP_K_JOBS)
            if st.session_state.matched_jobs and not st.session_state.selected_job:
                st.session_state.selected_job = st.session_state.matched_jobs[0]

        matched_jobs = st.session_state.matched_jobs

        # Filter bar
        col_fl1, col_fl2 = st.columns([3, 1])
        with col_fl1:
            search_query = st.text_input("Filter opportunities by title, tech stack, or location", placeholder="Search e.g. 'Machine Learning', 'Remote', 'AWS'...")
        with col_fl2:
            min_score = st.slider("Min Match %", min_value=0, max_value=90, value=30, step=5)

        filtered_jobs = [
            j for j in matched_jobs
            if j.get("match_score", 0) >= min_score and (
                not search_query or
                search_query.lower() in j["title"].lower() or
                search_query.lower() in j["company"].lower() or
                search_query.lower() in j["required_skills"].lower()
            )
        ]

        st.caption(f"Showing **{len(filtered_jobs)}** verified semantic match opportunities:")

        for job in filtered_jobs:
            score = job.get("match_score", 0.0)
            score_pill_class = "match-pill-high" if score >= 50 else "match-pill-mid"

            is_selected = st.session_state.selected_job and st.session_state.selected_job.get("job_id") == job.get("job_id")
            border_style = "border: 2px solid #2563eb;" if is_selected else ""

            st.markdown(f"""
            <div class="job-listing-card" style="{border_style}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin: 0;">{job['title']}</div>
                        <div style="font-size: 0.88rem; color: #64748b; margin-top: 2px;">
                            🏢 <b>{job['company']}</b> &nbsp;•&nbsp; 📍 {job['location']} &nbsp;•&nbsp; 💰 {job['salary_range']} &nbsp;•&nbsp; ⏳ {job['employment_type']}
                        </div>
                    </div>
                    <div class="{score_pill_class}">{int(round(score))}% Match</div>
                </div>
                <div style="font-size: 0.92rem; color: #334155; line-height: 1.5; margin-bottom: 12px;">{job['description']}</div>
                <div>
                    <span style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Required Skills:</span>
                    <span style="font-size: 0.82rem; color: #1e293b;"> {job['required_skills']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_btn1, col_btn2, col_btn3 = st.columns([1.6, 1.6, 2.8])
            with col_btn1:
                if st.button("🎯 Target in Resume Studio", key=f"tgt_{job['job_id']}", use_container_width=True):
                    st.session_state.selected_job = job
                    st.session_state.cv_suggestions = None
                    st.session_state.tailored_resume_content = None
                    st.toast(f"Targeted {job['title']}!")
                    navigate_to("Resume Studio")

            with col_btn2:
                if st.button("💬 Ask Mentor About Role", key=f"mnt_{job['job_id']}", use_container_width=True):
                    st.session_state.selected_job = job
                    st.session_state.mentor_chat_history.append({
                        "role": "user",
                        "content": f"How well do my qualifications fit the {job['title']} role at {job['company']}? What should I highlight in my interview?"
                    })
                    st.toast(f"Loaded {job['title']} into Career Mentor!")
                    navigate_to("Career Mentor")

            with col_btn3:
                st.caption(f"ID: {job['job_id']} · FAISS Distance: {job.get('raw_distance', 'N/A')}")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# =====================================================================
# 7. VIEW: 📄 RESUME STUDIO
# =====================================================================
elif st.session_state.active_tab == "Resume Studio":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 800; color: #2563eb; letter-spacing: 0.08em; text-transform: uppercase;">RESUME STUDIO</div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0f172a; margin-top: 4px; margin-bottom: 6px; letter-spacing: -0.02em;">
            Tailor your resume for a target role
        </h1>
        <p style="font-size: 1.05rem; color: #475569; margin: 0;">
            Address role-specific skill gaps and optimize experience bullet points strictly grounded in your verified history.
        </p>
    </div>
    """, unsafe_allow_html=True)

    has_profile = st.session_state.raw_profile is not None
    if not has_profile:
        st.markdown("""
        <div class="gating-card">
            <div class="badge-action-needed">● Action needed</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">
                Candidate profile confirmation required
            </div>
            <div style="font-size: 0.92rem; color: #475569; margin-bottom: 20px;">
                Resume tailoring requires an approved candidate profile with verified experience.
            </div>
        </div>
        """, unsafe_allow_html=True)
        col_rg1, col_rg2 = st.columns([1.5, 4])
        with col_rg1:
            if st.button("Upload resume in Profile →", type="primary", use_container_width=True):
                navigate_to("Profile")
    else:
        profile = st.session_state.current_profile
        selected_job = st.session_state.selected_job

        if not selected_job:
            st.warning("Please select a target job from the Jobs tab to begin tailoring.")
            if st.button("Browse Matching Opportunities in Jobs →", type="primary"):
                navigate_to("Jobs")
        else:
            # Active Target Role Header Box
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: #2563eb; text-transform: uppercase;">Target Opportunity</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-top: 2px;">{selected_job['title']}</div>
                        <div style="font-size: 0.88rem; color: #64748b; margin-top: 2px;">🏢 {selected_job['company']} &nbsp;•&nbsp; 📍 {selected_job['location']} &nbsp;•&nbsp; 💰 {selected_job['salary_range']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.cv_suggestions is None:
                if st.button("✨ Run AI Gap Analysis & Google XYZ Optimization", type="primary"):
                    with st.spinner("Analyzing candidate experience against job requirements..."):
                        res = generate_cv_improvements(profile, selected_job)
                        st.session_state.cv_suggestions = res
                        st.rerun()

            suggestions = st.session_state.cv_suggestions
            if suggestions:
                # Fit Score & Skills Overlap
                col_ft1, col_ft2 = st.columns([1, 2])
                with col_ft1:
                    st.markdown(f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; text-align: center;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Role Alignment Score</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #059669; margin: 6px 0;">{suggestions.overall_fit_score}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(suggestions.overall_fit_score / 100.0)

                with col_ft2:
                    st.markdown("""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px;">
                        <h4 style="margin: 0 0 10px 0; color: #0f172a;">Competency Breakdown</h4>
                    """, unsafe_allow_html=True)
                    match_tags = "".join([f'<span class="skill-tag skill-tag-match">✓ {s}</span>' for s in suggestions.matching_skills])
                    miss_tags = "".join([f'<span class="skill-tag skill-tag-missing">+ {s}</span>' for s in suggestions.missing_hard_skills])
                    st.markdown(f"**Verified Strengths:**<br/>{match_tags}", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top: 8px;'><b>Critical Gaps to Address:</b><br/>{miss_tags}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;'>", unsafe_allow_html=True)

                # Google XYZ Formula Bullet Critiques
                st.markdown('<h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;">Experience Bullet Optimization</h3>', unsafe_allow_html=True)
                st.caption("Google XYZ Impact Formula: 'Accomplished [X] as measured by [Y], by doing [Z]'")

                for idx, critique in enumerate(suggestions.bullet_point_improvements, 1):
                    st.markdown(f"""
                    <div class="xyz-critique-box">
                        <div style="color: #dc2626; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">❌ Original Experience Bullet:</div>
                        <div style="background: #fef2f2; border-left: 3px solid #dc2626; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 12px; font-size: 0.92rem; color: #1e293b;">
                            "{critique.original_bullet}"
                        </div>
                        <div style="color: #d97706; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">⚠️ Recruiter Assessment & Critique:</div>
                        <div style="color: #475569; margin-bottom: 12px; font-size: 0.9rem; line-height: 1.5;">
                            {critique.critique_weakness}
                        </div>
                        <div style="color: #059669; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">✅ Google XYZ Impact Formula Transformation:</div>
                        <div style="background: #ecfdf5; border-left: 3px solid #059669; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.93rem; font-weight: 600; color: #065f46;">
                            {critique.rewritten_bullet_with_metrics}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;'>", unsafe_allow_html=True)

                st.markdown('<h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 8px;">Tailored Executive Summary</h3>', unsafe_allow_html=True)
                st.info(suggestions.tailored_summary)

                st.markdown('<h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 8px;">Interview Talking Points</h3>', unsafe_allow_html=True)
                for tp in suggestions.interview_talking_points:
                    st.markdown(f"• 🗣️ **Interview Prep:** {tp}")

                # Stretch Goal 3: Complete Tailored Resume Rewrite
                st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;'>", unsafe_allow_html=True)
                st.markdown('<h3 style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;">ATS-Tailored Resume Generator</h3>', unsafe_allow_html=True)
                st.caption("Generate a full-length ATS-optimized resume markdown document customized for this exact job description.")

                if st.button("📄 Generate Full Tailored Resume Document", key="btn_gen_full_cv"):
                    with st.spinner("Rewriting complete resume with verified ATS keywords..."):
                        tailored_md = rewrite_tailored_resume(profile, selected_job)
                        st.session_state.tailored_resume_content = tailored_md

                if st.session_state.tailored_resume_content:
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown("#### 📄 Original Resume")
                        st.text_area("Original Content", profile.to_search_text(), height=360, disabled=True)
                    with col_t2:
                        st.markdown("#### ✨ AI-Tailored Resume")
                        st.text_area("Tailored Markdown", st.session_state.tailored_resume_content, height=360)

                    st.download_button(
                        label="💾 Download Tailored Resume (.md)",
                        data=st.session_state.tailored_resume_content,
                        file_name=f"tailored_resume_{selected_job.get('title', 'job').replace(' ', '_').lower()}.md",
                        mime="text/markdown"
                    )


# =====================================================================
# 8. VIEW: 💬 CAREER MENTOR (CONVERSATIONAL AI AGENT)
# =====================================================================
elif st.session_state.active_tab == "Career Mentor":
    # Top action bar matching reference app
    col_t_left, col_t_right = st.columns([3, 2])
    with col_t_left:
        st.markdown(f"""
        <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 14px;">
            <span style="font-size: 0.78rem; font-weight: 600; color: #475569; background: #ffffff; border: 1px solid #e2e8f0; padding: 4px 12px; border-radius: 8px;">
                ↻ History ({len(st.session_state.mentor_chat_history)})
            </span>
            <span style="font-size: 0.78rem; font-weight: 600; color: #059669; background: #ecfdf5; border: 1px solid #a7f3d0; padding: 4px 12px; border-radius: 8px;">
                ● Memory Active
            </span>
            <span style="font-size: 0.78rem; font-weight: 600; color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; padding: 4px 12px; border-radius: 8px;">
                ● Database Synced
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_t_right:
        c_nw, c_clr = st.columns(2)
        with c_nw:
            if st.button("+ New Chat", use_container_width=True):
                st.session_state.mentor_chat_history = [
                    {
                        "role": "assistant",
                        "content": "Conversation memory reset. What's on your career agenda today?",
                        "citations": []
                    }
                ]
                st.rerun()

    # Title matching reference app
    st.markdown("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 24px;">
        <h1 style="font-size: 2.3rem; font-weight: 800; color: #0f172a; margin-bottom: 8px; letter-spacing: -0.02em;">
            What's on the agenda today?
        </h1>
        <p style="font-size: 1.05rem; color: #64748b; max-width: 650px; margin: 0 auto;">
            Ask anything freely about your career path, behavioral interviews, salary negotiation, or technical roadmaps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Active Context Capsule
    active_cand_str = st.session_state.current_profile.name if st.session_state.current_profile else "No resume loaded"
    active_role_str = st.session_state.selected_job.get("title", "No target job chosen") if st.session_state.selected_job else "No target role chosen"

    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 18px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div style="font-size: 0.85rem; color: #0f172a;">
            <b style="color: #2563eb;">ACTIVE MENTOR CONTEXT:</b> &nbsp; Candidate: <b>{active_cand_str}</b> &nbsp;•&nbsp; Target Job: <b>{active_role_str}</b>
        </div>
        <span style="font-size: 0.75rem; color: #059669; font-weight: 700; background: #ecfdf5; border: 1px solid #a7f3d0; padding: 2px 8px; border-radius: 9999px;">
            Resume auto-linked via chat
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Render Conversation History
    for msg in st.session_state.mentor_chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "⚡"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg.get("citations"):
                with st.expander("📚 Verified Grounded Sources & Citations", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(f"""
                        <div class="mentor-citation-box">
                            <b>• {c['title']}</b> ({c['source_type']})<br/>
                            <span style="color: #64748b; font-size: 0.8rem;">"{c['snippet']}"</span>
                        </div>
                        """, unsafe_allow_html=True)

    # Prompt Action Chips matching reference app
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Suggested discussion topics:")
    col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
    suggested_q = None

    with col_sc1:
        if st.button("🎯 Explain My Role Match", use_container_width=True):
            suggested_q = "Explain my match score for my active job and what experience gives me the highest leverage."
    with col_sc2:
        if st.button("🔍 Top 3 Missing Skills", use_container_width=True):
            suggested_q = "What are the top 3 critical skill gaps between my resume and target job that I should address first?"
    with col_sc3:
        if st.button("🎙️ Mock Interview Prep", use_container_width=True):
            suggested_q = "Act as the hiring manager for my target job and ask me 3 tough behavioral and technical interview questions based on my background."
    with col_sc4:
        if st.button("🚀 30-Day Transition Plan", use_container_width=True):
            suggested_q = "Create a structured 30-day learning roadmap to close the skill gap for my target job role."

    # Chat Input Box
    user_message = st.chat_input("Ask anything freely about your career path, interviews, or roadmaps...")
    query_to_run = user_message or suggested_q

    if query_to_run:
        st.session_state.mentor_chat_history.append({"role": "user", "content": query_to_run})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query_to_run)

        cand_ctx = st.session_state.current_profile.to_search_text() if st.session_state.current_profile else "No resume loaded"
        job_ctx = str(st.session_state.selected_job) if st.session_state.selected_job else "No target job chosen"

        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("SmartHire Mentor referencing verified roadmaps and career history..."):
                response = answer_career_question(
                    question=query_to_run,
                    chat_history=st.session_state.mentor_chat_history,
                    candidate_context=cand_ctx,
                    job_context=job_ctx
                )
                st.markdown(response["answer"])

                if response.get("citations"):
                    with st.expander("📚 Verified Grounded Sources & Citations", expanded=False):
                        for c in response["citations"]:
                            st.markdown(f"""
                            <div class="mentor-citation-box">
                                <b>• {c['title']}</b> ({c['source_type']})<br/>
                                <span style="color: #64748b; font-size: 0.8rem;">"{c['snippet']}"</span>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.mentor_chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "citations": response.get("citations", [])
                })


# =====================================================================
# 9. VIEW: 🔬 MODEL LAB (STRETCH GOAL 4 & EVALUATION)
# =====================================================================
elif st.session_state.active_tab == "Model Lab":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 0.75rem; font-weight: 800; color: #2563eb; letter-spacing: 0.08em; text-transform: uppercase;">BENCHMARK & EVALUATION</div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #0f172a; margin-top: 4px; margin-bottom: 6px; letter-spacing: -0.02em;">
            Dual-Model Embedding Benchmark Lab
        </h1>
        <p style="font-size: 1.05rem; color: #475569; margin: 0;">
            Compare high-dimensional semantic vector embeddings (Google Gemini text-embedding-004) against lexical keyword representations (TF-IDF).
        </p>
    </div>
    """, unsafe_allow_html=True)

    jobs_df = pd.read_csv(JOBS_DATA_PATH)
    job_dicts = jobs_df.to_dict(orient="records")

    sample_query = st.text_input(
        "Candidate Query for Embedding Comparison",
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

            st.markdown("<br>", unsafe_allow_html=True)
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

    st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 28px 0;'>", unsafe_allow_html=True)
    st.markdown('### 📋 Capstone Evaluation Report (answer_quality.md)')

    eval_file = REPORTS_DIR / "answer_quality.md"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Report file not found. Run python -m src.evaluate to compile.")
