"""
Central Configuration Module for SmartHire GenAI.
Loads environment variables, defines paths, and manages model settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (parent directory of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file from project root
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
# Ensure GOOGLE_API_KEY environment variable is set for langchain-google-genai
if GEMINI_API_KEY and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Model Configurations
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))

# Data and Asset Directories
DATA_DIR = PROJECT_ROOT / "data"
JOBS_DATA_PATH = PROJECT_ROOT / os.getenv("DATA_JOBS_PATH", "data/jobs/jobs.csv")
RESUMES_DIR = PROJECT_ROOT / os.getenv("RESUMES_DIR", "data/resumes")
CAREER_NOTES_DIR = PROJECT_ROOT / os.getenv("CAREER_NOTES_DIR", "data/career_notes")

# Vector Store Paths
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
JOB_VECTORSTORE_PATH = PROJECT_ROOT / os.getenv("JOB_VECTORSTORE_PATH", "vectorstore/job_faiss_index")
MENTOR_VECTORSTORE_PATH = PROJECT_ROOT / os.getenv("MENTOR_VECTORSTORE_PATH", "vectorstore/career_faiss_index")

# Reports Directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# Search & Retrieval Defaults
TOP_K_JOBS = int(os.getenv("TOP_K_JOBS", "5"))
TOP_K_CAREER_NOTES = int(os.getenv("TOP_K_CAREER_NOTES", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

def check_api_key() -> bool:
    """Check if the Gemini API key is configured."""
    return bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")
