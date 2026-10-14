"""
CV Suggestions Generator Module for SmartHire GenAI.
Analyzes skill gaps, critiques resume bullet points using the XYZ formula,
generates tailored summaries, and creates tailored resume rewrites.
"""

import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, LLM_MODEL, DEFAULT_TEMPERATURE, check_api_key
from src.parsing.resume_parser import CandidateProfile
from src.generate.prompts import (
    BASELINE_CV_IMPROVEMENT_PROMPT,
    OPTIMIZED_CV_IMPROVEMENT_PROMPT,
    TAILORED_RESUME_REWRITE_PROMPT,
    PROMPT_COMPARISON_NOTES
)


# ==========================================
# 1. Pydantic Schemas for CV Suggestions
# ==========================================

class BulletCritique(BaseModel):
    original_bullet: str = Field(description="The original candidate resume bullet point")
    critique_weakness: str = Field(description="Analysis of why this bullet is weak, passive, or lacking metrics")
    rewritten_bullet_with_metrics: str = Field(description="Enhanced bullet point following the Google XYZ formula")


class CVSuggestionsResult(BaseModel):
    overall_fit_score: int = Field(default=75, description="Estimated percentage fit score for the role")
    matching_skills: List[str] = Field(default_factory=list, description="Skills present in both resume and job requirements")
    missing_hard_skills: List[str] = Field(default_factory=list, description="Key skills required by the job that are missing from candidate profile")
    bullet_point_improvements: List[BulletCritique] = Field(default_factory=list, description="List of bullet critiques and improved rewrites")
    tailored_summary: str = Field(description="Custom executive professional summary targeted for this specific job")
    interview_talking_points: List[str] = Field(default_factory=list, description="Strategic talking points to prepare for interviews")


# ==========================================
# 2. Heuristic Mock Fallback Generator
# ==========================================

def _mock_generate_cv_improvements(profile: CandidateProfile, job: Dict[str, Any]) -> CVSuggestionsResult:
    """Deterministic fallback for local testing and offline execution."""
    job_skills_str = str(job.get("required_skills", ""))
    job_skills = [s.strip() for s in re.split(r"[,;|]+", job_skills_str) if s.strip()]
    candidate_skills_lower = {s.lower() for s in profile.skills}

    matching = []
    missing = []
    for js in job_skills:
        if js.lower() in candidate_skills_lower or any(js.lower() in cs for cs in candidate_skills_lower):
            matching.append(js)
        else:
            missing.append(js)

    if not missing:
        missing = ["CI/CD Automation", "Production Monitoring"]

    # Gather sample bullets to improve
    bullets = []
    for exp in profile.experience:
        for b in exp.description:
            bullets.append(b)

    critiques = []
    sample_bullet = bullets[0] if bullets else "Worked on application development and data pipelines."
    critiques.append(
        BulletCritique(
            original_bullet=sample_bullet,
            critique_weakness="Lacks quantifiable metrics, business impact, and specific architectural scale.",
            rewritten_bullet_with_metrics="Architected and deployed production pipelines, reducing latency by 42% and processing 500k+ daily transactions."
        )
    )

    fit_score = max(55, min(95, int(job.get("match_score", 75))))

    return CVSuggestionsResult(
        overall_fit_score=fit_score,
        matching_skills=matching[:5] if matching else ["Python", "SQL"],
        missing_hard_skills=missing[:5],
        bullet_point_improvements=critiques,
        tailored_summary=(
            f"Versatile {profile.target_role} with strong background in {', '.join(matching[:3]) if matching else 'modern software engineering'}. "
            f"Proven track record of driving operational efficiency, eager to leverage technical skills to accelerate {job.get('company', 'target company')}'s mission."
        ),
        interview_talking_points=[
            f"Discuss hands-on experience bridging {', '.join(profile.skills[:2])} with production business needs.",
            f"Explain how you plan to rapidly upskill in missing tools like {missing[0] if missing else 'cloud tools'}.",
            "Highlight measurable performance improvements from previous role achievements."
        ]
    )


# ==========================================
# 3. LLM CV Improvement Generator
# ==========================================

def generate_cv_improvements(
    profile: CandidateProfile,
    target_job: Dict[str, Any],
    model_name: Optional[str] = None
) -> CVSuggestionsResult:
    """
    Generates actionable CV improvement recommendations using Gemini with structured JSON output.
    """
    if not check_api_key():
        return _mock_generate_cv_improvements(profile, target_job)

    model_to_use = model_name or LLM_MODEL
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_to_use,
            temperature=DEFAULT_TEMPERATURE,
            api_key=GEMINI_API_KEY
        )

        structured_llm = llm.with_structured_output(CVSuggestionsResult)

        # Collect bullet points from candidate profile
        bullets = []
        for exp in profile.experience:
            for b in exp.description:
                bullets.append(f"- {b}")
        bullets_text = "\n".join(bullets[:6]) if bullets else "- Developed applications and collaborated with cross-functional teams."

        prompt_str = OPTIMIZED_CV_IMPROVEMENT_PROMPT.format(
            job_title=target_job.get("title", ""),
            company=target_job.get("company", ""),
            required_skills=target_job.get("required_skills", ""),
            job_description=target_job.get("description", "")[:1500],
            candidate_name=profile.name,
            target_role=profile.target_role,
            candidate_skills=", ".join(profile.skills),
            candidate_summary=profile.summary,
            experience_bullets=bullets_text
        )

        result: CVSuggestionsResult = structured_llm.invoke(prompt_str)
        return result

    except Exception as exc:
        print(f"[Warning] LLM CV improvement generation failed: {exc}. Using fallback.")
        return _mock_generate_cv_improvements(profile, target_job)


def rewrite_tailored_resume(
    profile: CandidateProfile,
    target_job: Dict[str, Any],
    model_name: Optional[str] = None
) -> str:
    """
    Stretch Goal: Rewrites the candidate's entire resume to tailor it directly to the target job.
    """
    if not check_api_key():
        # Fallback markdown generator
        return f"""# {profile.name}
**Target Role:** {target_job.get('title', profile.target_role)} | **Application for:** {target_job.get('company', '')}

## Professional Summary
Dedicated and analytical professional with a strong track record of designing scalable solutions. Expert in {', '.join(profile.skills[:4])}, eager to deliver measurable impact as a {target_job.get('title', profile.target_role)} at {target_job.get('company', '')}.

## Tailored Technical Competencies
- **Core Skills:** {', '.join(profile.skills)}
- **Role Alignment:** {target_job.get('required_skills', '')}

## Professional Experience
### {profile.target_role} — Previous Organization
- Engineered performant software systems aligning with modern engineering standards.
- Accelerated development lifecycles while maintaining 99.9% uptime and rigorous test coverage.

## Education
{', '.join([f'{edu.degree} — {edu.institution}' for edu in profile.education]) if profile.education else 'B.S. in Computer Science'}
"""

    model_to_use = model_name or LLM_MODEL
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_to_use,
            temperature=DEFAULT_TEMPERATURE,
            api_key=GEMINI_API_KEY
        )
        prompt_str = TAILORED_RESUME_REWRITE_PROMPT.format(
            job_title=target_job.get("title", ""),
            company=target_job.get("company", ""),
            required_skills=target_job.get("required_skills", ""),
            resume_text=profile.to_search_text()
        )
        response = llm.invoke(prompt_str)
        return response.content
    except Exception as e:
        print(f"[Warning] Tailored resume rewrite failed: {e}")
        return f"Could not generate tailored resume: {str(e)}"


def run_prompt_comparison(
    profile: CandidateProfile,
    target_job: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes both Baseline and Optimized prompts for evaluation and side-by-side analysis.
    """
    baseline_output = ""
    if check_api_key():
        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                api_key=GEMINI_API_KEY
            )
            raw_base = llm.invoke(BASELINE_CV_IMPROVEMENT_PROMPT.format(
                resume_text=profile.to_search_text(),
                job_description=target_job.get("description", "")
            ))
            baseline_output = raw_base.content
        except Exception as e:
            baseline_output = f"[Error running baseline prompt: {e}]"
    else:
        baseline_output = (
            "Here are some suggestions:\n"
            "1. You should learn more about the tools mentioned in the job description.\n"
            "2. Make your resume shorter and more concise.\n"
            "3. Add more details about your daily tasks and responsibilities."
        )

    optimized_result = generate_cv_improvements(profile, target_job)

    return {
        "job_title": target_job.get("title", ""),
        "company": target_job.get("company", ""),
        "baseline_prompt_output": baseline_output,
        "optimized_prompt_output": optimized_result.model_dump(),
        "comparison_critique": PROMPT_COMPARISON_NOTES
    }
