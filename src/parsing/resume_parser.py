"""
Resume Parser Module for SmartHire GenAI.
Uses LLMs with strict structured output (Pydantic schema) to extract clean JSON profiles from raw resume text.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, LLM_MODEL, DEFAULT_TEMPERATURE, check_api_key


# ==========================================
# 1. Pydantic Schemas for Strict Validation
# ==========================================

class WorkExperience(BaseModel):
    company: str = Field(description="Name of the company or organization")
    role: str = Field(description="Job title or role held")
    duration: str = Field(description="Dates or duration of employment, e.g., '2021 - Present'")
    description: List[str] = Field(default_factory=list, description="Key achievements, projects, or responsibilities")


class Education(BaseModel):
    institution: str = Field(description="University, college, or school name")
    degree: str = Field(description="Degree and major, e.g., 'B.S. in Computer Science'")
    graduation_year: Optional[str] = Field(default=None, description="Year of graduation or expected date")


class CandidateProfile(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(default="", description="Email address if present")
    phone: Optional[str] = Field(default="", description="Phone number if present")
    location: Optional[str] = Field(default="", description="Current location or city/state")
    target_role: str = Field(description="Inferred or stated target career role (e.g. Data Scientist, Full Stack Developer, DevOps Engineer)")
    years_of_experience: float = Field(default=0.0, description="Total estimated years of relevant professional experience")
    skills: List[str] = Field(default_factory=list, description="List of technical, analytical, and domain skills extracted from the resume")
    experience: List[WorkExperience] = Field(default_factory=list, description="Chronological work experience history")
    education: List[Education] = Field(default_factory=list, description="Educational background")
    summary: str = Field(description="Concise 2-3 sentence executive profile summary of the candidate's core strengths and career goals")

    def to_search_text(self) -> str:
        """
        Creates a rich semantic representation for embedding and vector similarity search.
        """
        skills_str = ", ".join(self.skills)
        exp_roles = ", ".join([exp.role for exp in self.experience])
        return (
            f"Target Role: {self.target_role}. "
            f"Key Technical Skills: {skills_str}. "
            f"Experience Roles: {exp_roles}. "
            f"Professional Summary: {self.summary}"
        )


def redact_profile_pii(profile: CandidateProfile, anonymize_name: bool = False) -> CandidateProfile:
    """
    Redacts Personally Identifiable Information (PII) to prevent sensitive candidate
    contact data from being exposed on screen or to external viewers.
    """
    masked_email = ""
    if profile.email:
        parts = profile.email.split("@")
        if len(parts) == 2:
            user, domain = parts
            masked_email = f"{user[0]}***{user[-1] if len(user) > 1 else ''}@{domain}"
        else:
            masked_email = "***@***.com"

    masked_phone = ""
    if profile.phone:
        digits = re.sub(r"\D", "", profile.phone)
        if len(digits) >= 4:
            masked_phone = f"(***) ***-{digits[-4:]}"
        else:
            masked_phone = "***-***-****"

    masked_location = "[Location Redacted for Privacy]" if profile.location else ""
    masked_name = profile.name
    if anonymize_name:
        import hashlib
        short_hash = hashlib.md5(profile.name.encode()).hexdigest()[:4].upper()
        masked_name = f"Candidate #{short_hash}"

    return profile.model_copy(update={
        "name": masked_name,
        "email": masked_email,
        "phone": masked_phone,
        "location": masked_location
    })


# ==========================================
# 2. Mock Fallback Parser (For Testing without API Key)
# ==========================================

def _mock_parse_resume(raw_text: str) -> CandidateProfile:
    """
    Deterministic rule-based fallback parser when API key is not configured.
    Ensures zero crashes during initial setup and testing.
    """
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    name = lines[0] if lines else "Candidate"
    
    # Extract email & phone via regex
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", raw_text)
    phone_match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""

    # Detect skills from common keyword list
    common_skills = [
        "Python", "SQL", "Machine Learning", "Scikit-Learn", "Pandas", "NumPy",
        "React", "Node.js", "TypeScript", "JavaScript", "Docker", "Kubernetes",
        "AWS", "Terraform", "FastAPI", "PostgreSQL", "Git", "Tableau", "Power BI",
        "LangChain", "PyTorch", "TensorFlow", "Next.js", "Redis", "CI/CD"
    ]
    detected_skills = [skill for skill in common_skills if re.search(r"\b" + re.escape(skill) + r"\b", raw_text, re.IGNORECASE)]
    if not detected_skills:
        detected_skills = ["Software Engineering", "Problem Solving", "Data Analysis"]

    # Inferred target role
    target_role = "Software Engineer"
    lower_text = raw_text.lower()
    if "devops" in lower_text or "kubernetes" in lower_text:
        target_role = "DevOps & Cloud Engineer"
    elif "data scientist" in lower_text or "machine learning" in lower_text:
        target_role = "Data Scientist"
    elif "data analyst" in lower_text or "tableau" in lower_text or "power bi" in lower_text:
        target_role = "Data Analyst"
    elif "full stack" in lower_text or "react" in lower_text:
        target_role = "Full Stack Developer"

    return CandidateProfile(
        name=name,
        email=email,
        phone=phone,
        location="Local / Remote",
        target_role=target_role,
        years_of_experience=3.0,
        skills=detected_skills,
        experience=[
            WorkExperience(
                company="MetroAnalytics / CloudNexus",
                role=target_role,
                duration="2021 - Present",
                description=["Built resilient production systems", "Optimized performance and scalability"]
            )
        ],
        education=[
            Education(
                institution="University Partner",
                degree="B.S. in Computer Science",
                graduation_year="2021"
            )
        ],
        summary=f"Dedicated {target_role} proficient in {', '.join(detected_skills[:5])}. Experienced in developing reliable applications and driving business outcomes."
    )


# ==========================================
# 3. LLM-Powered Structured Parser
# ==========================================

RESUME_PARSE_PROMPT = """You are an expert technical recruiter and resume parsing AI.
Extract structured information from the provided raw resume text into the requested JSON schema.

Strict Requirements:
1. Identify the candidate's full name, contact information, and primary target role.
2. If the target role is not explicitly stated in an objective, infer the most accurate professional title from their latest experience and skills.
3. Extract all explicit technical and domain skills into a clean, normalized list of single skills (e.g. ['Python', 'SQL', 'FastAPI']).
4. Extract work history with company names, roles, employment duration, and distinct bullet points.
5. Extract educational qualifications with institution, degree, and graduation year.
6. Provide an executive 2-3 sentence summary highlighting the candidate's core value proposition.

Resume Content:
-------------------
{resume_text}
-------------------

Return ONLY the raw JSON adhering to the specified schema, without any conversational preamble or markdown codeblocks outside the JSON format.
"""

def parse_resume(raw_text: str, model_name: Optional[str] = None) -> CandidateProfile:
    """
    Parses resume text using Google Gemini with structured Pydantic output validation.
    Falls back gracefully if the API key is not configured or in case of rate limits.
    """
    if not check_api_key():
        # Fallback to local heuristic parser
        return _mock_parse_resume(raw_text)

    model_to_use = model_name or LLM_MODEL
    try:
        # Initialize LangChain Google GenAI chat model
        llm = ChatGoogleGenerativeAI(
            model=model_to_use,
            temperature=DEFAULT_TEMPERATURE,
            api_key=GEMINI_API_KEY
        )

        # Attempt native structured output
        structured_llm = llm.with_structured_output(CandidateProfile)
        prompt = RESUME_PARSE_PROMPT.format(resume_text=raw_text[:12000])
        parsed_profile: CandidateProfile = structured_llm.invoke(prompt)
        return parsed_profile

    except Exception as exc:
        # If native structured output failed (e.g., json parsing glitch or network),
        # use JsonOutputParser with prompt template as secondary robust attempt
        try:
            parser = JsonOutputParser(pydantic_object=CandidateProfile)
            prompt_template = PromptTemplate(
                template="{instructions}\n\nResume Content:\n{resume_text}\n\n{format_instructions}",
                input_variables=["instructions", "resume_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            chain = prompt_template | llm | parser
            result_dict = chain.invoke({
                "instructions": "Extract the candidate profile accurately into the specified JSON format.",
                "resume_text": raw_text[:12000]
            })
            return CandidateProfile(**result_dict)
        except Exception as fallback_exc:
            print(f"[Warning] LLM parser encountered an error: {fallback_exc}. Falling back to rule-based parser.")
            return _mock_parse_resume(raw_text)
