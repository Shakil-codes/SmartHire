# SmartHire GenAI: Capstone Final Project Report
**Resume Matching, Actionable CV Optimization & Grounded AI Career Mentor**

---

## 1. Executive Summary
**SmartHire GenAI** is an end-to-end generative AI career portal designed to bridge candidate credentials with industry opportunities. Rather than relying on rigid keyword matching, SmartHire leverages state-of-the-art dense semantic vector embeddings, LLM structured output parsing, multi-stage guardrails, and Retrieval-Augmented Generation (RAG).

The system allows a job seeker or student to:
1. Upload their raw resume (PDF or DOCX) to extract a clean, validated JSON profile.
2. Discover top matching jobs across a pre-collected corpus using dense vector similarity in FAISS.
3. Receive actionable, metric-driven CV improvement recommendations based on Google's XYZ formula.
4. Interact with a grounded AI Career Mentor with session conversation memory, strict hallucination prevention, and verified document citations.

---

## 2. System Architecture & Component Design

```
Raw CV (PDF/DOCX) ──► Loader & Chunker ──► LLM Structured Parser ──► Pydantic Profile (JSON)
                                                                            │
      ┌─────────────────────────────────────────────────────────────────────┼─────────────────────────┐
      ▼                                                                     ▼                         ▼
Profile Search Query                                                Target Job Selection        Career Query
      │                                                                     │                         │
      ▼                                                                     ▼                         ▼
Dense Embedding (text-embedding-004)                               Prompt Engineering        Guardrails Layer
      │                                                                     │                   (Safe / On-Topic)
      ▼                                                                     ▼                         │
FAISS Vector Search (Jobs Corpus)                                 CV Improvement & Rewrites           ▼
      │                                                                     │                FAISS Career Search
      ▼                                                                     │                         │
Ranked Matches with % Scores ───────────────────────────────────────────────┘                         ▼
                                                                                             Grounded RAG Pipeline
                                                                                             (LangChain + Memory)
                                                                                                      │
                                                                                                      ▼
                                                                                              Answer + Citations
```

### Key Modules:
- **`src/config.py`**: Centralized configuration management for API keys, model parameters, and file paths.
- **`src/parsing/loader.py`**: Robust text extraction supporting both local files and Streamlit in-memory buffers across PDF, DOCX, and TXT formats.
- **`src/parsing/resume_parser.py`**: Extracts high-fidelity candidate schemas using Gemini structured output with heuristic fallbacks.
- **`src/search/embed.py`**: Unified embedding interface utilizing Google `text-embedding-004` with deterministic local hash projection fallbacks.
- **`src/search/job_search.py`**: Vector index creation and similarity scoring via FAISS, calculating percentage match scores.
- **`src/generate/prompts.py` & `cv_suggestions.py`**: Standardized prompt library enforcing recruiter persona and the Google XYZ bullet formula.
- **`src/mentor/rag_chain.py`**: LangChain-orchestrated RAG pipeline with conversation memory, strict context grounding, and citation tracking.
- **`src/safety/guardrails.py`**: Defense against prompt injection, jailbreaking, and off-topic domain queries.
- **`app/streamlit_app.py`**: Interactive, responsive portal UI with custom glassmorphic aesthetics.

---

## 3. Detailed Technical Analysis

### 3.1 Module 1: Resume Parser (Structured Outputs)
- **Challenge:** Raw resume text varies wildly in layout (multi-column tables, varied headers, inconsistent dates). Free-form LLM text generation frequently produces non-deterministic markdown blocks that break downstream code.
- **Solution:** Enforced strict Pydantic schemas (`CandidateProfile`, `WorkExperience`, `Education`) using `ChatGoogleGenerativeAI.with_structured_output()`. If API calls are throttled or keys are absent, a deterministic regex/heuristic parser provides fallback continuity.

### 3.2 Module 2: Semantic Job Search (Embeddings & FAISS)
- **Challenge:** Keyword matching fails when candidate resumes use different terminology than job postings (e.g., "Predictive Modeling" vs. "Machine Learning", or "Kubernetes/Docker" vs. "Container Orchestration").
- **Solution:** Embeddings project textual representations into a continuous vector space where semantic proximity represents conceptual alignment. Vectors are indexed in FAISS using inner product / L2 metric computation, converting distance values to an intuitive 0-100% match score.

### 3.3 Module 3: Prompt Optimization & CV Improvement
- **Challenge:** Generic prompts produce vague feedback (e.g. *"improve your summary"*, *"learn more tools"*).
- **Solution:** Implemented the Google XYZ formula: `"Accomplished [X] as measured by [Y], by doing [Z]"`. The prompt forces the LLM to identify exact skill gaps and critique passive bullets into metric-driven achievements.
- **Prompt Comparison:**
  - *Baseline Prompt:* Produced generic, unformatted bullet lists with no metrics.
  - *Optimized Prompt:* Generated strict JSON containing hard skill gaps, overlapping strengths, bullet point rewrites with quantitative metrics, and interview talking points.

### 3.4 Module 4 & 5: Grounded RAG Mentor & Guardrails
- **Challenge:** Open-ended LLMs hallucinate non-existent certifications or courses, and are vulnerable to off-topic distraction or prompt injection attacks.
- **Solution:**
  1. *Guardrails Pre-Filter:* Intercepts prompt injection strings (`"ignore previous instructions"`) and off-topic queries (`"recipe for cake"`), returning courteous scoped refusals without incurring LLM token costs.
  2. *Strict Grounding:* The system prompt mandates that the model answer *only* from retrieved document chunks. If the query lies outside the knowledge base, the model returns: *"I do not have enough verified information in my career knowledge base to answer this question accurately."*
  3. *Citations:* Each response includes document titles and source excerpts.

---

## 4. Quantitative Evaluation Summary

As documented in `reports/answer_quality.md`, the system was evaluated against 4 quantitative benchmarks:

| Benchmark | Target | Measured Result | Analysis |
| :--- | :--- | :--- | :--- |
| **Retrieval Relevance (Hit Rate)** | > 75% | **75.0%** (9/12) | Successfully matched 4 diverse profiles (Data Science, Full Stack, DevOps, Fresher Analyst) to their respective role domains. |
| **Mentor Grounding Score** | Factual & Cited | **Passed** | Grounded answers consistently cite specific markdown guides and job references. |
| **Hallucination Prevention** | 100% Refusal | **100% Passed** | Out-of-corpus queries (e.g., quantum teleportation rocket propulsion) were identified and refused. |
| **Guardrails Efficacy** | Zero Injection Leak | **100% Passed** | Injection attempts and off-topic queries were blocked prior to LLM generation. |

---

## 5. Stretch Goals Implemented
1. **Conversation Memory:** The mentor maintains multi-turn dialog history (`st.session_state.mentor_chat_history`), enabling follow-up questions within a session.
2. **Document Citations:** Every mentor answer provides source titles and text excerpts.
3. **Full Resume Rewriter:** A dedicated feature that rewrites the candidate's entire resume in markdown, custom-tailored to the target job description.
4. **Interactive Sample Profiles:** The portal includes 4 pre-loaded candidate personas for quick evaluation.

---

## 6. What Worked & Lessons Learned
1. **Pydantic Validation is Essential:** Enforcing schemas at the LLM interface prevents UI rendering errors.
2. **Hybrid Heuristics + Embeddings:** Pairing fast regex pre-filters with dense vector retrieval provides speed and cost savings.
3. **Semantic Search Superiority:** Semantic vector search successfully connected candidate profiles to related job postings where exact keyword searches would have failed.

---

## 7. Limitations & Future Roadmap
1. **OCR for Scanned Resumes:** The current loader uses `pypdf`. For image-only scanned PDFs, integrating Tesseract OCR or Gemini Vision would expand coverage.
2. **Hybrid Search (BM25 + Dense Vectors):** Combining keyword search (BM25) with dense vector embeddings (FAISS) via Reciprocal Rank Fusion (RRF) would improve precision for rare tool names and exact acronyms.
3. **Live Job API Connectors:** Connecting with approved public job APIs (such as USAJOBS or Greenhouse public feeds) would allow real-time job indexing.
