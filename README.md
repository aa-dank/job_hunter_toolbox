# Job Hunter Toolbox

This toolbox automates resume and cover letter generation using LaTeX, Python, and LLM services to streamline the job application process. It also includes a resume ranking feature to help you select the best existing resume for a given job listing.

## Features

### 1. Resume & Cover Letter Generation
Automatically generate tailored resumes and cover letters from job descriptions using LLM-powered content extraction and LaTeX rendering.

### 2. Resume Ranking
Rank and compare existing resumes against a job listing using hybrid scoring (semantic similarity, lexical matching, and keyword coverage).

## Dependencies and Setup
1. Ensure you have a modern Python version installed (Python 3.13+).
2. Install [uv](https://docs.astral.sh/uv/) for Python package management.
3. Install LaTeX tools (TeX Live, MikTeX, etc.) with at least `pdflatex`.  
   • Optionally install `xelatex` for advanced formatting.  
4. Configure API credentials (e.g., OpenAI keys) for LLMProvider classes.

## Installation
```bash
# Install dependencies
uv sync

# Activate the virtual environment (optional - uv run handles this automatically)
source .venv/bin/activate
```

## Usage

### Resume Generation
• Configure an LLMProvider with your API credentials.
• Run the full application via:
  
  uv run python main.py

• The application flow will:
  - Extract job and user data.
  - Generate resume JSON and LaTeX files.
  - Open the generated resume and cover letter files for editing.
  - Upon confirmation (pressing Enter), compile the resume LaTeX file into a PDF and convert the cover letter text into a PDF.
  
• Edit the `.tex` and cover letter `.txt` files as needed before final PDF generation.

### Resume Ranking

Rank existing resumes against a job listing in a notebook or script:

```python
from ranking import rank_resumes

# Basic usage (no LLM required - fast and deterministic)
results = rank_resumes(
    resume_paths=["path/to/resume1.pdf", "path/to/resume2.pdf"],
    job_listing_path="path/to/job_listing.pdf",
    use_llm_job_extract=False,  # Works on plain text
)

# Display results
for i, score in enumerate(results, 1):
    print(f"{i}. {score.path}: {score.overall:.3f}")
    print(f"   Semantic: {score.semantic:.3f}, Lexical: {score.lexical:.3f}, Coverage: {score.coverage:.3f}")
```

**Advanced usage with LLM extraction:**

```python
from ranking import rank_resumes
from models import ChatGPT
from creds import OPENAI_KEY

# Initialize LLM
llm = ChatGPT(api_key=OPENAI_KEY, model="gpt-4o-mini", system_prompt="", temperature=0.3)

# Rank with structured job feature extraction
results = rank_resumes(
    resume_paths=["resume1.pdf", "resume2.pdf"],
    job_listing_path="job_listing.pdf",
    use_llm_job_extract=True,  # Extract structured keywords and requirements
    llm=llm,
    include_evidence=True,  # Include matching snippets
    weights={"semantic": 0.6, "lexical": 0.3, "coverage": 0.1},  # Custom weights
)

# Access detailed scores
for score in results:
    print(f"Resume: {score.path}")
    print(f"Overall: {score.overall:.3f}")
    print(f"Matched keywords: {score.matched_keywords[:5]}")
    print(f"Missing keywords: {score.missing_keywords[:5]}")
    print(f"Evidence: {score.evidence_snippets[0][:100]}...")
```

See `dev/ranking_demo.py` for more examples.

## Tools & Scripts
• The `LatexToolBox` class in `latex_toolbox.py` handles LaTeX compilation.  
• Main application logic for resume generation is in `main.py`.  
• Resume ranking API is in `ranking.py` with supporting modules (`parsers.py`, `chunking.py`, `job_extract.py`, `ranking_models.py`).  
• Demo scripts are in the `dev/` directory.
