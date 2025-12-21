"""
Lightweight data models for resume ranking.
"""
from dataclasses import dataclass, field


@dataclass
class JobFeatures:
    """
    Structured representation of a job listing.
    Always includes raw_text; structured fields are optional/best-effort.
    """
    raw_text: str
    job_title: str | None = None
    company_name: str | None = None
    required_qualifications: list[str] = field(default_factory=list)
    preferred_qualifications: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class ResumeDoc:
    """Container for resume file path and text content."""
    path: str
    raw_text: str


@dataclass
class ResumeScore:
    """
    Final ranking output for a single resume.
    Includes component scores and explanations.
    """
    path: str
    overall: float
    semantic: float
    lexical: float
    coverage: float
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
