"""
Resume ranking API for notebook usage.
Scores and ranks resumes against a job listing.
"""
import hashlib
import numpy as np
import os
import pickle
from typing import Callable, Sequence

from chunking import chunk_job, chunk_resume
from job_extract import extract_job_features_from_text
from logger import setup_logger
from metrics import SentenceTransformerStrategy, TfidfCosineStrategy, KeywordCoverageStrategy, max_chunk_similarity
from models import LLMProvider
from parsers import read_job_text, read_resume_text
from ranking_models import JobFeatures, ResumeDoc, ResumeScore

# Initialize logger
logger = setup_logger(name="ResumeRanking", log_file="resume_ranking.log")


def rank_resumes(
    resume_paths: Sequence[str],
    job_listing_path: str,
    *,
    resume_text_loader: Callable[[str], str] | None = None,
    job_text_loader: Callable[[str], str] | None = None,
    use_llm_job_extract: bool = True,
    use_llm_resume_extract: bool = False,
    llm: LLMProvider | None = None,
    enrich_top_k: int = 10,
    weights: dict[str, float] | None = None,
    include_evidence: bool = True,
    cache_dir: str | None = ".cache_ranker",
) -> list[ResumeScore]:
    """
    Rank resumes against a job listing using hybrid scoring.
    
    Args:
        resume_paths: List of paths to resume files
        job_listing_path: Path to job listing file
        resume_text_loader: Optional custom loader for resume text
        job_text_loader: Optional custom loader for job text
        use_llm_job_extract: Whether to use LLM for job feature extraction
        use_llm_resume_extract: Whether to use LLM for resume extraction (not implemented in MVP)
        llm: LLM provider for extraction (required if use_llm_job_extract=True)
        enrich_top_k: Number of top resumes to enrich (not implemented in MVP)
        weights: Custom weights for scoring metrics (semantic, lexical, coverage)
        include_evidence: Whether to include evidence snippets
        cache_dir: Directory for caching embeddings, None to disable caching
        
    Returns:
        list[ResumeScore]: Sorted list of resume scores (best to worst)
    """
    logger.info(f"Ranking {len(resume_paths)} resumes against job listing: {job_listing_path}")
    
    # Set default weights
    if weights is None:
        weights = {"semantic": 0.6, "lexical": 0.3, "coverage": 0.1}
    
    # Validate weights
    total_weight = sum(weights.values())
    if not (0.99 <= total_weight <= 1.01):
        logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Load job text
    logger.info("Loading job listing text")
    if job_text_loader:
        job_text = job_text_loader(job_listing_path)
    else:
        job_text = read_job_text(job_listing_path)
    
    # Extract job features if requested
    job_features = None
    if use_llm_job_extract:
        if llm is None:
            logger.warning("LLM extraction requested but no LLM provided, skipping extraction")
        else:
            logger.info("Extracting job features with LLM")
            job_features = extract_job_features_from_text(job_text, llm)
            logger.info(f"Extracted job: {job_features.job_title} at {job_features.company_name}")
    
    # Load all resume texts
    logger.info("Loading resume texts")
    resume_docs = []
    for path in resume_paths:
        try:
            if resume_text_loader:
                text = resume_text_loader(path)
            else:
                text = read_resume_text(path)
            resume_docs.append(ResumeDoc(path=path, raw_text=text))
        except Exception as e:
            logger.error(f"Error loading resume {path}: {e}")
    
    if not resume_docs:
        logger.warning("No resumes loaded successfully")
        return []
    
    # Initialize scoring strategies
    logger.info("Initializing scoring strategies")
    semantic_scorer = SentenceTransformerStrategy()
    lexical_scorer = TfidfCosineStrategy()
    keyword_scorer = KeywordCoverageStrategy()
    
    # Extract keywords for coverage scoring
    keywords = []
    if job_features:
        # Use structured keywords from job features
        keywords.extend(job_features.keywords)
        keywords.extend(job_features.required_qualifications)
        keywords.extend(job_features.preferred_qualifications)
        # Deduplicate and normalize
        keywords = list(set(kw.lower() for kw in keywords if kw))
    
    # Score each resume
    logger.info(f"Scoring {len(resume_docs)} resumes")
    results = []
    
    for idx, resume_doc in enumerate(resume_docs):
        logger.info(f"Scoring resume {idx + 1}/{len(resume_docs)}: {resume_doc.path}")
        score_result = _score_resume(
            job_text=job_text,
            job_features=job_features,
            resume_doc=resume_doc,
            weights=weights,
            semantic_scorer=semantic_scorer,
            lexical_scorer=lexical_scorer,
            keyword_scorer=keyword_scorer,
            keywords=keywords,
            include_evidence=include_evidence,
            cache_dir=cache_dir,
        )
        results.append(score_result)
    
    # Sort by overall score (descending)
    results.sort(key=lambda x: x.overall, reverse=True)
    
    logger.info("Ranking complete")
    return results


def _score_resume(
    job_text: str,
    job_features: JobFeatures | None,
    resume_doc: ResumeDoc,
    weights: dict[str, float],
    semantic_scorer: SentenceTransformerStrategy,
    lexical_scorer: TfidfCosineStrategy,
    keyword_scorer: KeywordCoverageStrategy,
    keywords: list[str],
    include_evidence: bool,
    cache_dir: str | None,
) -> ResumeScore:
    """
    Score a single resume against the job.
    
    Internal helper function for the ranking pipeline.
    """
    # Calculate semantic score
    semantic_score = _get_cached_similarity(
        job_text, resume_doc.raw_text, semantic_scorer, cache_dir
    )
    
    # Calculate lexical score
    lexical_score = lexical_scorer.calculate_score(job_text, resume_doc.raw_text)
    
    # Calculate coverage score
    if keywords:
        coverage_score = keyword_scorer.calculate_score(
            job_text, resume_doc.raw_text, keywords=keywords
        )
        matched_keywords, missing_keywords = keyword_scorer.get_matched_keywords(
            job_text, resume_doc.raw_text, keywords=keywords
        )
    else:
        coverage_score = keyword_scorer.calculate_score(job_text, resume_doc.raw_text)
        matched_keywords, missing_keywords = [], []
    
    # Combine scores
    overall_score = _combine_scores(
        {"semantic": semantic_score, "lexical": lexical_score, "coverage": coverage_score},
        weights
    )
    
    # Extract evidence snippets if requested
    evidence_snippets = []
    if include_evidence:
        evidence_snippets = _extract_evidence(
            job_text, job_features, resume_doc.raw_text, semantic_scorer
        )
    
    return ResumeScore(
        path=resume_doc.path,
        overall=overall_score,
        semantic=semantic_score,
        lexical=lexical_score,
        coverage=coverage_score,
        matched_keywords=matched_keywords[:10],  # Limit to top 10
        missing_keywords=missing_keywords[:10],  # Limit to top 10
        evidence_snippets=evidence_snippets,
        meta={
            "weights": weights,
            "num_matched_keywords": len(matched_keywords),
            "num_missing_keywords": len(missing_keywords),
        }
    )


def _combine_scores(metric_scores: dict[str, float], weights: dict[str, float]) -> float:
    """
    Combine multiple metric scores using weighted sum.
    
    Args:
        metric_scores: Dictionary of metric names to scores
        weights: Dictionary of metric names to weights
        
    Returns:
        float: Combined score
    """
    total = 0.0
    for metric, score in metric_scores.items():
        weight = weights.get(metric, 0.0)
        total += score * weight
    return total


def _extract_evidence(
    job_text: str,
    job_features: JobFeatures | None,
    resume_text: str,
    scorer: SentenceTransformerStrategy
) -> list[str]:
    """
    Extract evidence snippets showing best matches between job and resume.
    
    Args:
        job_text: Job listing text
        job_features: Optional structured job features
        resume_text: Resume text
        scorer: Scoring strategy for chunk matching
        
    Returns:
        list[str]: Top evidence snippets from resume
    """
    # Chunk job and resume
    job_chunks = chunk_job(job_text, job_features)
    resume_chunks = chunk_resume(resume_text)
    
    if not job_chunks or not resume_chunks:
        return []
    
    # Find best matches
    _, evidence_tuples = max_chunk_similarity(job_chunks, resume_chunks, scorer)
    
    # Sort by score and take top matches
    evidence_tuples.sort(key=lambda x: x[2], reverse=True)
    
    # Extract resume chunks as evidence
    evidence_snippets = [resume_chunk for _, resume_chunk, _ in evidence_tuples[:5]]
    
    return evidence_snippets


def _get_cached_similarity(
    text1: str,
    text2: str,
    scorer: SentenceTransformerStrategy,
    cache_dir: str | None
) -> float:
    """
    Calculate similarity with caching support.
    
    Args:
        text1: First text
        text2: Second text
        scorer: Scoring strategy
        cache_dir: Cache directory, None to disable caching
        
    Returns:
        float: Similarity score
    """
    if cache_dir is None:
        return scorer.calculate_score(text1, text2)
    
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache key
    model_name = scorer.config.get("model_name", "default")
    text1_hash = hashlib.sha256(text1.encode()).hexdigest()[:16]
    text2_hash = hashlib.sha256(text2.encode()).hexdigest()[:16]
    cache_key = f"{model_name}_{text1_hash}_{text2_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_key)
    
    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                score = pickle.load(f)
            return score
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
    
    # Calculate and cache
    score = scorer.calculate_score(text1, text2)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(score, f)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")
    
    return score
