"""
Deterministic text chunking for evidence support.
"""
import re
from ranking_models import JobFeatures


def chunk_job(text: str, job_features: JobFeatures | None = None) -> list[str]:
    """
    Chunk job listing text for scoring and evidence extraction.
    
    If structured features exist, includes required/preferred/responsibilities as chunks.
    Otherwise, uses heuristic bullet/line splitting.
    
    Args:
        text: Raw job listing text
        job_features: Optional structured job features
        
    Returns:
        list[str]: List of text chunks
    """
    chunks = []
    
    # Use structured fields if available
    if job_features:
        # Add qualifications as chunks
        chunks.extend(job_features.required_qualifications)
        chunks.extend(job_features.preferred_qualifications)
        chunks.extend(job_features.responsibilities)
    
    # Always add heuristic chunks from raw text
    heuristic_chunks = _chunk_by_bullets_and_lines(text)
    chunks.extend(heuristic_chunks)
    
    # Deduplicate and filter out very short chunks
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 20 and chunk.lower() not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk.lower())
    
    return unique_chunks


def chunk_resume(text: str) -> list[str]:
    """
    Chunk resume text for scoring.
    
    Uses heuristic bullet/line splitting and filters very short lines.
    
    Args:
        text: Raw resume text
        
    Returns:
        list[str]: List of text chunks
    """
    chunks = _chunk_by_bullets_and_lines(text)
    
    # Filter out very short chunks
    filtered_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]
    
    return filtered_chunks


def _chunk_by_bullets_and_lines(text: str) -> list[str]:
    """
    Split text by newlines and bullet symbols.
    
    Args:
        text: Raw text
        
    Returns:
        list[str]: List of text chunks
    """
    # Split by newlines first
    lines = text.split('\n')
    
    chunks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Further split by bullet symbols
        bullet_pattern = r'[•\-\*]\s+'
        sub_chunks = re.split(bullet_pattern, line)
        
        for chunk in sub_chunks:
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
    
    return chunks
