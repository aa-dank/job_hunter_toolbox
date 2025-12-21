"""
Pure file-to-text conversion utilities.
Reusable across ranking and application generation workflows.
"""
import json
import os
from pathlib import Path
from markitdown import MarkItDown


class UnsupportedFileFormatException(Exception):
    """Custom exception for unsupported file formats."""
    pass


def load_doc_text(path: str, md_converter: MarkItDown | None = None) -> str:
    """
    Load json/md/txt directly; otherwise convert via MarkItDown and return text_content.
    
    This is a pure function with no side effects:
    - No file writes
    - No build mutation
    - Only reads and converts
    
    Args:
        path: Path to the document file
        md_converter: Optional MarkItDown instance for conversion
        
    Returns:
        str: The document text content
        
    Raises:
        UnsupportedFileFormatException: If file format is not supported by MarkItDown
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    file_path = Path(path)
    extension = file_path.suffix.lower()
    
    # Handle JSON files - return pretty-printed JSON string
    if extension == '.json':
        with open(file_path, 'r') as file:
            data = json.load(file)
            return json.dumps(data, indent=2)
    
    # Handle markdown and text files - read as text
    if extension in ['.md', '.txt']:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    # Handle other formats via MarkItDown
    if md_converter is None:
        md_converter = MarkItDown()
    
    try:
        conversion_result = md_converter.convert(str(file_path))
        text_content = conversion_result.text_content
        
        if not text_content:
            raise ValueError(f"Empty content after conversion: {path}")
        
        return text_content
    
    except Exception as e:
        if e.__class__.__name__ == "UnsupportedFormatException":
            raise UnsupportedFileFormatException(f"Unsupported file format: {extension}")
        else:
            raise


def read_job_text(path: str) -> str:
    """
    Read job listing text from file.
    
    Args:
        path: Path to job listing file
        
    Returns:
        str: Job listing text
    """
    return load_doc_text(path)


def read_resume_text(path: str) -> str:
    """
    Read resume text from file.
    
    Args:
        path: Path to resume file
        
    Returns:
        str: Resume text
    """
    return load_doc_text(path)
