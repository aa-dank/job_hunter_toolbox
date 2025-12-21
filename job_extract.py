"""
Optional LLM-based job enrichment.
Pure function with no side effects (no file writes, no build mutation).
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from generation_schemas import JobDetails
from models import LLMProvider
from ranking_models import JobFeatures
from prompts.extraction_prompts import JOB_DETAILS_EXTRACTOR


def extract_job_features_from_text(job_text: str, llm: LLMProvider) -> JobFeatures:
    """
    Extract structured job features from text using LLM.
    
    Uses the same JOB_DETAILS_EXTRACTOR prompt and JobDetails schema
    as the existing application generation pipeline, but returns
    a JobFeatures object instead of mutating a build object.
    
    Args:
        job_text: Raw text content of the job listing
        llm: LLM provider for extraction
        
    Returns:
        JobFeatures: Structured job features with fallback to raw text only
    """
    try:
        json_parser = JsonOutputParser(pydantic_object=JobDetails)
        
        prompt = PromptTemplate(
            template=JOB_DETAILS_EXTRACTOR,
            input_variables=["job_description"],
            partial_variables={"format_instructions": json_parser.get_format_instructions()},
            template_format="jinja2",
            validate_template=False
        ).format(job_description=job_text)
        
        job_details = llm.get_response(prompt=prompt, need_json_output=True)
        
        if not isinstance(job_details, dict):
            # Fallback to raw text only
            return JobFeatures(raw_text=job_text)
        
        # Extract fields from JobDetails schema
        job_title = job_details.get("job_title")
        company_name = job_details.get("company_name")
        
        # Combine various qualification/requirement fields into lists
        required_qualifications = []
        if "required_qualifications" in job_details:
            req = job_details["required_qualifications"]
            if isinstance(req, list):
                required_qualifications = req
            elif isinstance(req, str):
                required_qualifications = [req]
        
        preferred_qualifications = []
        if "preferred_qualifications" in job_details:
            pref = job_details["preferred_qualifications"]
            if isinstance(pref, list):
                preferred_qualifications = pref
            elif isinstance(pref, str):
                preferred_qualifications = [pref]
        
        responsibilities = []
        if "responsibilities" in job_details:
            resp = job_details["responsibilities"]
            if isinstance(resp, list):
                responsibilities = resp
            elif isinstance(resp, str):
                responsibilities = [resp]
        
        keywords = []
        if "keywords" in job_details:
            kw = job_details["keywords"]
            if isinstance(kw, list):
                keywords = kw
            elif isinstance(kw, str):
                keywords = [kw]
        
        return JobFeatures(
            raw_text=job_text,
            job_title=job_title,
            company_name=company_name,
            required_qualifications=required_qualifications,
            preferred_qualifications=preferred_qualifications,
            responsibilities=responsibilities,
            keywords=keywords
        )
    
    except Exception as e:
        # Fallback to raw text on any error
        return JobFeatures(raw_text=job_text)
