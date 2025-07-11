import difflib
import jinja2
import json
import os
import PyPDF2
import re

from dataclasses import dataclass
from datetime import datetime
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from markitdown import MarkItDown
from pathlib import Path
from typing import Optional

# local imports
from generation_schemas import Achievements, Certifications, Educations, Experiences, JobDetails, Projects, ResumeSchema, SkillSections
from logger import setup_logger
from metrics import ScoringStrategy
from models import LLMProvider
from prompts.extraction_prompts import RESUME_DETAILS_EXTRACTOR, JOB_DETAILS_EXTRACTOR, COVER_LETTER_GENERATOR
from prompts.resume_section_prompts import EXPERIENCE, SKILLS, PROJECTS, EDUCATIONS, CERTIFICATIONS, ACHIEVEMENTS, RESUME_WRITER_PERSONA
from utils import LatexToolBox, text_to_pdf

# Initialize logger
logger = setup_logger(name="ApplicationGenerator", log_file="application_generator.log")

resume_section_prompt_map = {
    "work_experience": {"prompt":EXPERIENCE, "schema": Experiences},
    "skill_section": {"prompt":SKILLS, "schema": SkillSections},
    "projects": {"prompt":PROJECTS, "schema": Projects},
    "education": {"prompt":EDUCATIONS, "schema": Educations},
    "certifications": {"prompt":CERTIFICATIONS, "schema": Certifications},
    "achievements": {"prompt":ACHIEVEMENTS, "schema": Achievements},
}

class UnsupportedFileFormatException(Exception):
    """Custom exception for unsupported file formats."""
    pass

def extract_pdf_text(pdf_path: str): #kinda deprecated
    """ Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        str: The extracted text from the PDF file.
    """

    resume_text = ""

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text().split("\n")

            # Remove Unicode characters from each line
            cleaned_text = [re.sub(r'[^\x00-\x7F]+', '', line) for line in text]

            # Join the lines into a single string
            cleaned_text_string = '\n'.join(cleaned_text)
            resume_text += cleaned_text_string
        
        return resume_text

@dataclass
class JobApplicationBuild:
    """
    Dataclass for setting up the job application process.
    Usually initialized with at least the job and resume content paths.

    Attributes:
        job_content_path (str): File path to the job description or listing.
        user_details_content_path (str): File path to the user's resume or personal data.
        resume_tex_template_path (str): Path to the LaTeX template used for resume generation.
        resume_cls_path (str): Path to the LaTeX class file for the resume.
        output_destination (str): Directory for storing output files.
        timestamp (str): Timestamp used to distinguish output files.
        org (str): Name of the organization for which the job is applied.
        job_title (str): Parsed or assigned job title.
        parsed_job_details (dict): Extracted job details from the job content file.
        parsed_job_details_path (str): File path where job details JSON is saved.
        resume_details_dict (dict): Generated resume details in JSON format.
        resume_json_path (str): File path where the resume JSON is saved.
        resume_latex_text (str): Generated LaTeX content for the resume.
        resume_tex_path (str): File path where the resume LaTeX file is saved.
        parsed_user_data (dict): Extracted user data from the resume.
        cover_letter_text (str): Generated cover letter text.
        cover_letter_path (str): File path where the cover letter is saved.
    """
    # Required attributes
    job_content_path: str
    user_details_content_path: str

    # Optional attributes with default values
    resume_tex_template_path: str = "basic_template.tex"
    resume_cls_path: str = "resume.cls"
    output_destination: str = "output_files"
    timestamp: str = datetime.now().strftime(r"%Y%m%d%H%M%S")
    scoring_strategy: Optional[ScoringStrategy] = None #
    
    # Attributes to be populated by the builder methods during the application generation process
    org: str = None
    job_title: str = None
    parsed_job_details: dict = None
    parsed_job_details_path: str = None
    resume_details_dict: dict = None
    resume_json_path: str = None
    resume_latex_text: str = None
    resume_tex_path: str = None
    parsed_user_data: dict = None
    cover_letter_text: str = None
    cover_letter_path: str = None

    def get_job_doc_path(self, file_type: str = ""):
        """
        Generate the file path for job-related documents.

        Constructs a directory path in the format:
        `<self.output_destination>/<org>/<job_title>_<timestamp>`,
        where `org` is the cleaned company name and `job_title` is the cleaned job title.

        Parameters:
            file_type (str): Specifies the type of file to generate the path for.
                Accepted values are:
                - "jd": Returns the path for the job details JSON file named "JD.json".
                - "resume_json": Returns the path for the resume JSON file named "resume.json".
                - "resume": Returns the path for the resume JSON file named "resume.json".
                - "cover_letter": Returns the path for the cover letter text file named "cv.txt".
                - Any other value or empty string: Returns the base directory path without a filename.

        Returns:
            str: The full file path for the specified file type. If a file with the same name
            already exists, appends an incrementing counter to the filename to ensure uniqueness.
        """
        def clean_string(text: str):
            text = text.title().replace(" ", "").strip()
            text = re.sub(r"[^a-zA-Z0-9]+", "", text)
            return text

        company_name = clean_string(self.org)
        job_title = clean_string(self.job_title)[:15]
        timestamp = self.timestamp

        doc_dir = os.path.join(self.output_destination, company_name, f"{job_title}_{timestamp}")
        os.makedirs(doc_dir, exist_ok=True)


        if not file_type or (file_type not in ["jd", "resume_json", "resume", "cover_letter"]):
            return doc_dir
        
        elif file_type == "jd":
            filename_base = "JD.json"
        elif file_type == "resume_json":
            filename_base = "resume.json"
        elif file_type == "resume":
            filename_base = "resume.tex"
        elif file_type == "cover_letter":
            filename_base = "cover_letter.txt"

        filepath = os.path.join(doc_dir, filename_base)

        counter = 1
        base_filename, extension = os.path.splitext(filename_base)

        while os.path.exists(filepath):
            filename_base_with_counter = f"{base_filename}_{counter}{extension}"
            filepath = os.path.join(doc_dir, filename_base_with_counter)
            counter += 1

        return filepath


class JobApplicationBuilder:
    """
    A builder class that leverages an LLM to generate various parts of a job application.
    All methods in this class receive a JobApplicationBuild object as a parameter and modify it in place.
    Each method returns the updated build object.
    """
 
    def __init__(self, llm: LLMProvider):
        """
        Initializes the JobApplicationBuilder class.
        """
        self.llm = llm
        self.md_converter = MarkItDown()
        self.similarity_scoring_model: str = "all-MiniLM-L6-v2"
        

    def extract_job_content(self, build: JobApplicationBuild):
        logger.info("Extracting job content from the provided file.")
        try:
            if not build.job_content_path:
                raise ValueError("Job content path is missing from the JobApplicationBuild object.")
            
            if build.parsed_job_details:
                return build
            
            job_content_path = build.job_content_path
            job_content_path = Path(job_content_path)
            extension = job_content_path.suffix[1:]
            if (extension == 'json'):
                with open(job_content_path, 'r') as file:
                    job_content_str = json.load(file)
            elif (extension == 'md'):
                with open(job_content_path, 'r') as file:
                    job_content_str = file.read()
            else:
                try:
                    # use the markitdown library to convert the file to markdown
                    convertion_result = self.md_converter.convert(str(job_content_path))
                    job_content_str = convertion_result.text_content
                    if not job_content_str:
                        raise Exception("Empty Markdown Convertion")
                except Exception as e:
                    
                    # if markitdown._markitdown.UnsupportedFormatException...
                    if e.__class__.__name__ == "UnsupportedFormatException":
                        raise UnsupportedFileFormatException(f"Unsupported file format: {extension}")
                    else:
                        raise e


            json_parser = JsonOutputParser(pydantic_object=JobDetails)
            
            prompt = PromptTemplate(
                template=JOB_DETAILS_EXTRACTOR,
                input_variables=["job_description"],
                partial_variables={"format_instructions": json_parser.get_format_instructions()},
                template_format="jinja2",
                validate_template=False
                ).format(job_description=job_content_str)

            job_details = self.llm.get_response(prompt=prompt, need_json_output=True)
            build.org = job_details["company_name"]
            build.job_title = job_details["job_title"]
            build.parsed_job_details = job_details

            structured_job_details_filepath = build.get_job_doc_path(file_type="jd")

            # Save the job details in a JSON file
            with open(structured_job_details_filepath, 'w') as file:
                json.dump(job_details, file, indent=4)
            
            logger.info(f"Job Details JSON generated at: {structured_job_details_filepath}")

            build.parsed_job_details = job_details
            build.parsed_job_details_path = structured_job_details_filepath
            return build
        except Exception as e:
            logger.error(f"Error extracting job content: {e}", exc_info=True)
            raise
    
    def _resume_to_json(self, resume_text: str):
        """
        Converts a resume in PDF format to JSON format.

        Args:
            resume_text (str): The text extracted from the resume file.

        Returns:
            dict: The resume data in JSON format.
        """

        json_parser = JsonOutputParser(pydantic_object=ResumeSchema)

        prompt = PromptTemplate(
            template=RESUME_DETAILS_EXTRACTOR,
            input_variables=["resume_text"],
            partial_variables={"format_instructions": json_parser.get_format_instructions()},
            template_format="jinja2",
            validate_template=False
        ).format(resume_text=resume_text)

        resume_json = self.llm.get_response(prompt=prompt, need_json_output=True)
        return resume_json

    def user_data_extraction(self, build: JobApplicationBuild):
        logger.info("Extracting user data from the provided resume file.")
        try:
            if not build.user_details_content_path:
                raise ValueError("Resume content path is missing from the JobApplicationBuild object.")

            if build.parsed_user_data:
                return build

            extension = os.path.splitext(build.user_details_content_path)[1]

            if extension == ".json":
                # user_dat from json file
                with open(build.user_details_content_path, 'r') as file:
                    user_data = json.load(file)
            elif extension == ".md":
                # user_data from markdown file
                with open(build.user_details_content_path, 'r') as file:
                    user_data = file.read()
            else:
                try:
                    # use the markitdown library to convert the file to markdown
                    user_file_convertion = self.md_converter.convert(build.user_details_content_path)
                    user_info_md = user_file_convertion.text_content
                    if not user_info_md:
                        raise Exception("Empty Markdown Convertion")
                    user_data = self._resume_to_json(user_info_md)
                except Exception as e:
                    if e.__class__.__name__ == "UnsupportedFormatException":
                        raise UnsupportedFileFormatException(f"Unsupported file format: {extension}")
                    else:
                        raise e
            
            build.parsed_user_data = user_data
            logger.info("User data extracted successfully.")
            return build
        except Exception as e:
            logger.error(f"Error extracting user data: {e}", exc_info=True)
            raise

    def score_description_points(self, build: JobApplicationBuild, model: str = None):
        """
        Score individual description points in experiences and projects based on relevance to the job.
        
        Args:
            build: JobApplicationBuild object with parsed job and user data
            
        Returns:
            JobApplicationBuild: Updated build with scored description points
        """
        
        def score_points_for_section(items, description_key, job_text, model):
            """
            Scores each point in the description list of each item in a section.
            Modifies the items in-place by adding a 'scored_description' field.
            """
            from metrics import sentence_transformer_similarity

            for item in items:
                points = []
                for point in item.get(description_key, []):
                    if not point.strip():
                        continue
                    point_score = sentence_transformer_similarity(
                        document1=point,
                        document2=job_text,
                        model=model
                    )
                    points.append({
                        "text": point,
                        "score": point_score
                    })
        
        if not build.parsed_job_details or not build.parsed_user_data:
            logger.warning("Cannot score content: missing job details or user data")
            return build
        
        if not build.scoring_strategy:
            logger.info("Description point scoring disabled, skipping")
            return build
        
        from metrics import sentence_transformer_similarity

        try:
            job_text = " ".join([
                build.parsed_job_details.get("job_title", ""),
                build.parsed_job_details.get("job_purpose", ""),
                " ".join(build.parsed_job_details.get("keywords", [])),
                " ".join(build.parsed_job_details.get("job_duties_and_responsibilities", [])),
                " ".join(build.parsed_job_details.get("required_qualifications", [])),
                " ".join(build.parsed_job_details.get("preferred_qualifications", []))
            ])

            # Score experiences
            experiences = build.parsed_user_data.get("experiences", {}).get("work_experience", [])
            score_points_for_section(experiences, "description", job_text, model)

            # Score projects
            projects = build.parsed_user_data.get("projects", {}).get("projects", [])
            score_points_for_section(projects, "description", job_text, model)

            # To add more sections in the future, just call:
            # score_points_for_section(other_items, "description", job_text, model)

            return build

        except Exception as e:
            logger.error(f"Error scoring description points: {e}", exc_info=True)
            build.scoring_strategy = None
            return build


    def generate_resume_json(self, build: JobApplicationBuild):
        logger.info("Generating resume JSON from job details and user data.")
        try:
            if not build.parsed_user_data:
                raise ValueError("User data is missing from the JobApplicationBuild object.")
            
            if not build.parsed_job_details:
                raise ValueError("Job details are missing from the JobApplicationBuild object.")

            if build.resume_details_dict and build.resume_json_path:
                return build
            
            if build.scoring_strategy:
                build = self.score_description_points(build)
            
            resume_details = dict()
            
            # Personal Information Section
            # build.parsed_user_data is a dict representation of ResumeSchema after _resume_to_json
            parsed_personal_data_obj = build.parsed_user_data.get("personal") # This will be a dict like {"name": "...", "github": HttpUrl(...)}
            if parsed_personal_data_obj:
                resume_details["personal"] = {
                    "name": parsed_personal_data_obj.get("name"),
                    "phone": parsed_personal_data_obj.get("phone"),
                    "email": parsed_personal_data_obj.get("email"),
                    "github": str(parsed_personal_data_obj.get("github")) if parsed_personal_data_obj.get("github") else None,
                    "linkedin": str(parsed_personal_data_obj.get("linkedin")) if parsed_personal_data_obj.get("linkedin") else None,
                }
            else:
                resume_details["personal"] = {} 
                logger.warning("Personal details not found in parsed_user_data after extraction.")

            # Summary Section (direct copy from parsed_user_data)
            resume_details["summary"] = build.parsed_user_data.get("summary", None)

            # Other Sections that require LLM processing
            # section_map: key is for resume_details and build.parsed_user_data (ResumeSchema keys)
            # value is for resume_section_prompt_map and the key within the LLM's JSON response for that section
            section_map = {
                "experiences": "work_experience",
                "projects": "projects",
                "skills": "skill_section",
                "educations": "education",
                "certifications": "certifications",
                "achievements": "achievements"
            }

            for resume_key, map_key in section_map.items():
                logger.info(f"Processing Resume's {resume_key.upper()} Section...")
                
                prompt_info = resume_section_prompt_map.get(map_key)
                if not prompt_info:
                    logger.warning(f"No prompt info found for {map_key} (from {resume_key}). Initializing section as empty.")
                    resume_details[resume_key] = {map_key: []} # Default structure e.g., {"work_experience": []}
                    continue

                json_parser = JsonOutputParser(pydantic_object=prompt_info["schema"])
                
                # Extract the list of items for the current section from build.parsed_user_data
                # e.g., for resume_key="experiences", user_section_container = build.parsed_user_data.get("experiences")
                # which would be a dict like {"work_experience": [...]}. We need the list part.
                user_section_container = build.parsed_user_data.get(resume_key) 
                actual_user_data_list = []
                if isinstance(user_section_container, dict):
                    actual_user_data_list = user_section_container.get(map_key, [])
                elif resume_key == "achievements" and isinstance(user_section_container, list): # achievements might be a direct list if schema was simpler before
                    actual_user_data_list = user_section_container
                
                scoring_instructions = ""
                if build.scoring_strategy and resume_key in ["experiences", "projects"]:
                    import copy
                    actual_user_data_list = copy.deepcopy(actual_user_data_list)

                    # Replace the descriptions with scored descriptions for the LLM
                    for item in actual_user_data_list:
                        if "scored_description" in item:
                            # Add the scores to the description text for the LLM
                            item["description"] = [f"{p['text']} [relevance: {p['score']:.2f}]" 
                                                for p in item["scored_description"]]
                    
                    # Add instructions about scores
                    scoring_instructions = (
                        f"\nIMPORTANT: Description points include relevance scores in format [relevance: X.XX]. "
                        f"Higher scores indicate stronger relevance to the job requirements. "
                        f"Prioritize points with higher scores when creating the resume, and make sure to REMOVE "
                        f"the relevance notation in your final output."
                    )

                prompt = PromptTemplate(
                    template=prompt_info["prompt"] + scoring_instructions,
                    partial_variables={"format_instructions": json_parser.get_format_instructions(),
                                       "includes_relevance_scores": build.scoring_strategy is not None},
                    template_format="jinja2",
                    validate_template=False
                ).format(section_data=json.dumps(actual_user_data_list), 
                         job_description=json.dumps(build.parsed_job_details))

                response = self.llm.get_response(prompt=prompt, need_json_output=True) # Expected: {"map_key": [...]}

                if isinstance(response, dict):
                    processed_section_data_list = response.get(map_key) 
                    if processed_section_data_list is not None: # Check for None, empty list is valid data
                        if resume_key == "skills": 
                            valid_skill_groups = []
                            if isinstance(processed_section_data_list, list):
                                valid_skill_groups = [i for i in processed_section_data_list if isinstance(i, dict) and i.get('skills')]
                            resume_details[resume_key] = {map_key: valid_skill_groups}
                        else: 
                            # Ensures the data for map_key is a list
                            resume_details[resume_key] = {map_key: processed_section_data_list if isinstance(processed_section_data_list, list) else []}
                    else:
                        logger.warning(f"LLM response for {map_key} (from {resume_key}) did not contain the key '{map_key}'. Initializing as empty list for this part.")
                        resume_details[resume_key] = {map_key: []} 
                else:
                    logger.warning(f"LLM response for {map_key} (from {resume_key}) was not a dict. Initializing as empty list for this part.")
                    resume_details[resume_key] = {map_key: []}
            
            # Keywords section (not part of ResumeSchema but used by template)
            if build.parsed_job_details and 'keywords' in build.parsed_job_details:
                resume_details['keywords'] = ', '.join(build.parsed_job_details['keywords'])
            else:
                resume_details['keywords'] = '' 
            
            resume_json_filepath = build.get_job_doc_path(file_type="resume_json")

            with open(resume_json_filepath, 'w') as file:
                json.dump(resume_details, file, indent=4)

            build.resume_details_dict = resume_details
            build.resume_json_path = resume_json_filepath
            logger.info(f"Resume JSON saved at: {resume_json_filepath}")
            return build
        except Exception as e:
            logger.error(f"Error generating resume JSON: {e}", exc_info=True)
            raise
    
    def resume_json_to_resume_tex(self, build: JobApplicationBuild):
        logger.info("Converting resume JSON to LaTeX format.")
        try:
            if not build.resume_details_dict:
                raise ValueError("Resume details are missing from the JobApplicationBuild object. Cannot generate resume without resume details.")
            
            if build.resume_latex_text and build.resume_tex_path:
                return build
            
            templates_path = os.path.join(os.getcwd(), 'templates')
            output_path = build.get_job_doc_path(file_type="resume")
            
            latex_jinja_env = jinja2.Environment(
                block_start_string="\BLOCK{",
                block_end_string="}",
                variable_start_string="\VAR{",
                variable_end_string="}",
                comment_start_string="\#{",
                comment_end_string="}",
                line_statement_prefix="%-",
                line_comment_prefix="%#",
                trim_blocks=True,
                autoescape=False,
                loader=jinja2.FileSystemLoader(templates_path),
            )

            escaped_resume_dict = LatexToolBox.escape_for_latex(build.resume_details_dict)

            flattened_resume = {
                'personal': escaped_resume_dict.get('personal', {}),
                'summary': escaped_resume_dict.get('summary', None),
                'keywords': escaped_resume_dict.get('keywords', ''),
                
                # Extract nested lists with safe fallbacks
                'work_experience': escaped_resume_dict.get('experiences', {}).get('work_experience', []),
                'education': escaped_resume_dict.get('educations', {}).get('education', []),
                'projects': escaped_resume_dict.get('projects', {}).get('projects', []),
                'skill_section': escaped_resume_dict.get('skills', {}).get('skill_section', [])
            }

            # Add optional sections only if they exist
            if 'achievements' in escaped_resume_dict and isinstance(escaped_resume_dict['achievements'], dict):
                flattened_resume['achievements'] = escaped_resume_dict.get('achievements', {}).get('achievements', [])
            if 'certifications' in escaped_resume_dict and isinstance(escaped_resume_dict['certifications'], dict):
                flattened_resume['certifications'] = escaped_resume_dict['certifications'].get('certifications', [])



            resume_latex = LatexToolBox.tex_resume_from_jinja_template(jinja_env=latex_jinja_env,
                                                                       json_resume=flattened_resume,
                                                                       tex_jinja_template=build.resume_tex_template_path)

            
            # Save the resume in a tex file
            with open(output_path, 'w') as file:
                file.write(resume_latex)

            build.resume_latex_text = resume_latex
            build.resume_tex_path = output_path
            logger.info(f"Resume LaTeX file saved at: {output_path}")
            return build
        except Exception as e:
            logger.error(f"Error converting resume JSON to LaTeX: {e}", exc_info=True)
            raise

    def generate_cover_letter(self, build: JobApplicationBuild, custom_instructions: str = "", need_pdf: bool = True):
        logger.info("Generating cover letter.")
        try:
            if not build.parsed_job_details or not build.parsed_user_data:
                if not build.parsed_job_details:
                    raise ValueError("Job details are missing from the JobApplicationBuild object.")
                else:
                    raise ValueError("User data is missing from the JobApplicationBuild object.")
                
            if build.cover_letter_text and build.cover_letter_path:
                return build

            prompt = PromptTemplate(
                template=COVER_LETTER_GENERATOR,
                template_format="jinja2",
                input_variables=["job_description", "my_work_information", "application_specific_instructions"], 
                validate_template=False
            ).format(
                job_description=build.parsed_job_details,
                my_work_information=build.parsed_user_data,
                application_specific_instructions=custom_instructions or ""
            )

            cover_letter = self.llm.get_response(prompt=prompt)

            cover_letter_path = build.get_job_doc_path(file_type="cover_letter")
            # Save the cover letter in a text file
            with open(cover_letter_path, 'w') as file:
                file.write(cover_letter)

            logger.info(f"Cover letter generated at: {cover_letter_path}")
            if need_pdf:
                text_to_pdf(cover_letter, cover_letter_path.replace(".txt", ".pdf"))
                logger.info(f"Cover letter PDF generated at: {cover_letter_path.replace('.txt', '.pdf')}")
                build.cover_letter_text = cover_letter
                build.cover_letter_path = cover_letter_path.replace(".txt", ".pdf")
                return build

            build.cover_letter_text = cover_letter
            build.cover_letter_path = cover_letter_path
            return build     
        except Exception as e:
            logger.error(f"Error generating cover letter: {e}", exc_info=True)
            raise

    def validate_resume_json(self, build: JobApplicationBuild, print_viz_changes: bool = False):
        """
        Validates the generated resume JSON against the original user data to detect and correct hallucinations.

        Modifies:
            - build.resume_details_dict with the validated resume details.
            - Saves the corrected JSON to the same file path.

        Returns:
            JobApplicationBuild: The updated build object.
        """
        try:
            if not build.resume_details_dict:
                raise ValueError("Resume details are missing from the JobApplicationBuild object. Cannot validate without resume details.")

            if not build.parsed_user_data:
                raise ValueError("User data is missing from the JobApplicationBuild object. Cannot validate without user data.")

            from prompts.validation_prompts import RESUME_VALIDATION_PROMPT
            from langchain_core.output_parsers import JsonOutputParser

            json_parser = JsonOutputParser(pydantic_object=ResumeSchema)

            prompt = PromptTemplate(
                template=RESUME_VALIDATION_PROMPT,
                input_variables=["original_user_data", "generated_resume_json"],
                partial_variables={"format_instructions": json_parser.get_format_instructions()},
                template_format="jinja2",
                validate_template=False
            ).format(
                original_user_data=json.dumps(build.parsed_user_data),
                generated_resume_json=json.dumps(build.resume_details_dict)
            )

            validated_resume = self.llm.get_response(prompt=prompt, need_json_output=True)

            # Save the validated resume JSON to the same file path
            with open(build.resume_json_path, 'w') as file:
                json.dump(validated_resume, file, indent=4)

            build.resume_details_dict = validated_resume

            if print_viz_changes:
                original = json.dumps(build.parsed_user_data, indent=4).splitlines()
                validated = json.dumps(validated_resume, indent=4).splitlines()
                diff = difflib.unified_diff(
                    original,
                    validated,
                    fromfile="original_user_data",
                    tofile="validated_resume",
                    lineterm=""
                )
                print("\n".join(diff))
                logger.info("Validation diff printed to terminal.")

            return build

        except Exception as e:
            logger.error(f"Error validating resume JSON: {e}", exc_info=True)
            return None

    def cleanup_files(self, build: JobApplicationBuild):
        logger.info("Cleaning up temporary files.")
        try:
            import shutil
            if build.job_content_path:
                if not os.path.exists(build.job_content_path):
                    logger.warning(f"Job content path {build.job_content_path} does not exist.")
                else:
                    shutil.move(build.job_content_path, build.get_job_doc_path())
            logger.info("Temporary files cleaned up successfully.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            raise



