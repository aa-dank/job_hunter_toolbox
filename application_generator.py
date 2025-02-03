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
from models import LLMProvider
from generation_schemas import Achievements, Certifications, Educations, Experiences, JobDetails, Projects, ResumeSchema, SkillSections
from pathlib import Path
from prompts.extraction_prompts import RESUME_DETAILS_EXTRACTOR, JOB_DETAILS_EXTRACTOR, COVER_LETTER_GENERATOR
from prompts.resume_section_prompts import EXPERIENCE, SKILLS, PROJECTS, EDUCATIONS, CERTIFICATIONS, ACHIEVEMENTS, RESUME_WRITER_PERSONA
from utils import LatexToolBox, text_to_pdf


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

    def extract_job_content(self, build: JobApplicationBuild):
        """
        Extracts job content from the provided file and updates the build object.

        Modifies:
            - build.structured_job_details with the extracted job details.
            - build.structured_job_details_path with the JSON file path.
        
        Returns:
            JobApplicationBuild: The updated build object.
        """
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
        
        print(f"Job Details JSON generated at: {structured_job_details_filepath}")

        build.parsed_job_details = job_details
        build.parsed_job_details_path = structured_job_details_filepath
        return build
    
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
        """
        Extracts user resume data from the provided file and updates the build object.

        Modifies:
            - build.structured_user_data with the extracted user data.
        
        Returns:
            JobApplicationBuild: The updated build object.
        """
        print("\nFetching user data...")
        
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
        return build
    
    def generate_resume_json(self, build: JobApplicationBuild):
        """
        Generates a resume JSON from job details and user data, updating the build object.

        Modifies:
            - build.resume_details_dict with the generated resume details.
            - build.resume_json_path with the JSON file path.
        
        Returns:
            JobApplicationBuild: The updated build object.
        """
        try:
            print("\nGenerating Resume Details...")

            if not build.parsed_user_data:
                raise ValueError("User data is missing from the JobApplicationBuild object.")
            
            if not build.parsed_job_details:
                raise ValueError("Job details are missing from the JobApplicationBuild object.")

            if build.resume_details_dict and build.resume_json_path:
                return build
            
            resume_details = dict()
            
            # Personal Information Section
            resume_details["personal"] = { 
                "name": build.parsed_user_data.get("name", ""),
                "phone": build.parsed_user_data.get("phone", ""),
                "email": build.parsed_user_data.get("email", ""),
                "github": build.parsed_user_data.get("media", {}).get("github", ""),
                "linkedin": build.parsed_user_data.get("media", {}).get("linkedin", ""),
                }

            # Other Sections
            for section in ['work_experience', 'projects', 'skill_section', 'education', 'certifications', 'achievements']:
                print(f"Processing Resume's {section.upper()} Section...")
                json_parser = JsonOutputParser(pydantic_object=resume_section_prompt_map[section]["schema"])
                
                prompt = PromptTemplate(
                    template=resume_section_prompt_map[section]["prompt"],
                    partial_variables={"format_instructions": json_parser.get_format_instructions()},
                    template_format="jinja2",
                    validate_template=False
                ).format(section_data = json.dumps(build.parsed_user_data[section]),
                         job_description = json.dumps(build.parsed_job_details))

                response = self.llm.get_response(prompt=prompt, need_json_output=True)

                # Check for empty sections
                if isinstance(response, dict):
                    section_data = response.get(section)
                    if section_data:
                        if section == "skill_section":
                            resume_details[section] = [i for i in section_data if i.get('skills')]
                        else:
                            resume_details[section] = section_data

            resume_details['keywords'] = ', '.join(build.parsed_job_details['keywords'])
            
            resume_json_filepath = build.get_job_doc_path(file_type="resume_json")

            # Save the resume details in a JSON file
            with open(resume_json_filepath, 'w') as file:
                json.dump(resume_details, file, indent=4)

            build.resume_details_dict = resume_details
            build.resume_json_path = resume_json_filepath
            return build
        
        except Exception as e:
            print(e)
            return None
    
    def resume_json_to_resume_tex(self, build: JobApplicationBuild):
        """
        Generates a LaTeX file from the resume JSON and updates the build object.

        Modifies:
            - build.resume_latex with the LaTeX content.
            - build.resume_tex_path with the path to the generated .tex file.
        
        Returns:
            JobApplicationBuild: The updated build object.
        """
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

            resume_latex = LatexToolBox.tex_resume_from_jinja_template(jinja_env=latex_jinja_env,
                                                                       json_resume=escaped_resume_dict,
                                                                       tex_jinja_template=build.resume_tex_template_path)

            
            # Save the resume in a tex file
            with open(output_path, 'w') as file:
                file.write(resume_latex)

            build.resume_latex_text = resume_latex
            build.resume_tex_path = output_path
            return build
        
        except Exception as e:
            print(e)
            return None, None

    def generate_cover_letter(self, build: JobApplicationBuild, custom_instructions: str = "", need_pdf: bool = True):
        """
        Generates a tailored cover letter by leveraging the job details and user data,
        then updates the build object with the cover letter information.

        Modifies:
            - build.cover_letter_text with the cover letter text.
            - build.cover_letter_path with the path to the saved cover letter (PDF or text).
        
        Returns:
            JobApplicationBuild: The updated build object.
        """
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

            print("Cover Letter generated at: ", cover_letter_path)
            if need_pdf:
                text_to_pdf(cover_letter, cover_letter_path.replace(".txt", ".pdf"))
                print("Cover Letter PDF generated at: ", cover_letter_path.replace(".txt", ".pdf"))
                build.cover_letter_text = cover_letter
                build.cover_letter_path = cover_letter_path.replace(".txt", ".pdf")
                return build

            build.cover_letter_text = cover_letter
            build.cover_letter_path = cover_letter_path
            return build     

        except Exception as e:
            print(e)
            return None, None

    def cleanup_files(self, build: JobApplicationBuild):
        import shutil
        if build.job_content_path:
            shutil.move(build.job_content_path, build.get_job_doc_path())



