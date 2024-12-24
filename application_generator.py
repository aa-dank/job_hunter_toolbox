import jinja2
import json
import os
import PyPDF2
import re
import markitdown

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


section_mapping_dict = {
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


class JobApplicationBuilder:
 
    def __init__(
        self, llm: LLMProvider, output_destination: str = "output_files",
        system_prompt: str = RESUME_WRITER_PERSONA
    ):
        self.system_prompt = system_prompt
        self.llm = llm
        self.md_converter = MarkItDown()
        self.output_destination = output_destination
        self.org = None
        self.job_title = None
        self.timestamp = datetime.now().strftime(r"%Y%m%d%H%M%S")
        self.resume_template_file = "basic_template.tex"

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

        if file_type == "jd":
            filename_base = "JD.json"
        elif file_type == "resume_json":
            filename_base = "resume.json"
        elif file_type == "resume":
            filename_base = "resume.tex"
        elif file_type == "cover_letter":
            filename_base = "cover_letter.txt"
        else:
            filename_base = ""

        filepath = os.path.join(doc_dir, filename_base)

        counter = 1
        base_filename, extension = os.path.splitext(filename_base)

        while os.path.exists(filepath):
            filename_base_with_counter = f"{base_filename}_{counter}{extension}"
            filepath = os.path.join(doc_dir, filename_base_with_counter)
            counter += 1

        return filepath
    

    def extract_job_content(self, job_content_path: str):
        job_content_path = Path(job_content_path)
        extension = job_content_path.suffix[1:]
        if extension == 'json':
            with open(job_content_path, 'r') as file:
                job_content_str = json.load(file)
        elif extension == 'md':
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
            partial_variables={"format_instructions": json_parser.get_format_instructions()}
            ).format(job_description=job_content_str)

        job_details = self.llm.get_response(prompt=prompt, need_json_output=True)
        self.org = job_details["company_name"]
        self.job_title = job_details["job_title"]

        jd_path = self.get_job_doc_path(file_type="jd")

        # Save the job details in a JSON file
        with open(jd_path, 'w') as file:
            json.dump(job_details, file, indent=4)
        
        print(f"Job Details JSON generated at: {jd_path}")

        return job_details, jd_path
    
    def resume_to_json(self, resume_text: str):
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
            partial_variables={"format_instructions": json_parser.get_format_instructions()}
            ).format(resume_text=resume_text)

        resume_json = self.llm.get_response(prompt=prompt, need_json_output=True)
        return resume_json


    def user_data_extraction(self, user_data_path: str):
        """
        Extracts user data from the given file path.

        Args:
            user_data_path (str): The path to the user data file.

        Returns:
            dict: The extracted user data in JSON format.
        """
        print("\nFetching user data...")


        extension = os.path.splitext(user_data_path)[1]

        if extension == ".json":
            # user_dat from json file
            with open(user_data_path, 'r') as file:
                user_data = json.load(file)
        elif extension == ".md":
            # user_data from markdown file
            with open(user_data_path, 'r') as file:
                user_data = file.read()
        else:
            try:
                # use the markitdown library to convert the file to markdown
                user_file_convertion = self.md_converter.convert(user_data_path)
                user_info_md = user_file_convertion.text_content
                if not user_info_md:
                    raise Exception("Empty Markdown Convertion")
                user_data = self.resume_to_json(user_info_md)
            except Exception as e:
                if e.__class__.__name__ == "UnsupportedFormatException":
                    raise UnsupportedFileFormatException(f"Unsupported file format: {extension}")
                else:
                    raise e
        
        return user_data
    
    def generate_resume_json(self, job_content, user_data):
        """
        Generates a resume in json format using the job content and user data.
        """
        try:
            print("\nGenerating Resume Details...")

            resume_details = dict()

            # Personal Information Section
            resume_details["personal"] = { 
                "name": user_data.get("name", ""),
                "phone": user_data.get("phone", ""),
                "email": user_data.get("email", ""),
                "github": user_data.get("media", {}).get("github", ""),
                "linkedin": user_data.get("media", {}).get("linkedin", ""),
                }

            # Other Sections
            for section in ['work_experience', 'projects', 'skill_section', 'education', 'certifications', 'achievements']:
                print(f"Processing Resume's {section.upper()} Section...")
                json_parser = JsonOutputParser(pydantic_object=section_mapping_dict[section]["schema"])
                
                prompt = PromptTemplate(
                    template=section_mapping_dict[section]["prompt"],
                    partial_variables={"format_instructions": json_parser.get_format_instructions()}
                    ).format(section_data = json.dumps(user_data[section]), job_description = json.dumps(job_content))

                response = self.llm.get_response(prompt=prompt, need_json_output=True)

                # Check for empty sections
                if isinstance(response, dict):
                    section_data = response.get(section)
                    if section_data:
                        if section == "skill_section":
                            resume_details[section] = [i for i in section_data if i.get('skills')]
                        else:
                            resume_details[section] = section_data

            resume_details['keywords'] = ', '.join(job_content['keywords'])
            
            resume_path = self.get_job_doc_path(file_type="resume_json")

            # Save the resume details in a JSON file
            with open(resume_path, 'w') as file:
                json.dump(resume_details, file, indent=4)

            return resume_details, resume_path
        
        except Exception as e:
            print(e)
            return None
    
    def resume_json_to_resume_tex(self, resume_details):
        """
        Turns either a json file or dictionary into a tex file using the resume template.
        :param resume_details: The resume details in json or dictionary format.
        """
        try:

            if type(resume_details) == str:
                with open(resume_details, 'r') as file:
                    resume_details = json.load(file)


            templates_path = os.path.join(os.getcwd(), 'templates')
            output_path = self.get_job_doc_path(file_type="resume")

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

            escaped_resume_dict = LatexToolBox.escape_for_latex(resume_details)

            resume_latex = LatexToolBox.tex_resume_from_jinja_template(jinja_env=latex_jinja_env,
                                                                       json_resume=escaped_resume_dict,
                                                                       tex_jinja_template=self.resume_template_file)

            
            # Save the resume in a tex file
            with open(output_path, 'w') as file:
                file.write(resume_latex)

            return resume_latex, output_path
        
        except Exception as e:
            print(e)
            return None, None

    def generate_cover_letter(self, job_details: dict, user_data: dict, need_pdf: bool = True):
        """
        Generates a tailored cover letter based on job details and user data.

        This method constructs a prompt using the COVER_LETTER_GENERATOR template and
        generates a cover letter using the language model (self.llm). It then saves the
        cover letter as a text file, and if requested, converts it into a PDF file.

        Args:
            job_details (dict): A dictionary containing the job description and requirements.
            user_data (dict): A dictionary containing the user's work information and personal details.
            need_pdf (bool, optional): If True, converts the generated cover letter to a PDF file.
                Defaults to True.

        Returns:
            tuple:
                - cover_letter (str or None): The generated cover letter text, or None if an exception occurs.
                - cover_letter_path (str or None): The file path to the saved cover letter (text or PDF),
                  or None if an exception occurs.

        Notes:
            - The method uses `get_job_doc_path` to determine where to save the cover letter.
            - If `need_pdf` is True, it converts the text file to a PDF using `text_to_pdf`.
            - Any exceptions are caught, printed, and the method returns (None, None).
        """
        try:
            prompt = PromptTemplate(
                template=COVER_LETTER_GENERATOR,
                input_variables=["my_work_information", "job_description"],
                ).format(job_description=job_details, my_work_information=user_data)

            cover_letter = self.llm.get_response(prompt=prompt)

            cover_letter_path = self.get_job_doc_path(file_type="cover_letter")
            # Save the cover letter in a text file
            with open(cover_letter_path, 'w') as file:
                file.write(cover_letter)

            print("Cover Letter generated at: ", cover_letter_path)
            if need_pdf:
                text_to_pdf(cover_letter, cover_letter_path.replace(".txt", ".pdf"))
                print("Cover Letter PDF generated at: ", cover_letter_path.replace(".txt", ".pdf"))
                return cover_letter, cover_letter_path.replace(".txt", ".pdf")

            return cover_letter, cover_letter_path     

        except Exception as e:
            print(e)
            return None, None


