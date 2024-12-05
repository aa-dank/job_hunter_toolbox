import jinja2
import json
import os
import PyPDF2
import re
import shutil
import subprocess
import validators

from bs4 import BeautifulSoup
from fpdf import FPDF
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from models import LLMProvider
from generation_schemas import Achievements, Certifications, Educations, Experiences, JobDetails, Projects, ResumeSchema, SkillSections
from zlm import AutoApplyModel
from prompts.extraction_prompts import RESUME_DETAILS_EXTRACTOR, JOB_DETAILS_EXTRACTOR, CV_GENERATOR
from prompts.resume_section_prompts import EXPERIENCE, SKILLS, PROJECTS, EDUCATIONS, CERTIFICATIONS, ACHIEVEMENTS, RESUME_WRITER_PERSONA



llm_mapping_dict = {
    'GPT': {
        "api_env": "OPENAI_API_KEY",
        "model": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-1106-preview", "gpt-3.5-turbo"], 
    },
    'Gemini': {
        "api_env": "GEMINI_API_KEY",
        "model": ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-1.5-pro-latest", "gemini-1.5-pro-exp-0801"], # "gemini-1.0-pro", "gemini-1.0-pro-latest"
    }
}

section_mapping_dict = {
    "work_experience": {"prompt":EXPERIENCE, "schema": Experiences},
    "skill_section": {"prompt":SKILLS, "schema": SkillSections},
    "projects": {"prompt":PROJECTS, "schema": Projects},
    "education": {"prompt":EDUCATIONS, "schema": Educations},
    "certifications": {"prompt":CERTIFICATIONS, "schema": Certifications},
    "achievements": {"prompt":ACHIEVEMENTS, "schema": Achievements},
}

def extract_pdf_text(pdf_path: str):
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

def escape_for_latex(data):
    if isinstance(data, dict):
        new_data = {}
        for key in data.keys():
            new_data[key] = escape_for_latex(data[key])
        return new_data
    elif isinstance(data, list):
        return [escape_for_latex(item) for item in data]
    elif isinstance(data, str):
        # Adapted from https://stackoverflow.com/q/16259923
        latex_special_chars = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\^{}",
            "\\": r"\textbackslash{}",
            "\n": "\\newline%\n",
            "-": r"{-}",
            "\xA0": "~",  # Non-breaking space
            "[": r"{[}",
            "]": r"{]}",
        }
        return "".join([latex_special_chars.get(c, c) for c in data])

    return data

def use_template(jinja_env, json_resume):
    try:
        resume_template = jinja_env.get_template(f"resume.tex.jinja")
        resume = resume_template.render(json_resume)

        return resume
    except Exception as e:
        print(e)
        return None
    
def job_doc_name(job_details: dict, output_dir: str = "output", type: str = ""):
    def clean_string(text: str):
        text = text.title().replace(" ", "").strip()
        text = re.sub(r"[^a-zA-Z0-9]+", "", text)
        return text
    
    company_name = clean_string(job_details["company_name"])
    job_title = clean_string(job_details["job_title"])[:15]
    doc_name = "_".join([company_name, job_title])
    doc_dir = os.path.join(output_dir, company_name)
    os.makedirs(doc_dir, exist_ok=True)

    if type == "jd":
        return os.path.join(doc_dir, f"{doc_name}_JD.json")
    elif type == "resume":
        return os.path.join(doc_dir, f"{doc_name}_resume.json")
    elif type == "cv":
        return os.path.join(doc_dir, f"{doc_name}_cv.txt")
    else:
        return os.path.join(doc_dir, f"{doc_name}_")

    
def text_to_pdf(text: str, file_path: str):
    """Converts the given text to a PDF and saves it to the specified file path.

    Args:
        text (str): The text to be converted to PDF.
        file_path (str): The file path where the PDF will be saved.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    # Encode the text explicitly using 'latin-1' encoding
    encoded_text = text.encode('utf-8').decode('latin-1')
    pdf.multi_cell(0, 5, txt=encoded_text)
    pdf.output(file_path)

class JobApplicationBuilder:
 
    def __init__(
        self, llm: LLMProvider, output_destination: str,
        system_prompt: str = RESUME_WRITER_PERSONA
    ):
        self.system_prompt = system_prompt
        self.llm = llm
        self.output_destination = output_destination
        # Remove get_llm_instance and related initialization
        # ...existing code...

    # Update methods to use self.llm directly
    def extract_job_content(self, job_filepath: str):
        extension = job_filepath.split('.')[-1]
        if not extension in ['pdf', 'json', 'txt', 'html', 'md']:
            raise ValueError(f"Unsupported file format: {extension}\nfile_path: {job_filepath}")

        if extension == 'pdf':
            job_content_str = extract_pdf_text(job_filepath)
        elif extension == 'json':
            with open(job_filepath, 'r') as file:
                job_content_str = json.load(file)
        elif extension == 'txt':
            with open(job_filepath, 'r') as file:
                job_content_str = file.read()
        elif extension == 'md':
            with open(job_filepath, 'r') as file:
                job_content_str = file.read()
        
        elif extension == 'html':
            #TODO: Implement the processing of HTML files
            #job_content_str = self.process_job_html(job_filepath)
            raise ValueError(f"Unsupported file format: {extension}\nfile_path: {job_filepath}")

        json_parser = JsonOutputParser(pydantic_object=JobDetails)
        
        prompt = PromptTemplate(
            template=JOB_DETAILS_EXTRACTOR,
            input_variables=["job_description"],
            partial_variables={"format_instructions": json_parser.get_format_instructions()}
            ).format(job_description=job_content_str)

        job_details = self.llm.get_response(prompt=prompt, need_json_output=True)

        jd_path = job_doc_name(job_details, self.output_destination, "jd")

        # Save the job details in a JSON file
        with open(jd_path, 'w') as file:
            json.dump(job_details, file, indent=4)
        
        print(f"Job Details JSON generated at: {jd_path}")

        return job_details, jd_path
    
    def resume_to_json(self, pdf_path):
        """
        Converts a resume in PDF format to JSON format.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            dict: The resume data in JSON format.
        """
        resume_text = extract_pdf_text(pdf_path)

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

        if extension == ".pdf":
            user_data = self.resume_to_json(user_data_path)
        elif extension == ".json":
            # user_dat from json file
            with open(user_data_path, 'r') as file:
                user_data = json.load(file)
        else:
            raise Exception("Invalid file format. Please provide a PDF, JSON file or url.")
        
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
                "name": user_data["name"], 
                "phone": user_data["phone"], 
                "email": user_data["email"],
                "github": user_data["media"]["github"], 
                "linkedin": user_data["media"]["linkedin"]
                }

            # Other Sections
            for section in ['work_experience', 'projects', 'skill_section', 'education', 'certifications', 'achievements']:
                section_log = f"Processing Resume's {section.upper()} Section..."
                json_parser = JsonOutputParser(pydantic_object=section_mapping_dict[section]["schema"])
                
                prompt = PromptTemplate(
                    template=section_mapping_dict[section]["prompt"],
                    partial_variables={"format_instructions": json_parser.get_format_instructions()}
                    ).format(section_data = json.dumps(user_data[section]), job_description = json.dumps(job_content))

                response = self.llm.get_response(prompt=prompt, need_json_output=True)

                # Check for empty sections
                if response is not None and isinstance(response, dict):
                    if section in response:
                        if response[section]:
                            if section == "skill_section":
                                resume_details[section] = [i for i in response['skill_section'] if len(i['skills'])]
                            else:
                                resume_details[section] = response[section]

            resume_details['keywords'] = ', '.join(job_content['keywords'])
            
            resume_path = job_doc_name(job_content, self.output_destination, "resume")

            # Save the resume details in a JSON file
            with open(resume_path, 'w') as file:
                json.dump(resume_details, file, indent=4)

            return resume_details, resume_path
        
        except Exception as e:
            print(e)
            return None
    
    def resume_json_to_resume_tex(self, resume_details, tex_filename):
        """
        Turns either a json file or dictionary into a tex file using the resume template.
        :param resume_details: The resume details in json or dictionary format.
        """
        try:

            if type(resume_details) == str:
                with open(resume_details, 'r') as file:
                    resume_details = json.load(file)


            templates_path = os.path.join(os.getcwd(), 'templates')
            output_path = os.path.join(os.getcwd(), 'output_files')

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

            escaped_resume_dict = escape_for_latex(resume_details)

            resume_latex = use_template(latex_jinja_env, escaped_resume_dict)

            tex_temp_path = os.path.join(os.path.realpath(output_path), tex_filename)
            
            # Save the resume in a tex file
            with open(tex_temp_path, 'w') as file:
                file.write(resume_latex)

            return resume_latex, tex_temp_path
        
        except Exception as e:
            print(e)
            return None, None
        
    @staticmethod
    def save_latex_as_pdf(tex_file_path: str, dst_path: str):
        try:
            # Call pdflatex to convert LaTeX to PDF
            prev_loc = os.getcwd()
            os.chdir(os.path.dirname(tex_file_path))
            
            try:
                # Check if pdflatex is available
                if shutil.which("pdflatex") is None:
                    print("Pdflatex is not installed or not available in the system PATH.")
                    return None

                result = subprocess.run(
                    ["pdflatex", tex_file_path, "&>/dev/null"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                print("Pdflatex failed to convert tex file to pdf.")
                print(e)


            os.chdir(prev_loc)
            resulted_pdf_path = tex_file_path.replace(".tex", ".pdf")
            dst_tex_path = dst_path.replace(".pdf", ".tex")

            os.rename(resulted_pdf_path, dst_path)
            os.rename(tex_file_path, dst_tex_path)

            if result.returncode != 0:
                print("Exit-code not 0, check result!")
            try:
                pass
                # open_file(dst_path)
            except Exception as e:
                print("Unable to open the PDF file.")

            filename_without_ext = os.path.basename(tex_file_path).split(".")[0]
            unnessary_files = [
                file
                for file in os.listdir(os.path.dirname(os.path.realpath(tex_file_path)))
                if file.startswith(filename_without_ext)
            ]

            for file in unnessary_files:
                file_path = os.path.join(os.path.dirname(tex_file_path), file)
                if os.path.exists(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(e)
            return None
        
    def cover_letter_generator(self, job_details: dict, user_data: dict, need_pdf: bool = True):

        try:
            prompt = PromptTemplate(
                template=CV_GENERATOR,
                input_variables=["my_work_information", "job_description"],
                ).format(job_description=job_details, my_work_information=user_data)

            cover_letter = self.llm.get_response(prompt=prompt)

            cv_path = job_doc_name(job_details, self.output_destination, "cv")
            # Save the cover letter in a text file
            with open(cv_path, 'w') as file:
                file.write(cover_letter)

            print("Cover Letter generated at: ", cv_path)
            if need_pdf:
                text_to_pdf(cover_letter, cv_path.replace(".txt", ".pdf"))
                print("Cover Letter PDF generated at: ", cv_path.replace(".txt", ".pdf"))
            
            return cover_letter, cv_path.replace(".txt", ".pdf")
        except Exception as e:
            print(e)
            return None, None


