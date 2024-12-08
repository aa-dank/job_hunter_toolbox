import jinja2
import json
import os
import PyPDF2
import re
import shutil
import subprocess


from datetime import datetime
from fpdf import FPDF
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from models import LLMProvider
from generation_schemas import Achievements, Certifications, Educations, Experiences, JobDetails, Projects, ResumeSchema, SkillSections
#from zlm import AutoApplyModel
from prompts.extraction_prompts import RESUME_DETAILS_EXTRACTOR, JOB_DETAILS_EXTRACTOR, COVER_LETTER_GENERATOR
from prompts.resume_section_prompts import EXPERIENCE, SKILLS, PROJECTS, EDUCATIONS, CERTIFICATIONS, ACHIEVEMENTS, RESUME_WRITER_PERSONA


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

def use_template(jinja_env: jinja2.Environment, json_resume: dict):
    try:
        resume_template = jinja_env.get_template(f"resume.tex.jinja")
        resume = resume_template.render(json_resume)

        return resume
    except Exception as e:
        print(e)
        return None

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
    return file_path

class JobApplicationBuilder:
 
    def __init__(
        self, llm: LLMProvider, output_destination: str = "output_files",
        system_prompt: str = RESUME_WRITER_PERSONA
    ):
        self.system_prompt = system_prompt
        self.llm = llm
        self.output_destination = output_destination
        self.org = None
        self.job_title = None
        self.timestamp = datetime.now().strftime(r"%Y%m%d%H%M%S")

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
        self.org = job_details["company_name"]
        self.job_title = job_details["job_title"]

        jd_path = self.get_job_doc_path(file_type="jd")

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
                print(f"Processing Resume's {section.upper()} Section...")
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

            escaped_resume_dict = escape_for_latex(resume_details)

            resume_latex = use_template(latex_jinja_env, escaped_resume_dict)

            
            # Save the resume in a tex file
            with open(output_path, 'w') as file:
                file.write(resume_latex)

            return resume_latex, output_path
        
        except Exception as e:
            print(e)
            return None, None
        
    @staticmethod
    def save_latex_as_pdf(tex_file_path: str, destination_path: str = None):
        """
        Compiles a LaTeX `.tex` file into a PDF using `pdflatex` and saves it to a specified directory.

        This method runs `pdflatex` on the provided `.tex` file. If `destination_path` is given,
        the resulting PDF is saved there; otherwise, it's saved in the same directory as the `.tex` file.
        After compilation, auxiliary files generated by LaTeX are removed to keep the output directory clean.

        Args:
            tex_file_path (str): The path to the LaTeX `.tex` file to compile.
            destination_path (str, optional): The directory where the PDF should be saved.
                If not specified, defaults to the directory of `tex_file_path`.

        Returns:
            None

        Prints:
            - Success message indicating where the PDF was saved.
            - Error messages if PDF generation fails.
        """
        try:
            # Absolute path of the LaTeX file
            tex_file_path = os.path.abspath(tex_file_path)
            tex_dir = os.path.dirname(tex_file_path)
            tex_filename = os.path.basename(tex_file_path)
            filename_without_ext = os.path.splitext(tex_filename)[0]

            # Determine output directory
            if destination_path:
                output_dir = os.path.abspath(destination_path)
            else:
                output_dir = tex_dir

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Copy the resume.cls file to the output directory
            cls_file_path = os.path.join(os.getcwd(), 'templates', 'resume.cls')
            if os.path.exists(cls_file_path):
                shutil.copy(cls_file_path, output_dir)
            else:
                print(f"Error: resume.cls file not found at {cls_file_path}")
                return None

            # Run pdflatex with the output directory option
            result = subprocess.run(
                ["pdflatex", "-output-directory", output_dir, tex_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                print("Error during PDF generation:")
                print(result.stderr.decode())
                return None

            # Clean up auxiliary files
            aux_extensions = [".aux", ".log", ".out", ".toc"]
            for ext in aux_extensions:
                aux_file = os.path.join(output_dir, filename_without_ext + ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)

            # Remove the resume.cls file if it was copied
            if cls_file_path != os.path.join(os.getcwd(), 'templates', 'resume.cls'):
                os.remove(os.path.join(output_dir, "resume.cls"))

            print(f"PDF successfully saved to {output_dir}")

        except Exception as e:
            print("An error occurred:")
            print(e)
            return None

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


