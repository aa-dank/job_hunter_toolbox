import jinja2
import json
import os
import re
import shutil
import subprocess
import validators

from bs4 import BeautifulSoup
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from models import ChatGPT, Gemini, OllamaModel
from generation_schemas import Achievements, Certifications, Educations, Experiences, JobDetails, Projects, ResumeSchema, SkillSections
from zlm import AutoApplyModel
from prompts.extraction_prompts import RESUME_DETAILS_EXTRACTOR, JOB_DETAILS_EXTRACTOR, CV_GENERATOR, RESUME_WRITER_PERSONA
from prompts.resume_section_prompts import EXPERIENCE, SKILLS, PROJECTS, EDUCATIONS, CERTIFICATIONS, ACHIEVEMENTS
from zlm.utils.data_extraction import extract_text
from zlm.utils.latex_ops import escape_for_latex, use_template
from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity, vector_embedding_similarity
from zlm.utils import utils


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


class JobApplicationBuilder:
 
    def __init__(
        self, api_key: str, provider: str, model: str, output_destination: str, system_prompt: str = RESUME_WRITER_PERSONA
    ):
        
        default_llm_provider = "GPT"
        default_llm_model = "gpt-4o"
        self.system_prompt = system_prompt
        self.provider = default_llm_provider if provider is None or provider.strip() == "" else provider
        self.model = default_llm_model if model is None or model.strip() == "" else model
        self.output_destination = output_destination
        if api_key is None or api_key.strip() == "os":
                api_env = llm_mapping_dict[self.provider]["api_env"]
                if api_env != None and api_env.strip() != "":
                    self.api_key = os.environ.get(llm_mapping_dict[self.provider]["api_env"]) 
                else:
                    self.api_key = None
        else:
            self.api_key = api_key

        self.llm = self.get_llm_instance()

    def get_llm_instance(self):
        if self.provider == "GPT":
            return ChatGPT(api_key=self.api_key, model=self.model, system_prompt=self.system_prompt)
        elif self.provider == "Gemini":
            return Gemini(api_key=self.api_key, model=self.model, system_prompt=self.system_prompt)
        elif self.provider == "Ollama":
            return OllamaModel(model=self.model, system_prompt=self.system_prompt)
        else:
            raise Exception("Invalid LLM Provider")

    @staticmethod
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

    def process_job_html(self, job_filepath: str):
        with open(job_filepath, 'r') as file:
            job_html = file.read()
        
        soup = BeautifulSoup(job_html, 'html.parser')
        job_content = soup.get_text()
        return job_content

    def extract_job_content(self, job_filepath: str):
        extension = job_filepath.split('.')[-1]
        if not extension in ['pdf', 'json', 'txt', 'html', 'md']:
            raise ValueError(f"Unsupported file format: {extension}\nfile_path: {job_filepath}")

        if extension == 'pdf':
            job_content_str = extract_text(job_filepath)
        elif extension == 'json':
            with open(job_filepath, 'r') as file:
                job_content_str = json.load(file)
        elif extension == 'txt':
            with open(job_filepath, 'r') as file:
                job_content_str = file.read()
        elif extension == 'md':
            with open(job_filepath, 'r') as file:
                job_content_str = file.read()
        
        # process html file into job_content_str
        elif extension == 'html':
            job_content_str = self.process_job_html(job_filepath)

        json_parser = JsonOutputParser(pydantic_object=JobDetails)
        
        prompt = PromptTemplate(
            template=JOB_DETAILS_EXTRACTOR,
            input_variables=["job_description"],
            partial_variables={"format_instructions": json_parser.get_format_instructions()}
            ).format(job_description=job_content_str)

        job_details = self.llm.get_response(prompt=prompt, need_json_output=True)

        jd_path = self.job_doc_name(job_details, self.output_destination, "jd")

        utils.write_json(jd_path, job_details)
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
        resume_text = extract_text(pdf_path)

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
            user_data = utils.read_json(user_data_path)
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

                response = self.llm.get_response(prompt=prompt, expecting_longer_output=True, need_json_output=True)

                # Check for empty sections
                if response is not None and isinstance(response, dict):
                    if section in response:
                        if response[section]:
                            if section == "skill_section":
                                resume_details[section] = [i for i in response['skill_section'] if len(i['skills'])]
                            else:
                                resume_details[section] = response[section]

            resume_details['keywords'] = ', '.join(job_content['keywords'])
            
            resume_path = self.job_doc_name(job_content, self.output_destination, "resume")

            utils.write_json(resume_path, resume_details)

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
            utils.write_file(tex_temp_path, resume_latex)
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

            cover_letter = self.llm.get_response(prompt=prompt, expecting_longer_output=True)

            cv_path = self.job_doc_name(job_details, self.output_destination, "cv")
            utils.write_file(cv_path, cover_letter)
            print("Cover Letter generated at: ", cv_path)
            if need_pdf:
                utils.text_to_pdf(cover_letter, cv_path.replace(".txt", ".pdf"))
                print("Cover Letter PDF generated at: ", cv_path.replace(".txt", ".pdf"))
            
            return cover_letter, cv_path.replace(".txt", ".pdf")
        except Exception as e:
            print(e)
            return None, None

    
