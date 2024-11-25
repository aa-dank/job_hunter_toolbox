import jinja2
import json
import os
import subprocess

from bs4 import BeautifulSoup
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from zlm import AutoApplyModel
from zlm.prompts.resume_prompt import RESUME_DETAILS_EXTRACTOR, JOB_DETAILS_EXTRACTOR
from zlm.schemas.job_details_schema import JobDetails
from zlm.utils.data_extraction import read_data_from_url, extract_text
from zlm.utils.latex_ops import escape_for_latex, use_template
from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity, vector_embedding_similarity
from zlm.utils import utils
from zlm.variables import section_mapping


def save_latex_as_pdf(tex_file_path: str, dst_path: str):
    try:
        # Call pdflatex to convert LaTeX to PDF
        prev_loc = os.getcwd()
        os.chdir(os.path.dirname(tex_file_path))
        try:
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


def latex_to_pdf(json_resume, dst_path):
    try:
        module_dir = os.path.dirname(__file__)
        templates_path = os.path.join(os.path.dirname(module_dir), 'templates')

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

        escaped_json_resume = escape_for_latex(json_resume)

        resume_latex = use_template(latex_jinja_env, escaped_json_resume)

        tex_temp_path = os.path.join(os.path.realpath(templates_path), os.path.basename(dst_path).replace(".pdf", ".tex"))

        utils.write_file(tex_temp_path, resume_latex)
        save_latex_as_pdf(tex_temp_path, dst_path)
        return resume_latex
    except Exception as e:
        print(e)
        return None

class ResumeGenerator(AutoApplyModel):

    def __init__(self, api_key, provider, downloads_dir, model):
        super().__init__(api_key=api_key, provider=provider, downloads_dir=downloads_dir, model=model)


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

        jd_path = utils.job_doc_name(job_details, self.downloads_dir, "jd")

        utils.write_json(jd_path, job_details)
        print(f"Job Details JSON generated at: {jd_path}")

        return job_details, jd_path
    
    def resume_builder(self, job_details: dict, user_data: dict):
        """
        Builds a resume based on the provided job details and user data.

        Args:
            job_details (dict): A dictionary containing the job description.
            user_data (dict): A dictionary containing the user's resume or work information.

        Returns:
            dict: The generated resume details.

        Raises:
            FileNotFoundError: If the system prompt files are not found.
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
                json_parser = JsonOutputParser(pydantic_object=section_mapping[section]["schema"])
                
                prompt = PromptTemplate(
                    template=section_mapping[section]["prompt"],
                    partial_variables={"format_instructions": json_parser.get_format_instructions()}
                    ).format(section_data = json.dumps(user_data[section]), job_description = json.dumps(job_details))

                response = self.llm.get_response(prompt=prompt, expecting_longer_output=True, need_json_output=True)

                # Check for empty sections
                if response is not None and isinstance(response, dict):
                    if section in response:
                        if response[section]:
                            if section == "skill_section":
                                resume_details[section] = [i for i in response['skill_section'] if len(i['skills'])]
                            else:
                                resume_details[section] = response[section]

            resume_details['keywords'] = ', '.join(job_details['keywords'])
            
            resume_path = utils.job_doc_name(job_details, self.downloads_dir, "resume")

            utils.write_json(resume_path, resume_details)
            resume_path = resume_path.replace(".json", ".pdf")
            # st.write(f"resume_path: {resume_path}")

            resume_latex = latex_to_pdf(resume_details, resume_path)
            # st.write(f"resume_pdf_path: {resume_pdf_path}")

            return resume_path, resume_details
        except Exception as e:
            print(e)
            return resume_path, resume_details
        
    def resume_cv_pipeline(self, user_data_path: str, job_filepath: str):
        """
        Main pipeline for resume generation and analysis.
        """
        try:
            
            # Extract user data
            user_data = self.user_data_extraction(user_data_path)

            # Extract job details
            if not job_filepath:
                raise ValueError("Please provide a job description file path.")
            
            if job_filepath:
                job_details, jd_path = self.extract_job_content(job_filepath)

            # Build resume
            resume_path, resume_details = self.resume_builder(job_details, user_data)
            # resume_details = read_json("/Users/saurabh/Downloads/JobLLM_Resume_CV/Netflix/Netflix_MachineLearning_resume.json")

            # Generate cover letter
            cv_details, cv_path = self.cover_letter_generator(job_details, user_data)

            # Calculate metrics
            for metric in ['jaccard_similarity', 'overlap_coefficient', 'cosine_similarity']:
                print(f"\nCalculating {metric}...")

                if metric == 'vector_embedding_similarity':
                    llm = self.get_llm_instance('')
                    user_personlization = globals()[metric](llm, json.dumps(resume_details), json.dumps(user_data))
                    job_alignment = globals()[metric](llm, json.dumps(resume_details), json.dumps(job_details))
                    job_match = globals()[metric](llm, json.dumps(user_data), json.dumps(job_details))
                else:
                    user_personlization = globals()[metric](json.dumps(resume_details), json.dumps(user_data))
                    job_alignment = globals()[metric](json.dumps(resume_details), json.dumps(job_details))
                    job_match = globals()[metric](json.dumps(user_data), json.dumps(job_details))

                print("User Personlization Score(resume,master_data): ", user_personlization)
                print("Job Alignment Score(resume,JD): ", job_alignment)
                print("Job Match Score(master_data,JD): ", job_match)

            print("\nDone!!!")
        except Exception as e:
            # print full error and stack trace
            print(e)

            return None

    
