
import json
import os

from bs4 import BeautifulSoup
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from zlm import AutoApplyModel
from zlm.prompts.resume_prompt import RESUME_DETAILS_EXTRACTOR, JOB_DETAILS_EXTRACTOR
from zlm.schemas.job_details_schema import JobDetails
from zlm.utils.data_extraction import extract_text
from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity, vector_embedding_similarity
from zlm.utils import utils


module_dir = os.path.dirname(__file__)
demo_data_path = os.path.join(module_dir, "demo_data", "user_profile.json")

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

        if extension == 'pdf':
            job_content_str = extract_text(job_filepath)
        elif extension == 'json':
            with open(job_filepath, 'r') as file:
                job_content_str = json.load(file)
        elif extension == 'txt':
            with open(job_filepath, 'r') as file:
                job_content_str = file.read()
        
        # process html file into job_content_str
        elif extension == 'html':
            job_content_str = self.process_job_html(job_filepath)
        else:
            raise ValueError(f"Unsupported file format: {extension}")


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
        
    def resume_cv_pipeline(self, job_url: str = None, job_filepath: str = None, user_data_path: str = demo_data_path):
        """Run the Auto Apply Pipeline.

        Args:
            job_url (str): The URL of the job to apply for.
            user_data_path (str, optional): The path to the user profile data file.
                Defaults to os.path.join(module_dir, "master_data','user_profile.json").

        Returns:
            None: The function prints the progress and results to the console.
        """
        try:
            if user_data_path is None or user_data_path.strip() == "":
                user_data_path = demo_data_path
            
            # Extract user data
            user_data = self.user_data_extraction(user_data_path)

            # Extract job details
            if not job_url and not job_filepath:
                raise ValueError("Either job_url or job_filepath is required.")
            
            if job_url:
                job_details, jd_path = self.job_details_extraction(url=job_url)
            
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

    
