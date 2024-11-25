import os

from creds import OPENAI_KEY
from resume_generator import ResumeGenerator

generator = ResumeGenerator(
    api_key=OPENAI_KEY,
    provider="GPT", #ENTER PROVIDER <gemini> or <openai>
    downloads_dir=r"/Users/aaronrdankert/projects/job_hunter/output_files", #[optional] ENTER FOLDER PATH WHERE FILE GET DOWNLOADED, By default, 'downloads' folder
    model="gpt-4o"#"chatgpt-4o-latest"
)

#job_content_path = r"input_data/joby_sample_job.pdf"
job_content_path = r"input_data/google_internship.pdf"
resume_pdf_path = r"input_data/full_resume_contents_20241122.pdf"
#resume_pdf_path = r"input_data/Grad School Resume.pdf"
generator.resume_cv_pipeline(
    job_filepath=job_content_path, # .pdf or .json
    user_data_path= resume_pdf_path # .pdf or .json
) # Return and downloads curated resume and cover letter.