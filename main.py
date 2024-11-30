import os
from creds import OPENAI_KEY
from resume_generator import ResumeGenerator

generator = ResumeGenerator(
    api_key=OPENAI_KEY,
    provider="GPT", #ENTER PROVIDER <gemini> or <openai>
    downloads_dir=r"output_files", #[optional] ENTER FOLDER PATH WHERE FILE GET DOWNLOADED, By default, 'downloads' folder
    model="gpt-4o"#"chatgpt-4o-latest"
)

job_content_path = r"input_data/Adobe Finance Data Scientist.pdf"
resume_pdf_path = r"input_data/full_resume_contents_20241124.pdf"
#resume_pdf_path = r"input_data/Grad School Resume.pdf"
job_details_dict, job_details_filepath = generator.extract_job_content(job_content_path)
resume_details_dict, resume_details_filepath = generator.generate_resume_json(job_content=job_details_dict,
                                                                              user_data=resume_pdf_path)

resume_latex = generator.resume_json_to_resume_tex(resume_details=resume_details_dict, tex_filename="resume.tex")


