import os
from creds import OPENAI_KEY
from resume_generator import JobApplicationBuilder

generator = JobApplicationBuilder(
    api_key=OPENAI_KEY,
    provider="GPT", #ENTER PROVIDER <gemini> or <openai>
    output_destination=r"output_files", #[optional] ENTER FOLDER PATH WHERE FILE GET DOWNLOADED, By default, 'downloads' folder
    model="gpt-4o"#"chatgpt-4o-latest"
)

job_content_path = r"input_data/Adobe Finance Data Scientist.pdf"
resume_pdf_path = r"input_data/full_resume_contents_20241124.pdf"
#resume_pdf_path = r"input_data/Grad School Resume.pdf"
job_details_dict, job_details_filepath = generator.extract_job_content(job_content_path)
resume_details_dict, resume_details_filepath = generator.generate_resume_json(job_content=job_details_dict,
                                                                              user_data=resume_pdf_path)

resume_latex, resume_tex_path = generator.resume_json_to_resume_tex(resume_details=resume_details_dict, tex_filename="resume.tex")

# prompt user to edit the tex file before converting to pdf. elicit user input to continue with the conversion.
print(f"Please edit the resume.tex file before converting to pdf.")
print(f"Press Enter to continue after editing the resume.tex file.")
input()
generator.save_latex_as_pdf(tex_file_path=resume_tex_path, dst_path="output_files")
print(f"Resume PDF is saved at output_files folder.")

# Create corresponding cover letter
cover_letter, cover_letter_tex_path = generator.cover_letter_generator(job_details=job_details_dict, user_data=resume_details_dict)
print(f"Cover Letter is saved at {cover_letter_tex_path}")



