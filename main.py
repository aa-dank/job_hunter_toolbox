import os
from creds import OPENAI_KEY
from application_generator import JobApplicationBuilder
from models import ChatGPT
from prompts.resume_section_prompts import RESUME_WRITER_PERSONA
from utils import LatexToolBox, text_to_pdf

llm = ChatGPT(
    api_key=OPENAI_KEY,
    model="gpt-4o",
    system_prompt=RESUME_WRITER_PERSONA,
    max_output_tokens=None,
    temperature=0.7
)

generator = JobApplicationBuilder(
    llm=llm,
)

if __name__ == "__main__":
    job_content_path = r""
    resume_pdf_path = r""
    
    job_details_dict, job_details_filepath = generator.extract_job_content(job_content_path)
    resume_dict = generator.user_data_extraction(user_data_path=resume_pdf_path)
    resume_details_dict, resume_details_filepath = generator.generate_resume_json(
        job_content=job_details_dict,
        user_data=resume_dict
    )
    resume_latex, resume_tex_path = generator.resume_json_to_resume_tex(resume_details=resume_details_dict)
    print("Done generating resume tex file")

    cover_letter, cover_letter_txt_path = generator.generate_cover_letter(
        job_details=job_details_dict, 
        user_data=resume_details_dict,
        need_pdf=False
    )
    
    print('Edit the resume latex and cover letter text files as needed. Hit enter when you are ready to turn them into pdf files.')
    input()

    print("Compiling LaTeX to PDF...")
    LatexToolBox.compile_latex_to_pdf(tex_file_path=resume_tex_path)
    print(f"Resume PDF is saved at {resume_tex_path.replace('.tex','.pdf')}")

    cover_letter_pdf_path = cover_letter_txt_path.replace('.txt', '.pdf')
    cover_letter_pdf_path = text_to_pdf(cover_letter, cover_letter_pdf_path)
    print(f"Cover Letter PDF is saved at {cover_letter_pdf_path}")

    font_statuses = LatexToolBox.check_fonts_installed(resume_tex_path)
    if not all(font_statuses.values()):
        for k, v in font_statuses.items():
            if not v:
                print(f"{k} not installed")
    # ...existing code...



