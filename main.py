import os
import warnings
from creds import OPENAI_KEY
from application_generator import JobApplicationBuild, JobApplicationBuilder
from models import ChatGPT
from prompts.resume_section_prompts import RESUME_WRITER_PERSONA
from latex_toolbox import (
    check_fonts_installed,
    cleanup_latex_files,
    compile_resume_latex_to_pdf,
    extract_tex_font_dependencies,
)
from utils import text_to_pdf
from logger import setup_logger

# Initialize logger
logger = setup_logger(name="JobHunterToolbox", log_file="job_hunter_toolbox.log")

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger.info("Starting Job Hunter Toolbox application")
    
    llm = ChatGPT(
        api_key=OPENAI_KEY,
        model="gpt-4o",
        system_prompt=RESUME_WRITER_PERSONA,
        max_output_tokens=None
    )

    generator = JobApplicationBuilder(
        llm=llm,
    )

    build = JobApplicationBuild(
        # This is the template that will be used to generate the resume
        resume_tex_template_path="slightly_less_basic_template.tex",
        # This is the path to the file with a description of the job you are applying for
        job_content_path="input_data/job_description.pdf",
        # This is the path to the file with your resume content that will be used to generate the resume
        user_details_content_path="input_data/user_resume.pdf"
    )

    try:
        build = generator.extract_job_content(build)
        logger.info("Job content extracted successfully.")

        build = generator.user_data_extraction(build)
        logger.info("User data extracted successfully.")

        build = generator.generate_resume_json(build)
        logger.info("Resume JSON generated successfully.")

        build = generator.resume_json_to_resume_tex(build)
        logger.info("Resume TeX generated successfully.")

        build = generator.generate_cover_letter(build, custom_instructions="")
        logger.info("Cover letter generated successfully.")

        logger.info('Edit the resume latex and cover letter text files as needed. Hit enter when you are ready to turn them into pdf files.')
        input()

        resume_tex_fonts = extract_tex_font_dependencies(build.resume_tex_path)
        font_statuses = check_fonts_installed(resume_tex_fonts)
        if not all([v for v in font_statuses.values()]):
            for k, v in font_statuses.items():
                if not v:
                    logger.warning(f"{k} not installed")

        logger.info("Compiling LaTeX to PDF...")
        output_dir = os.path.dirname(os.path.abspath(build.resume_tex_path))
        success = compile_resume_latex_to_pdf(
            tex_filepath=build.resume_tex_path, 
            cls_filepath=build.resume_cls_path,
            output_destination_path=output_dir
        )
        if success:
            # Remove auxiliary files using resume base name (without extension)
            base_name = os.path.splitext(os.path.basename(build.resume_tex_path))[0]
            cleanup_latex_files(output_dir, base_name)
            logger.info(f"Resume PDF is saved at {build.resume_tex_path.replace('.tex','.pdf')}")
        else:
            logger.error("LaTeX compilation failed.")

        cover_letter_pdf_path = build.cover_letter_path.replace('.txt', '.pdf')
        cover_letter_pdf_path = text_to_pdf(build.cover_letter_path, cover_letter_pdf_path)
        logger.info(f"Cover Letter PDF is saved at {cover_letter_pdf_path}")

        generator.cleanup_files(build)
        logger.info("Temporary files cleaned up successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()



