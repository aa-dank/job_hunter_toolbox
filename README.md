# Job Hunter Toolbox

This toolbox automates resume and cover letter generation using LaTeX, Python, and LLM services to streamline the job application process.

## Dependencies and Setup
1. Ensure you have a modern Python version installed (e.g., Python 3.9+).
2. Install LaTeX tools (TeX Live, MikTeX, etc.) with at least `pdflatex`.  
   • Optionally install `xelatex` for advanced formatting.  
3. Configure API credentials (e.g., OpenAI keys) for LLMProvider classes.

## Usage
• Configure an LLMProvider with your API credentials.
• Run the full application via:
  
  python main.py

• The application flow will:
  - Extract job and user data.
  - Generate resume JSON and LaTeX files.
  - Open the generated resume and cover letter files for editing.
  - Upon confirmation (pressing Enter), compile the resume LaTeX file into a PDF and convert the cover letter text into a PDF.
  
• Edit the `.tex` and cover letter `.txt` files as needed before final PDF generation.

## Tools & Scripts
• The `LatexToolBox` class in `utils.py` handles LaTeX compilation.  
• Main application logic is implemented in `main.py`.  
• A similar flow applies for cover letters—edit then convert to PDF.
