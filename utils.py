import os
import re
from datetime import datetime
from fpdf import FPDF


def build_application_destination(
    company_name: str,
    job_title: str,
    output_destination: str = "output_files",
    timestamp: str | None = None,
) -> str:
    """Create the standard `<output>/<Company>/<JobTitle>_<timestamp>` directory."""

    def _clean(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "", value.title().replace(" ", "").strip())
        if not cleaned:
            raise ValueError("Value must contain at least one alphanumeric character after cleaning.")
        return cleaned

    if not company_name or not job_title:
        raise ValueError("company_name and job_title are required to build an application destination.")

    normalized_company = _clean(company_name)
    normalized_job = _clean(job_title)[:15]
    run_timestamp = timestamp or datetime.now().strftime(r"%Y%m%d%H%M%S")

    destination_dir = os.path.join(
        output_destination,
        normalized_company,
        f"{normalized_job}_{run_timestamp}",
    )
    os.makedirs(destination_dir, exist_ok=True)
    return destination_dir


def text_to_pdf(text: str, destination_path: str):
    """
    Convert a string of text into a PDF and save it to the specified destination.
    
    This function searches for fonts in the following order:
      1. '/System/Library/Fonts/Supplemental/DejaVuSans.ttf'
      2. '/Library/Fonts/Arial Unicode.ttf'
      3. The built-in 'Helvetica'
    
    It attempts to load the font file if present. If not found, it falls back to the next option.
    
    Args:
        text (str): The text to be added to the PDF.
        destination_path (str): The destination file path for the generated PDF. A ".pdf" extension is appended if absent.
    
    Returns:
        str: The absolute file path where the PDF is saved.
    """
    # validate destination path
    destination_path = os.path.abspath(destination_path)
    if not destination_path.endswith('.pdf'):
        destination_path += '.pdf'
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # Initialize PDF with default page setup
    pdf = FPDF()
    pdf.add_page()

    # Font selection cascade
    fonts_to_try = [
        ('/System/Library/Fonts/Supplemental/DejaVuSans.ttf', 'DejaVu'),
        ('/Library/Fonts/Arial Unicode.ttf', 'Arial Unicode'),
        (None, 'Helvetica')  # Built-in fallback
    ]

    # Try fonts in order until one works
    for font_path, font_name in fonts_to_try:
        if not font_path or os.path.exists(font_path):
            if font_path:
                pdf.add_font(font_name, '', font_path, uni=True)
            pdf.set_font(font_name, size=11)
            break

    # Write text and save
    pdf.multi_cell(0, 5, txt=text)
    pdf.output(destination_path)
    return destination_path

def print_directory_tree(start_path='.', indent=''):
    """Recursively prints a text-based directory structure."""
    try:
        items = sorted(os.listdir(start_path))  # Sort items alphabetically
    except PermissionError:
        print(f"{indent}[Permission Denied] {start_path}")
        return
    
    for index, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = (index == len(items) - 1)
        connector = '└── ' if is_last else '├── '
        
        print(indent + connector + item)
        
        if os.path.isdir(path):
            new_indent = indent + ('    ' if is_last else '│   ')
            print_directory_tree(path, new_indent)
