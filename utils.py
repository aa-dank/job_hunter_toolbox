import jinja2
import logging
import os
import re
import shutil
import subprocess
from fpdf import FPDF
from matplotlib.font_manager import FontManager
from typing import Union, Any, Dict, List


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


LatexData = Union[str, List['LatexData'], Dict[Any, 'LatexData']]

class LatexToolBox:
    
    @staticmethod
    def escape_for_latex(data: LatexData) -> LatexData:
        """
        Escapes special characters in the given data for LaTeX compatibility. The data passed
        can be a dictionary, list, or string. If a dictionary is passed, the function will
        recursively escape the special characters in the values of the dictionary.
        If a list is passed, the function will escape the special characters in each item of the list.
        If a string is passed, the function will escape the special characters in the string.
        :param data: The data to escape. 
        """
        if isinstance(data, dict):
            new_data = {}
            for key in data.keys():
                new_data[key] = LatexToolBox.escape_for_latex(data[key])
            return new_data
        elif isinstance(data, list):
            return [LatexToolBox.escape_for_latex(item) for item in data]
        elif isinstance(data, str):
            # Adapted from https://stackoverflow.com/q/16259923
            latex_special_chars = {
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\^{}",
                "\\": r"\textbackslash{}",
                "\n": "\\newline%\n",
                "-": r"{-}",
                "\xA0": "~",  # Non-breaking space
                "[": r"{[}",
                "]": r"{]}",
            }
            return "".join([latex_special_chars.get(c, c) for c in data])

        return data
    
    @staticmethod
    def tex_resume_from_jinja_template(jinja_env: jinja2.Environment, json_resume: dict, tex_jinja_template: str):
        """
        Renders LaTeX resume content using a Jinja2 template and the provided resume data.

        Args:
            jinja_env (jinja2.Environment): Environment configured for Jinja2 templates.
            json_resume (dict): The data to populate in the template.
            tex_jinja_template (str): The LaTeX template file name. Defaults to "resume.tex.jinja".

        Returns:
            str: Rendered LaTeX resume content as a string.
        """
        resume_template = jinja_env.get_template(tex_jinja_template)
        resume = resume_template.render(json_resume)
        return resume
    
    @staticmethod
    def check_fonts_installed(fonts_to_check: Union[str, List[str]]) -> Dict[str, bool]:
        """
        Checks if a list of font names are installed on the current system by comparing
        them with the available fonts.

        Args:
            fonts_to_check (Union[str, List[str]]): A single font name or a list of font names to check.

        Returns:
            Dict[str, bool]: A mapping of each font name to a boolean indicating installation status.
        """
        if type(fonts_to_check) == str:
            fonts_to_check = [fonts_to_check]

        fm = FontManager()
        system_fonts = set(f.name for f in fm.ttflist)
        installed_fonts = {}
        for font in fonts_to_check:
            if font in system_fonts:
                installed_fonts[font] = True
            else:
                installed_fonts[font] = False
        return installed_fonts
    
    @staticmethod
    def extract_tex_font_dependencies(tex_file_path):
        """
        Parses a LaTeX file for font commands and returns the fonts found and
        their associated command info, such as main, sans, or mono fonts.

        Args:
            tex_file_path (str): The path to the LaTeX file to inspect.

        Returns:
            Tuple[Set[str], List[Dict[str, str]]]: A set of unique font names and
            a list of command dictionaries that include the type and font name.
        """
        font_commands = []
        fonts = set()

        # Regular expressions to match font commands
        fontspec_regex = re.compile(r'\\set(main|sans|mono)font(?:\[.*?\])?\{([^}]+)\}')
        usepackage_regex = re.compile(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}')

        with open(tex_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove everything after unescaped '%' from each line to skip comments
            content_no_comments = re.sub(r'(?<!\\)%.*', '', content)

            # Find fontspec font commands
            matches = fontspec_regex.findall(content_no_comments)
            for match in matches:
                font_type, font_name = match
                font_name = font_name.strip()
                fonts.add(font_name)
                font_commands.append({'type': font_type, 'font': font_name})

        return fonts, font_commands

    @staticmethod
    def compile_resume_latex_to_pdf(tex_filepath: str,
                             cls_filepath: str,
                             output_destination_path: str,
                             latex_engine: str = None):
        """
        Compiles a LaTeX resume file into a PDF document.
        
        This method handles the entire LaTeX compilation process including:
        1. Resolving and copying the required .cls file to the output directory
        2. Detecting which LaTeX engine to use (xelatex or pdflatex)
        3. Running the LaTeX compiler twice to ensure proper rendering of references
        4. Handling compilation errors and reporting them
        
        Args:
            tex_filepath (str): Path to the LaTeX (.tex) file to compile
            cls_filepath (str): Path to the LaTeX class (.cls) file needed for compilation
            output_destination_path (str): Directory where the output PDF and intermediate files will be saved
            latex_engine (str, optional): LaTeX engine to use ('xelatex' or 'pdflatex'). 
                                         If None, will be auto-detected based on fontspec package usage.
        
        Returns:
            bool: True if compilation was successful, False otherwise
            
        Raises:
            FileNotFoundError: If required files cannot be found
            RuntimeError: If LaTeX compilation fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Resolve cls_filepath as before…
            if not os.path.exists(cls_filepath):
                project_dir = os.path.dirname(os.path.abspath(__file__))
                alt = os.path.join(project_dir, 'templates', cls_filepath)
                if os.path.exists(alt):
                    cls_filepath = alt
                else:
                    raise FileNotFoundError(...)

            # Determine which .cls filename the .tex actually wants:
            with open(tex_filepath, 'r', encoding='utf-8') as f:
                header = f.read(2048)
            m = re.search(r'\\documentclass\{([^}]+)\}', header)
            class_name = m.group(1) if m else os.path.splitext(os.path.basename(cls_filepath))[0]

            # Copy/rename .cls into the build folder as <class_name>.cls
            cls_dest = os.path.join(output_destination_path, f"{class_name}.cls")
            if not os.path.exists(cls_dest):
                shutil.copy2(cls_filepath, cls_dest)
                
            #copy tex file to output dir
            tex_filename = os.path.basename(tex_filepath)
            tex_dest = os.path.join(output_destination_path, tex_filename)
            if not os.path.exists(tex_dest):
                shutil.copy2(tex_filepath, tex_dest)

            # Detect engine if not specified
            if not latex_engine:
                with open(tex_filepath, 'r') as f:
                    latex_engine = 'xelatex' if '\\usepackage{fontspec}' in f.read() else 'pdflatex'

            # Run compilation
            cmd = [latex_engine, "-interaction=nonstopmode", tex_filename]
            logger.info(f"Running LaTeX compilation with command: {' '.join(cmd)}")
            for i in range(2):
                result = subprocess.run(
                    cmd,
                    cwd=output_destination_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Log the results of the subprocess run
                if result.returncode == 0:
                    logger.info(f"LaTeX compilation pass {i+1} succeeded.")
                    logger.info(f"Command: {' '.join(cmd)}")
                    logger.debug(f"Output: {result.stdout.decode()[:200]}...")  # Show first 200 chars of output
                else:
                    error_message = f"LaTeX compilation failed on pass {i+1}.\nCommand: {' '.join(cmd)}\nstderr: {result.stderr.decode()}\nstdout: {result.stdout.decode()}"
                    logger.error(error_message)
                    raise RuntimeError(error_message)

            return True

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return False
        except RuntimeError as e:
            logger.error(f"Compilation error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    @staticmethod
    def cleanup_latex_files(output_dir: str, base_name: str):
        """Removes auxiliary LaTeX files after compilation."""
        extensions = [".aux", ".log", ".out", ".toc", ".synctex.gz"]
        for ext in extensions:
            try:
                aux_file = os.path.join(output_dir, base_name + ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            except Exception as e:
                print(f"Warning: Could not remove {ext} file: {e}")