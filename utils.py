import jinja2
import os
import re
import shutil
import subprocess
from fpdf import FPDF
from matplotlib.font_manager import FontManager
from typing import Union, Any, Dict, List

def text_to_pdf(text: str, file_path: str):
    """Converts the given text to a PDF and saves it to the specified file path.

    Args:
        text (str): The text to be converted to PDF.
        file_path (str): The file path where the PDF will be saved.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    # Encode the text explicitly using 'latin-1' encoding
    encoded_text = text.encode('utf-8').decode('latin-1')
    pdf.multi_cell(0, 5, txt=encoded_text)
    pdf.output(file_path)
    return file_path


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

            # Find fontspec font commands
            matches = fontspec_regex.findall(content)
            for match in matches:
                font_type, font_name = match
                font_name = font_name.strip()
                fonts.add(font_name)
                font_commands.append({'type': font_type, 'font': font_name})

        return fonts, font_commands
    
    @staticmethod
    def compile_latex_to_pdf(tex_file_path: str, destination_path: str = None, latex_engine: str = None):
        """
        Compiles a LaTeX `.tex` file into a PDF using a specified LaTeX engine and saves it to a specified directory.

        This method runs the appropriate LaTeX engine (`pdflatex` or `xelatex`) on the provided `.tex` file.
        If `destination_path` is given, the resulting PDF is saved there; otherwise, it's saved in the same directory as
        the `.tex` file. After compilation, auxiliary files generated by LaTeX are removed to keep the output directory
        clean.

        Args:
            tex_file_path (str): The path to the LaTeX `.tex` file to compile.
            destination_path (str, optional): The directory where the PDF should be saved.
                If not specified, defaults to the directory of `tex_file_path`.
            latex_engine (str, optional): The LaTeX engine to use for compilation. If not specified, will attempt
                to detect the engine or default to `pdflatex`. Other common engines are `xelatex` and `lualatex`.

        Returns:
            None

        Prints:
            - Success message indicating where the PDF was saved.
            - Error messages if PDF generation fails.
        """
        def detect_latex_engine(tex_file_path):
            with open(tex_file_path, 'r') as tex_file:
                tex_content = tex_file.read()
            if '\\usepackage{fontspec}' in tex_content:
                return 'xelatex'
            else:
                return 'pdflatex'
        
        try:
            # Absolute path of the LaTeX file
            tex_file_path = os.path.abspath(tex_file_path)
            tex_dir = os.path.dirname(tex_file_path)
            tex_filename = os.path.basename(tex_file_path)
            filename_without_ext = os.path.splitext(tex_filename)[0]

            # Determine output directory
            if destination_path:
                output_dir = os.path.abspath(destination_path)
            else:
                output_dir = tex_dir

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # if the resume latex file uses the resume class...
            if '\documentclass{resume}' in open(tex_file_path).read():
                # Copy the resume.cls file to the output directory
                cls_file_path = os.path.join(os.getcwd(), 'templates', 'resume.cls')
                if os.path.exists(cls_file_path):
                    shutil.copy(cls_file_path, output_dir)
                else:
                    print(f"Error: resume.cls file not found at {cls_file_path}")
                    return None
            
            if not latex_engine:
                latex_engine = detect_latex_engine(tex_file_path)

            if not shutil.which(latex_engine):
                print(f"Error: LaTeX engine '{latex_engine}' not found.")
                return None

            # Run the LaTeX engine multiple times if necessary.
            # It is common for LaTeX to require multiple runs to resolve references.
            for _ in range(2):
                result = subprocess.run(
                    [latex_engine, "-output-directory", output_dir, tex_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                if result.returncode != 0:
                    print("Error during PDF generation:")
                    print(result.stderr.decode())
                    return None

            # Clean up auxiliary files
            aux_extensions = [".aux", ".log", ".out", ".toc", ".synctex.gz", ".cls"]
            for ext in aux_extensions:
                aux_file = os.path.join(output_dir, filename_without_ext + ext)
                if os.path.exists(aux_file):
                    os.remove(aux_file)

            # Remove the resume.cls file if it was copied
            if cls_file_path != os.path.join(os.getcwd(), 'templates', 'resume.cls'):
                os.remove(os.path.join(output_dir, "resume.cls"))

            print(f"PDF successfully saved to {output_dir}")

        except Exception as e:
            print("An error occurred:")
            print(e)
            return None