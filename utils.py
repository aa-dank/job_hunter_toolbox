import jinja2
import os
import re
import shutil
import subprocess
from fpdf import FPDF
from matplotlib.font_manager import FontManager
from typing import Union, Any, Dict, List


def text_to_pdf(text: str, file_path: str):
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
    def compile_latex_to_pdf(tex_file_path: str, cls_file_path: str):
        """
        Compile a LaTeX file to PDF.
        
        Args:
            tex_file_path (str): Path to the .tex file.
            cls_file_path (str): Path to the .cls file to be used.
        """
        def detect_latex_engine(tex_file_path: str) -> str:
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
            output_dir = tex_dir

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # if the resume latex file uses the resume class...
            with open(tex_file_path, 'r') as tex_file:
                if '\documentclass{resume}' in tex_file.read():
                    # Copy the resume.cls file to the output directory
                    if os.path.exists(cls_file_path):
                        # if the resume.cls file is not already in the output directory, copy it
                        if not os.path.exists(os.path.join(output_dir, "resume.cls")):
                            shutil.copy(cls_file_path, output_dir)
                    else:
                        print(f"Error: resume.cls file not found at {cls_file_path}")
                        return None
            
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