import jinja2
import logging
import os
import re
import shutil
import subprocess
from matplotlib.font_manager import FontManager
from typing import Union, Any, Dict, List, Set, Tuple

# Type alias for latex data structures
LatexData = Union[str, List['LatexData'], Dict[Any, 'LatexData']]


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
            new_data[key] = escape_for_latex(data[key])
        return new_data
    elif isinstance(data, list):
        return [escape_for_latex(item) for item in data]
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


def tex_resume_from_jinja_template(
    jinja_env: jinja2.Environment,
    json_resume: dict,
    tex_jinja_template: str,
):
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


def extract_tex_font_dependencies(tex_file_path: str) -> Tuple[Set[str], List[Dict[str, str]]]:
    """
    Parses a LaTeX file for font commands and returns the fonts found and
    their associated command info, such as main, sans, or mono fonts.

    Args:
        tex_file_path (str): The path to the LaTeX file to inspect.

    Returns:
        Tuple[Set[str], List[Dict[str, str]]]: A set of unique font names and
        a list of command dictionaries that include the type and font name.
    """
    font_commands: List[Dict[str, str]] = []
    fonts: Set[str] = set()

    fontspec_regex = re.compile(r"\\set(main|sans|mono)font(?:\[.*?\])?\{([^}]+)\}")

    with open(tex_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        content_no_comments = re.sub(r"(?<!\\)%.*", '', content)

        matches = fontspec_regex.findall(content_no_comments)
        for match in matches:
            font_type, font_name = match
            font_name = font_name.strip()
            fonts.add(font_name)
            font_commands.append({'type': font_type, 'font': font_name})

    return fonts, font_commands


def compile_resume_latex_to_pdf(
    tex_filepath: str,
    cls_filepath: str,
    output_destination_path: str,
    latex_engine: str = None,
) -> bool:
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
        if not os.path.exists(cls_filepath):
            project_dir = os.path.dirname(os.path.abspath(__file__))
            alt = os.path.join(project_dir, 'templates', cls_filepath)
            if os.path.exists(alt):
                cls_filepath = alt
            else:
                raise FileNotFoundError(...)

        with open(tex_filepath, 'r', encoding='utf-8') as f:
            header = f.read(2048)
        m = re.search(r"\\documentclass\{([^}]+)\}", header)
        class_name = m.group(1) if m else os.path.splitext(os.path.basename(cls_filepath))[0]

        cls_dest = os.path.join(output_destination_path, f"{class_name}.cls")
        if not os.path.exists(cls_dest):
            shutil.copy2(cls_filepath, cls_dest)

        tex_filename = os.path.basename(tex_filepath)
        tex_dest = os.path.join(output_destination_path, tex_filename)
        if not os.path.exists(tex_dest):
            shutil.copy2(tex_filepath, tex_dest)

        if not latex_engine:
            with open(tex_filepath, 'r') as f:
                latex_engine = 'xelatex' if '\\usepackage{fontspec}' in f.read() else 'pdflatex'

        cmd = [latex_engine, "-interaction=nonstopmode", tex_filename]
        logger.info(f"Running LaTeX compilation with command: {' '.join(cmd)}")
        for i in range(2):
            result = subprocess.run(
                cmd,
                cwd=output_destination_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode == 0:
                logger.info(f"LaTeX compilation pass {i+1} succeeded.")
                logger.info(f"Command: {' '.join(cmd)}")
                logger.info(f"Output: {result.stdout.decode()[:200]}...")
            else:
                error_message = (
                    f"   failed on pass {i+1}.\nCommand: {' '.join(cmd)}\nstderr: {result.stderr.decode()}\n"
                    f"stdout: {result.stdout.decode()}"
                )
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


def cleanup_latex_files(output_dir: str, base_name: str):
    """
    Removes auxiliary files generated during LaTeX compilation.

    This function deletes common temporary files (such as .aux, .log, .out, .toc, .synctex.gz)
    that are created when compiling a LaTeX document. It helps keep the output directory clean
    by removing these files after the PDF has been generated.

    Args:
        output_dir (str): The directory where the LaTeX files are located.
        base_name (str): The base name of the LaTeX file (without extension).

    Example:
        cleanup_latex_files('/path/to/output', 'resume')
        # This will attempt to remove files like '/path/to/output/resume.aux', etc.
    """
    extensions = [".aux", ".log", ".out", ".toc", ".synctex.gz"]
    for ext in extensions:
        try:
            aux_file = os.path.join(output_dir, base_name + ext)
            if os.path.exists(aux_file):
                os.remove(aux_file)
        except Exception as e:
            print(f"Warning: Could not remove {ext} file: {e}")
