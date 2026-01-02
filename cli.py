"""
CLI interface for Job Hunter Toolbox using Click.

This module provides command-line access to various Job Hunter Toolbox features,
starting with LaTeX resume compilation.
"""

import click
import logging
import os
import sys
from pathlib import Path

from latex_toolbox import compile_resume_latex_to_pdf, cleanup_latex_files
from logger import setup_logger


@click.group()
@click.version_option(version="0.1.0", prog_name="Job Hunter Toolbox")
def cli():
    """
    Job Hunter Toolbox CLI - Automate resume and cover letter generation.
    
    Use this CLI to compile LaTeX resumes, generate applications, and more.
    """
    pass


@cli.command(name="compile-resume")
@click.argument("tex_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-c",
    "--cls-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the LaTeX class (.cls) file. If not provided, will search in templates/ directory.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for the compiled PDF. Defaults to the directory containing the .tex file.",
)
@click.option(
    "-e",
    "--engine",
    type=click.Choice(["xelatex", "pdflatex"], case_sensitive=False),
    help="LaTeX engine to use. If not specified, auto-detects based on fontspec package usage.",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="Remove auxiliary LaTeX files after compilation (default: cleanup).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging output.",
)
def compile_resume(tex_file, cls_file, output_dir, engine, cleanup, verbose):
    """
    Compile a LaTeX resume file to PDF.
    
    This command compiles a .tex resume file into a PDF using either pdflatex or xelatex.
    The LaTeX class file (.cls) can be specified explicitly or will be auto-detected from
    the templates/ directory based on the \\documentclass declaration in the .tex file.
    
    Example usage:
    
        \b
        # Compile a resume with auto-detection
        job-hunter compile-resume resume.tex
        
        \b
        # Specify custom class file and output directory
        job-hunter compile-resume resume.tex -c custom.cls -o output/
        
        \b
        # Use specific engine without cleanup
        job-hunter compile-resume resume.tex -e xelatex --no-cleanup
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger("cli", level=log_level)
    
    # Resolve paths
    tex_file = tex_file.resolve()
    
    if output_dir:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = tex_file.parent
    
    # Determine cls file
    if cls_file:
        cls_file = cls_file.resolve()
    else:
        # Try to extract class name from tex file
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                import re
                header = f.read(2048)
                match = re.search(r'\\documentclass\{([^}]+)\}', header)
                if match:
                    class_name = match.group(1)
                    # Look in templates directory
                    project_dir = Path(__file__).parent
                    cls_file = project_dir / 'templates' / f'{class_name}.cls'
                    if not cls_file.exists():
                        click.echo(f"Error: Could not find class file {cls_file}", err=True)
                        click.echo(f"Please specify the class file with -c/--cls-file option", err=True)
                        sys.exit(1)
                else:
                    click.echo("Error: Could not determine document class from .tex file", err=True)
                    click.echo("Please specify the class file with -c/--cls-file option", err=True)
                    sys.exit(1)
        except Exception as e:
            click.echo(f"Error reading .tex file: {e}", err=True)
            sys.exit(1)
    
    # Display compilation info
    click.echo(f"Compiling resume...")
    click.echo(f"  Input:  {tex_file}")
    click.echo(f"  Class:  {cls_file}")
    click.echo(f"  Output: {output_dir}")
    if engine:
        click.echo(f"  Engine: {engine}")
    
    # Compile
    try:
        success = compile_resume_latex_to_pdf(
            tex_filepath=str(tex_file),
            cls_filepath=str(cls_file),
            output_destination_path=str(output_dir),
            latex_engine=engine,
        )
        
        if success:
            pdf_name = tex_file.stem + '.pdf'
            pdf_path = output_dir / pdf_name
            
            # Cleanup auxiliary files if requested
            if cleanup:
                click.echo(f"Cleaning up auxiliary files...")
                cleanup_latex_files(str(output_dir), tex_file.stem)
            
            click.secho(f"✓ Successfully compiled to {pdf_path}", fg="green", bold=True)
            return 0
        else:
            click.secho(f"✗ Compilation failed. Check logs for details.", fg="red", bold=True, err=True)
            sys.exit(1)
            
    except Exception as e:
        click.secho(f"✗ Error during compilation: {e}", fg="red", bold=True, err=True)
        if verbose:
            logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    cli()
