# logger.py
import logging
import os
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO, console: bool = True, notebook: bool = False) -> logging.Logger:
    """
    Sets up a logger with the specified name, log file, and logging level.

    Args:
        name (str): Name of the logger.
        log_file (Optional[str]): Path to the log file. If None, file logging is disabled.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        console (bool): Whether to log to the console as well.
        notebook (bool): Whether to log to Jupyter notebook cell output.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

        # Console handler (optional)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(console_handler)

        # Notebook handler (optional)
        if notebook:
            from IPython.display import display
            from ipywidgets import Output

            output_widget = Output()
            display(output_widget)

            class WidgetHandler(logging.Handler):
                def emit(self, record):
                    log_entry = self.format(record)
                    with output_widget:
                        print(log_entry)

            widget_handler = WidgetHandler()
            widget_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(widget_handler)

    return logger