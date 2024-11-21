import logging
from typing import Optional

def setup_logger(output_file):
    """
    Sets up a logger to write to both the console and a file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)

    # Log to file
    file_handler = logging.FileHandler(output_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger