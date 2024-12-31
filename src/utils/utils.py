import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Optional


def get_config(config_file) -> Dict:
    current_dir = Path(__file__).resolve().parent.parent
    config_path = current_dir.parent / "configs" / config_file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_logger_basic():
    # Setup
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    return logger


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