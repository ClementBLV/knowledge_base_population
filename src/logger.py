import logging
from typing import Optional


def setup_logging(debug: bool = False, redirect: Optional[str] = None) -> None:
    """Setup logging configuration.

    :param debug: Whether to enable debug mode. Defaults to False.
    :param redirect: Optional file path to redirect logs to. Defaults to None.
    """

    # Check if the root logger already has handlers (i.e., if logging is already configured)
    if logging.getLogger().hasHandlers():
        return  # Exit if logging is already set up

    # Set log format based on whether debug mode is enabled
    log_format = (
        "%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s"
        if debug
        else "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Basic configuration for console logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()],  # Stream to console
    )

    # Optionally redirect logs to a file if specified
    if redirect:
        file_handler = logging.FileHandler(redirect)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
