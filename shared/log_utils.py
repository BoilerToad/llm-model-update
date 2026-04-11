"""
shared/log_utils.py
───────────────────
Common logging setup for both ollama_updater and mlx_updater.
"""

import logging
import os
import sys
from pathlib import Path


USE_COLOR = sys.stdout.isatty()

C = {
    "reset":  "\033[0m"  if USE_COLOR else "",
    "green":  "\033[92m" if USE_COLOR else "",
    "yellow": "\033[93m" if USE_COLOR else "",
    "red":    "\033[91m" if USE_COLOR else "",
    "cyan":   "\033[96m" if USE_COLOR else "",
    "bold":   "\033[1m"  if USE_COLOR else "",
}


def setup_logging(log_path: str, name: str = "updater") -> logging.Logger:
    """
    Create a logger that writes DEBUG+ to a rolling file and INFO+ to stdout.
    Parent directories are created automatically.
    """
    log_file = Path(os.path.expanduser(log_path))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called more than once in the same process
    if logger.handlers:
        return logger

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
