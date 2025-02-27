"""Utility functions for Notebroom."""

import os
from pathlib import Path
from typing import List
from colorama import Fore, Style


def log_msg(msg: str, color=Fore.WHITE, emoji: str = '') -> None:
    """Print a formatted log message with optional color and emoji."""
    print(f"{emoji} {color}{msg}{Style.RESET_ALL}")


def is_header_only(text: str) -> bool:
    """Check if a text contains only headers and whitespace."""
    return all(line.strip().startswith('#') or not line.strip() for line in text.strip().split('\n'))


def find_notebooks(directory: str) -> List[str]:
    """Find all .ipynb files in a directory, recursively."""
    return list(str(p) for p in Path(directory).glob('**/*.ipynb'))
