import re
from colorama import Fore, Style

def clean_md(text):
    """Basic markdown cleanup"""
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    text = re.sub(r'^( *[-+*]) +', r'\1 ', text, flags=re.MULTILINE)
    text = re.sub(r'^(#+) *(.+)$', r'\1 \2', text, flags=re.MULTILINE)
    return text

def log_change(idx, old, new):
    """Log a cell change with colors"""
    print(f"\n{Fore.CYAN}Cell #{idx}:{Style.RESET_ALL}")
    print(f"{Fore.RED}Original:{Style.RESET_ALL}\n{old}")
    print(f"\n{Fore.GREEN}Rewritten:{Style.RESET_ALL}\n{new}\n")
    print("-" * 80)
