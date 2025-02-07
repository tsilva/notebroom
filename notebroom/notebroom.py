"""Notebroom - A tool to clean up Jupyter notebook markdown cells."""

__version__ = "0.1.0"

from dotenv import load_dotenv
load_dotenv()

import os
import re
import sys
import nbformat
from tqdm import tqdm
from openai import OpenAI
from colorama import Fore, Style, init

init()  # Initialize colorama

SYSTEM_PROMPT = """
Rewrite markdown text to be more concise and clear. Preserve meaning and formatting. Return only the revised markdown.
""".strip()

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

def rewrite_text(text):
    """Rewrite text using LLM"""
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return resp.choices[0].message.content.strip()

def get_output_filename(input_path):
    """Generate output filename by adding 'clean' before the extension."""
    base, ext = os.path.splitext(input_path)
    return f"{base}.clean{ext}"

def process_nb(infile, outfile):
    """Process notebook markdown cells"""
    nb = nbformat.read(infile, as_version=4)
    
    md_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'markdown']
    
    for idx, cell in tqdm(md_cells, desc="Processing"):
        clean = clean_md(cell.source)
        if clean.strip():
            new_text = rewrite_text(clean)
            log_change(idx, cell.source, new_text)
            cell.source = new_text
            
    nbformat.write(nb, outfile)
    print(f"{Fore.GREEN}Saved to {outfile}{Style.RESET_ALL}")

def main():
    if len(sys.argv) != 2:
        print("Usage: notebroom <notebook.ipynb>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = get_output_filename(input_file)
    process_nb(input_file, output_file)

if __name__ == "__main__":
    main()
