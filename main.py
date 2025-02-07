from dotenv import load_dotenv
load_dotenv()

import colorama
from colorama import Fore, Style
import openai

import os
import re
import argparse
import nbformat
from tqdm import tqdm

colorama.init()

SYSTEM_PROMPT = """
Rewrite markdown text to be more concise and clear. Preserve meaning and formatting. Return only the revised markdown.
""".strip()

def rewrite(text, client):
    """Rewrite text using LLM"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return resp.choices[0].message.content.strip()

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

def process_nb(infile, outfile):
    """Process notebook markdown cells"""
    nb = nbformat.read(infile, as_version=4)
    client = openai.OpenAI()
    
    md_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'markdown']
    
    for idx, cell in tqdm(md_cells, desc="Processing"):
        clean = clean_md(cell.source)
        if clean.strip():
            new_text = rewrite(clean, client)
            log_change(idx, cell.source, new_text)
            cell.source = new_text
            
    nbformat.write(nb, outfile)
    print(f"{Fore.GREEN}Saved to {outfile}{Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean up notebook markdown.')
    parser.add_argument('input', help='Input notebook')
    args = parser.parse_args()
    
    outfile = f"{os.path.splitext(args.input)[0]}.clean.ipynb"
    process_nb(args.input, outfile)