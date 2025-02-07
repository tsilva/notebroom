from dotenv import load_dotenv
load_dotenv()

import os
from . import utils
from .llm import rewrite_text
import nbformat
from tqdm import tqdm
from colorama import Fore, Style

def get_output_filename(input_path):
    """Generate output filename by adding 'clean' before the extension."""
    base, ext = os.path.splitext(input_path)
    return f"{base}.clean{ext}"

def process_nb(infile, outfile):
    """Process notebook markdown cells"""
    nb = nbformat.read(infile, as_version=4)
    
    md_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == 'markdown']
    
    for idx, cell in tqdm(md_cells, desc="Processing"):
        clean = utils.clean_md(cell.source)
        if clean.strip():
            new_text = rewrite_text(clean)
            utils.log_change(idx, cell.source, new_text)
            cell.source = new_text
            
    nbformat.write(nb, outfile)
    print(f"{Fore.GREEN}Saved to {outfile}{Style.RESET_ALL}")
