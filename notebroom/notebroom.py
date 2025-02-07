"""Notebroom - Jupyter notebook markdown cleaner using LLMs."""

import os
import sys
import re
import logging
import nbformat
import tiktoken
from tqdm import tqdm
from openai import OpenAI
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize
load_dotenv()
init(autoreset=True)  # Autoreset colorama colors

# Setup basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Rest of configurations
CONFIG = {
    'model': os.getenv('NOTEBROOM_MODEL', 'gpt-4o-mini'),
    'max_tokens': int(os.getenv('NOTEBROOM_MAX_TOKENS', '1000')),
    'keep_recent': int(os.getenv('NOTEBROOM_KEEP_RECENT', '3')),
    'temp': float(os.getenv('NOTEBROOM_TEMPERATURE', '0.2')),
    'window': int(os.getenv('NOTEBROOM_WINDOW_SIZE', '10')),
}

EMOJI = {
    'context': '📚',
    'cell': '📝',
    'original': '⚪',
    'rewritten': '✨',
    'summary': '📑',
    'error': '❌',
    'save': '💾',
}

def log_msg(msg, color=Fore.WHITE, emoji=''):
    """Print colored message to console"""
    print(f"{emoji} {color}{msg}{Style.RESET_ALL}")

def get_context(nb, idx):
    """Get context from previous cells"""
    enc = tiktoken.get_encoding("cl100k_base")
    cells = []
    
    for i in range(max(0, idx - CONFIG['window']), idx):
        cell = nb.cells[i]
        content = [f"{'='*40}", f"{cell.cell_type.upper()} CELL {i}", f"{'='*40}", cell.source.strip()]
        
        if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
            outputs = [out.get('text', out.get('data', {}).get('text/plain', '')) 
                      for out in cell.outputs if 'text' in out or 'data' in out]
            if outputs:
                content.extend([f"{'='*20} OUTPUT {'='*20}", *outputs])
        
        cells.append('\n'.join(content))
    
    context = '\n\n'.join(cells)
    log_msg(f"\nContext for cell {idx}:\n{context}", Fore.BLUE, EMOJI['context'])
    
    if len(enc.encode(context)) > CONFIG['max_tokens']:
        return summarize(context, cells[-CONFIG['keep_recent']:])
    return context

def summarize(old_context, recent_cells):
    """Summarize context if too long"""
    try:
        summary = OpenAI().chat.completions.create(
            model=CONFIG['model'],
            messages=[
                {"role": "system", "content": "Summarize these notebook cells concisely, preserving key concepts and code examples."},
                {"role": "user", "content": old_context}
            ],
            temperature=CONFIG['temp'],
            max_tokens=CONFIG['max_tokens']//2
        ).choices[0].message.content
        return f"{'='*40}\nCONTEXT SUMMARY\n{'='*40}\n{summary}\n\n" + '\n\n'.join(recent_cells)
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        return '\n\n'.join(recent_cells)

def process_notebook(infile, outfile):
    """Process notebook markdown cells"""
    nb = nbformat.read(infile, as_version=4)
    client = OpenAI()
    
    for idx, cell in enumerate(tqdm([c for c in nb.cells if c.cell_type == 'markdown'])):
        try:
            context = get_context(nb, idx)
            log_msg(f"\nProcessing cell {idx} with context:", Fore.CYAN, EMOJI['context'])
            log_msg(context, Fore.BLUE)
            
            new_text = client.chat.completions.create(
                model=CONFIG['model'],
                messages=[
                    {"role": "system", "content": "Rewrite markdown to be maximally educational yet concise, context-aware, and technically accurate."},
                    {"role": "user", "content": f"Context:\n\n{context}\n\nRewrite this:\n\n{cell.source}"}
                ],
                temperature=CONFIG['temp'],
                max_tokens=CONFIG['max_tokens']
            ).choices[0].message.content.strip()
            
            log_msg(f"\nCell #{idx}:", Fore.CYAN, EMOJI['cell'])
            log_msg("Original:", Fore.RED, EMOJI['original'])
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, EMOJI['rewritten'])
            log_msg(new_text, Fore.GREEN)
            log_msg("-" * 80)
            
            cell.source = new_text
        except Exception as e:
            log_msg(f"Error processing cell {idx}: {e}", Fore.RED, EMOJI['error'])
    
    nbformat.write(nb, outfile)
    log_msg(f"Saved to {outfile}", Fore.GREEN, EMOJI['save'])

def main():
    if len(sys.argv) != 2 or not os.getenv("OPENAI_API_KEY"):
        print("Usage: OPENAI_API_KEY=<key> notebroom <notebook.ipynb>")
        sys.exit(1)
    
    process_notebook(sys.argv[1], f"{os.path.splitext(sys.argv[1])[0]}.clean.ipynb")

if __name__ == "__main__":
    main()
