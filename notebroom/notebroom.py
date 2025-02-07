"""Notebroom - Jupyter notebook markdown cleaner using LLMs."""

import os
import sys
import re
import logging
import nbformat
import tiktoken
from tqdm import tqdm
from colorama import Fore, Style, init
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import create_extraction_chain

# Initialize
load_dotenv()
init(autoreset=True)

# Setup basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Rest of configurations
CONFIG = {
    'model': os.getenv('MODEL', 'gpt-4o-mini'),
    'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
    'keep_recent': int(os.getenv('KEEP_RECENT', '3')),
    'temp': float(os.getenv('TEMPERATURE', '0.2')),
    'window': int(os.getenv('WINDOW_SIZE', '10')),
}

def log_msg(msg, color=Fore.WHITE, emoji=''):
    """Print colored message to console"""
    print(f"{emoji} {color}{msg}{Style.RESET_ALL}")

def is_header_only(text):
    """Check if markdown cell only contains headers"""
    lines = text.strip().split('\n')
    return all(line.strip().startswith('#') or not line.strip() for line in lines)

# Prompts for LLM interactions
REWRITE_PROMPT = """
Your task is to make existing educational content more concise and clear.
Important rules:
- DO NOT add new information or change meaning
- DO NOT modify section headers
- FOCUS on making text more concise while preserving all information
- ENSURE clarity and educational value
- MAINTAIN technical accuracy
- USE context to avoid redundancy
""".strip()

SUMMARY_PROMPT = """
Summarize these notebook cells concisely, preserving key concepts and progression.
Important aspects:
- Maintain the educational flow
- Keep critical code examples and their purpose
- Preserve technical accuracy and terminology
- Create a flowing narrative that connects concepts
- Focus on relationships between ideas
""".strip()

class NotebookProcessor:
    def __init__(self, model=CONFIG['model']):
        self.llm = ChatOpenAI(model_name=model, temperature=CONFIG['temp'], max_tokens=CONFIG['max_tokens'])
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_PROMPT),
            HumanMessagePromptTemplate.from_template("{content}")
        ])
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITE_PROMPT),
            HumanMessagePromptTemplate.from_template("Context:\n\n{context}\n\nMake this text more concise:\n\n{text}")
        ])
        # Use .invoke instead of chains
        self.summary_chain = self.summary_prompt | self.llm
        self.rewrite_chain = self.rewrite_prompt | self.llm
        self.summary_cache = {}  # Store running summaries
        self.last_summary_idx = -1  # Track where we last summarized
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def get_context(self, nb, current_idx):
        """Get context with running summary management"""
        all_context = []
        
        # Add existing summary if we have one
        if self.last_summary_idx >= 0:
            all_context.append(
                f"{'='*40}\nSUMMARY OF CELLS 0-{self.last_summary_idx}\n{'='*40}\n"
                f"{self.summary_cache[self.last_summary_idx]}\n"
            )
        
        # Add cells since last summary
        recent_cells = []
        for i in range(self.last_summary_idx + 1, current_idx):
            cell = nb.cells[i]
            content = [
                f"{'='*40}",
                f"{cell.cell_type.upper()} CELL {i}",
                f"{'='*40}",
                cell.source.strip()
            ]
            
            if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                outputs = [out.get('text', out.get('data', {}).get('text/plain', '')) 
                          for out in cell.outputs if 'text' in out or 'data' in out]
                if outputs:
                    content.extend([f"{'='*20} OUTPUT {'='*20}", *outputs])
            
            recent_cells.append('\n'.join(content))
        
        # If context is too large, summarize up to this point
        all_cells = '\n\n'.join(recent_cells)
        if len(self.enc.encode(all_cells)) > CONFIG['max_tokens']:
            summary = self.summarize_cells('\n\n'.join([
                self.summary_cache.get(self.last_summary_idx, ''),
                all_cells
            ]))
            self.summary_cache[current_idx - 1] = summary
            self.last_summary_idx = current_idx - 1
            
            # Keep only the most recent cells
            recent_cells = recent_cells[-CONFIG['keep_recent']:]
        
        all_context.extend(recent_cells)
        return '\n\n'.join(all_context)
    
    def summarize_cells(self, content):
        """Create or extend summary using Langchain"""
        try:
            result = self.summary_chain.invoke({'content': content})
            return f"[Previous Content Summary]\n{result.content.strip()}"
        except Exception as e:
            logger.error(f"Summary failed: {e}")
            return "Summary generation failed"

    def rewrite_cell(self, context, cell_source):
        """Rewrite a cell using Langchain"""
        try:
            result = self.rewrite_chain.invoke({'context': context, 'text': cell_source})
            return result.content.strip()
        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            return cell_source

def process_notebook(infile, outfile):
    """Process notebook markdown cells"""
    nb = nbformat.read(infile, as_version=4)
    processor = NotebookProcessor()
    
    for idx, cell in enumerate(tqdm([c for c in nb.cells if c.cell_type == 'markdown'])):
        try:
            if is_header_only(cell.source):
                log_msg(f"\nSkipping header cell #{idx}:", Fore.YELLOW, '📌')
                log_msg(cell.source, Fore.YELLOW)
                continue
                
            context = processor.get_context(nb, idx)
            log_msg(f"\nProcessing cell {idx}:", Fore.CYAN, '📝')
            
            new_text = processor.rewrite_cell(context, cell.source)
            
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, '✨')
            log_msg(new_text, Fore.GREEN)
            log_msg("-" * 80)
            
            cell.source = new_text
        except Exception as e:
            log_msg(f"Error processing cell {idx}: {e}", Fore.RED, '❌')
    
    nbformat.write(nb, outfile)
    log_msg(f"Saved to {outfile}", Fore.GREEN, '💾')

def main():
    if len(sys.argv) != 2 or not os.getenv("OPENAI_API_KEY"):
        print("Usage: OPENAI_API_KEY=<key> notebroom <notebook.ipynb>")
        sys.exit(1)
    
    process_notebook(sys.argv[1], f"{os.path.splitext(sys.argv[1])[0]}.clean.ipynb")

if __name__ == "__main__":
    main()
