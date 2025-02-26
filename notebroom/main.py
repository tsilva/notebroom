"""Notebroom - Jupyter notebook tool with task-based processing using LLMs."""

import os
import sys
import logging
import argparse
import time
import random
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import re
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import BaseMessage
from ratelimit import limits
from backoff import on_exception, expo
from colorama import Fore, Style, init
import nbformat
from nbformat import NotebookNode
import tiktoken
from tqdm import tqdm

# Initialize environment
load_dotenv()
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class Config:
    """Configuration management for Notebroom."""
    
    def __init__(self):
        self.values = {
            'model': os.getenv('NOTEBROOM_MODEL', 'gpt-4o-mini'),
            'max_tokens': int(os.getenv('NOTEBROOM_MAX_TOKENS', '1000')),
            'keep_recent': int(os.getenv('NOTEBROOM_KEEP_RECENT', '3')),
            'temp': float(os.getenv('NOTEBROOM_TEMPERATURE', '0.2')),
            'tpm_limit': int(os.getenv('NOTEBROOM_TPM_LIMIT', '10000000')),
            'rpm_limit': int(os.getenv('NOTEBROOM_RPM_LIMIT', '100')),
            'max_retries': int(os.getenv('NOTEBROOM_MAX_RETRIES', '5')),
            'backoff_factor': float(os.getenv('NOTEBROOM_BACKOFF_FACTOR', '3')),
            'error_throttle_time': int(os.getenv('NOTEBROOM_ERROR_THROTTLE_TIME', '3')),
            'num_workers': int(os.getenv('NOTEBROOM_NUM_WORKERS', '4'))
        }
    
    def __getitem__(self, key: str) -> Any:
        return self.values.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class TokenRateLimiter:
    """Manages token rate limiting to stay within API limits."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tpm_lock = threading.Lock()
        self.tokens_used_this_minute = 0
        self.last_minute_start = time.time()
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def throttle(self, token_count: int) -> None:
        """Throttle requests to stay within TPM limit."""
        with self.tpm_lock:
            now = time.time()
            if now - self.last_minute_start > 60:
                self.tokens_used_this_minute = 0
                self.last_minute_start = now
                
            if self.tokens_used_this_minute + token_count > self.config['tpm_limit']:
                sleep_time = 60 - (now - self.last_minute_start)
                if sleep_time > 0:
                    log_msg(f"TPM limit reached. Sleeping for {sleep_time:.2f} seconds.", Fore.YELLOW)
                    time.sleep(sleep_time)
                self.tokens_used_this_minute = 0
                self.last_minute_start = time.time()
                
            self.tokens_used_this_minute += token_count
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        return len(self.encoder.encode(text))


class LLMService:
    """Manages LLM interactions with rate limiting and error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.rate_limiter = TokenRateLimiter(config)
        
        # Initialize the LLM model
        model_name = config['model']
        model_params = {
            'temperature': config['temp'],
            'max_tokens': config['max_tokens']
        }
        
        if "gemini" in model_name:
            self.llm = ChatVertexAI(model_name=model_name, 
                                   convert_system_message_to_human=True, 
                                   **model_params)
        else:
            self.llm = ChatOpenAI(model_name=model_name, **model_params)
    
    @limits(calls=100, period=60)
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with rate limiting and error handling."""
        try:
            result = self.llm.invoke(messages)
            token_count = self.rate_limiter.estimate_tokens(result.content)
            self.rate_limiter.throttle(token_count)
            return result.content.strip()
        except Exception as e:
            log_msg(f"LLM call failed: {e}. Throttling for {self.config['error_throttle_time']} seconds.", 
                   Fore.YELLOW)
            time.sleep(self.config['error_throttle_time'])
            raise e
    
    @on_exception(expo, Exception, max_tries=5, factor=3, jitter=random.random)
    def process_text(self, system_prompt: str, user_text: str) -> str:
        """Process text using the LLM with retries and backoff."""
        try:
            start_time = time.time()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
            result = self.call_llm(messages)
            end_time = time.time()
            logger.info(f"LLM call took {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return user_text


# Utility functions
def log_msg(msg: str, color=Fore.WHITE, emoji: str = '') -> None:
    """Print a formatted log message with optional color and emoji."""
    print(f"{emoji} {color}{msg}{Style.RESET_ALL}")


def is_header_only(text: str) -> bool:
    """Check if a text contains only headers and whitespace."""
    return all(line.strip().startswith('#') or not line.strip() for line in text.strip().split('\n'))


def find_notebooks(directory: str) -> List[str]:
    """Find all .ipynb files in a directory, recursively."""
    return list(str(p) for p in Path(directory).glob('**/*.ipynb'))


class Task(ABC):
    """Base class for notebook processing tasks."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    def process_cell(self, cell: NotebookNode, llm_service: Optional[LLMService] = None) -> NotebookNode:
        """Process a single notebook cell."""
        pass
    
    def process_notebook(self, infile: str, outfile: str, 
                         llm_service: Optional[LLMService], nb: NotebookNode) -> None:
        """Process an entire notebook."""
        for cell in nb.cells:
            self.process_cell(cell, llm_service)


class TextProcessingTask(Task):
    """Base class for text-processing tasks that use LLMs."""
    
    def __init__(self, config: Config, system_prompt: str):
        super().__init__(config)
        self.system_prompt = system_prompt
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional[LLMService] = None) -> NotebookNode:
        """Process a markdown cell using the LLM service."""
        if cell.cell_type != 'markdown' or llm_service is None:
            return cell
            
        if is_header_only(cell.source):
            log_msg(f"\nSkipping header cell:\n{cell.source}", Fore.YELLOW, '📌')
            return cell
        
        log_msg(f"\nProcessing cell:", Fore.CYAN, '📝')
        try:
            new_text = llm_service.process_text(self.system_prompt, cell.source)
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, '✨')
            log_msg(new_text, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = new_text
        except Exception as e:
            log_msg(f"Error processing cell: {e}", Fore.RED, '❌')
        
        return cell


class CleanMarkdownTask(TextProcessingTask):
    """Task for cleaning markdown cells in a Jupyter notebook."""
    
    PROMPT = """Your task is to make existing educational content more concise and clear.
    Important rules:
    - DO NOT add new information or change meaning.
    - DO NOT modify section headers.
    - FOCUS on making the given text more concise while preserving all information.
    - ENSURE clarity and educational value.
    - MAINTAIN technical accuracy.
    - USE emojis where applicable to increase engagement, but err on the side of not using them.
    Return ONLY the rewritten markdown cell. Do not include any introductory or concluding remarks.
    """.strip()
    
    def __init__(self, config: Config):
        super().__init__(config, self.PROMPT)


class EmojifyTask(TextProcessingTask):
    """Task for adding emojis to markdown cells using LLMs."""
    
    PROMPT = """Your task is to add emojis to existing text to make it more engaging.
    Important rules:
    - DO NOT add new information or change meaning.
    - Preserve the original content exactly.
    - Add emojis where they naturally fit to enhance readability and engagement.
    - Use emojis sparingly and appropriately.
    Return ONLY the emojified markdown cell. Do not include any introductory or concluding remarks.
    """.strip()
    
    def __init__(self, config: Config):
        super().__init__(config, self.PROMPT)


class FixColabLinks(Task):
    """Task for fixing 'Open in Colab' links to point to the correct GitHub repository."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.notebook_path = None
        self.repo_info = None
    
    def find_git_root(self, path: str) -> Optional[str]:
        """Find the root directory of the Git repository containing the given path."""
        current = os.path.abspath(path)
        while current != '/':
            if os.path.exists(os.path.join(current, '.git')):
                return current
            current = os.path.dirname(current)
        return None
    
    def get_repo_info(self, notebook_path: str) -> Optional[Dict[str, str]]:
        """Get the GitHub repository information for a notebook file."""
        if self.repo_info:
            return self.repo_info
            
        git_root = self.find_git_root(os.path.dirname(notebook_path))
        if not git_root:
            log_msg("Could not find Git repository", Fore.RED, '❌')
            return None
            
        try:
            # Get remote URL
            remote_url = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'],
                cwd=git_root, text=True
            ).strip()
            
            # Extract username and repo name
            if remote_url.startswith('git@'):
                match = re.match(r'git@github\.com:([^/]+)/([^.]+)\.?.*', remote_url)
                if match:
                    username, repo = match.groups()
            else:
                match = re.match(r'https?://github\.com/([^/]+)/([^.]+)\.?.*', remote_url)
                if match:
                    username, repo = match.groups()
                else:
                    log_msg(f"Could not parse GitHub URL: {remote_url}", Fore.RED, '❌')
                    return None
                    
            # Get relative path from repo root to notebook
            rel_path = os.path.relpath(notebook_path, git_root)
            
            self.repo_info = {
                'username': username,
                'repo': repo,
                'root': git_root,
                'rel_path': rel_path
            }
            return self.repo_info
            
        except subprocess.CalledProcessError:
            log_msg("Failed to get Git repository info", Fore.RED, '❌')
            return None
    
    def create_colab_url(self, notebook_path: str) -> Optional[str]:
        """Create a correct 'Open in Colab' URL for a notebook file."""
        repo_info = self.get_repo_info(notebook_path)
        if not repo_info:
            return None
            
        encoded_path = urllib.parse.quote(repo_info['rel_path'])
        url = (f"https://colab.research.google.com/github/{repo_info['username']}/"
               f"{repo_info['repo']}/blob/main/{encoded_path}")
        return url
    
    def fix_colab_links(self, cell_source: str, notebook_path: str) -> str:
        """Fix 'Open in Colab' links in a markdown cell."""
        if not self.notebook_path:
            self.notebook_path = notebook_path
            
        # Patterns for matching Colab links
        patterns = [
            r'\[(?:Open|Run|View) (?:in|on) Colab\]\((https?://colab\.research\.google\.com/[^\)]+)\)',
            r'\[\!\[(?:Open|Run|View) (?:in|on) Colab\]\(https?://[^\)]+\)\]\((https?://colab\.research\.google\.com/[^\)]+)\)'
        ]
        
        colab_url = self.create_colab_url(notebook_path)
        if not colab_url:
            return cell_source
            
        modified_source = cell_source
        for pattern in patterns:
            matches = re.finditer(pattern, cell_source)
            for match in matches:
                full_match = match.group(0)
                # Different groups for different patterns
                old_link = match.group(2) if "![" in full_match else match.group(1)
                    
                # Create replacement with same format but updated URL
                replacement = full_match.replace(old_link, colab_url)
                modified_source = modified_source.replace(full_match, replacement)
                
        return modified_source
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional[LLMService] = None) -> NotebookNode:
        """Process a single cell."""
        if cell.cell_type != 'markdown':
            return cell
            
        if not any(pattern in cell.source.lower() for pattern in ['colab', 'open in']):
            return cell
            
        log_msg(f"\nChecking cell for Colab links:", Fore.CYAN, '🔍')
        try:
            new_text = self.fix_colab_links(cell.source, self.notebook_path)
            if new_text != cell.source:
                log_msg("Original:", Fore.RED, '📄')
                log_msg(cell.source, Fore.RED)
                log_msg("Fixed:", Fore.GREEN, '✨')
                log_msg(new_text, Fore.GREEN)
                log_msg("-" * 80)
                cell.source = new_text
            else:
                log_msg("No Colab links to fix in this cell", Fore.YELLOW)
        except Exception as e:
            log_msg(f"Error fixing Colab links: {e}", Fore.RED, '❌')
        return cell
    
    def process_notebook(self, infile: str, outfile: str, 
                         llm_service: Optional[LLMService], nb: NotebookNode) -> None:
        """Process an entire notebook."""
        self.notebook_path = os.path.abspath(infile)
        super().process_notebook(infile, outfile, llm_service, nb)


def process_notebook(infile: str, task_name: str, output: Optional[str] = None) -> None:
    """Process a single notebook with the given task."""
    # Determine output file path
    base, ext = os.path.splitext(infile)
    if output:
        if os.path.isdir(output):
            outfile = os.path.join(output, os.path.basename(infile))
        else:
            outfile = output
    else:
        outfile = f"{base}.clean{ext}"
    
    # Initialize configuration
    config = Config()
    
    # Initialize LLM service if needed
    llm_service = None
    if task_name in ("clean_markdown", "emojify"):
        llm_service = LLMService(config)
    
    # Load the task
    task_map = {
        "clean_markdown": CleanMarkdownTask(config),
        "emojify": EmojifyTask(config),
        "fix_colab_links": FixColabLinks(config)
    }
    
    if task_name not in task_map:
        print(f"Error: Unknown task '{task_name}'.")
        return
        
    task = task_map[task_name]
    
    # Load notebook
    try:
        nb = nbformat.read(infile, as_version=4)
        cells = nb.cells
    except Exception as e:
        log_msg(f"Error loading notebook {infile}: {e}", Fore.RED, '❌')
        return
        
    log_msg(f"Processing {infile} with task '{task_name}'...", Fore.CYAN)
    
    # Process cells
    if task_name == "fix_colab_links":
        task.process_notebook(infile, outfile, llm_service, nb)
    else:
        with ThreadPoolExecutor(max_workers=config['num_workers']) as executor:
            futures = [executor.submit(task.process_cell, cell, llm_service) for cell in cells]
            results = [future.result() for future in tqdm(futures, total=len(cells))]
        # Update notebook with processed cells
        nb.cells = results
        
    # Write the notebook
    try:
        nbformat.write(nb, outfile)
        log_msg(f"Processed notebook saved to {outfile}", Fore.GREEN, '💾')
    except Exception as e:
        log_msg(f"Error saving notebook {outfile}: {e}", Fore.RED, '❌')


def main():
    """Main entry point for the Notebroom CLI."""
    available_tasks = ["clean_markdown", "emojify", "fix_colab_links"]
    parser = argparse.ArgumentParser(
        description="Jupyter notebook tool with task-based processing using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("notebook", help="Path to the input notebook file or directory containing notebooks.")
    parser.add_argument(
        "task",
        help="Name of the task to execute. Available tasks:\n" + "\n".join(available_tasks),
        nargs='?',
        default=None
    )
    parser.add_argument("-o", "--output", help="Path to the output notebook file or directory.", default=None)
    args = parser.parse_args()
    
    if not args.task:
        parser.print_help()
        sys.exit(1)
        
    if args.task not in available_tasks:
        print(f"Error: Unknown task '{args.task}'. Available tasks are: {', '.join(available_tasks)}")
        sys.exit(1)
        
    if not os.getenv("OPENAI_API_KEY") and args.task == "clean_markdown":
        print("Error: OPENAI_API_KEY environment variable must be set for the clean_markdown task.")
        sys.exit(1)
        
    infile = args.notebook
    task_name = args.task
    
    # Check if infile is a directory
    if os.path.isdir(infile):
        notebooks = find_notebooks(infile)
        if not notebooks:
            print(f"No .ipynb files found in {infile}")
            sys.exit(1)
            
        # Prompt for confirmation
        print(f"Found {len(notebooks)} notebook files in {infile}:")
        for nb in notebooks[:5]:  # Show first 5 notebooks
            print(f" - {os.path.basename(nb)}")
        if len(notebooks) > 5:
            print(f" ... and {len(notebooks) - 5} more")
            
        confirm = input(f"Process all {len(notebooks)} notebooks with task '{task_name}'? [y/N] ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
                
        # Check output option for multiple notebooks
        if args.output and not os.path.isdir(args.output):
            print("Error: When processing multiple notebooks, output (-o) must be a directory.")
            sys.exit(1)
                
        # Process each notebook
        for nb_file in notebooks:
            process_notebook(nb_file, task_name, args.output)
    else:
        # Process single notebook
        process_notebook(infile, task_name, args.output)


if __name__ == "__main__":
    main()
