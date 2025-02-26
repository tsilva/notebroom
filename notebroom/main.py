"""Notebroom - Jupyter notebook tool with task-based processing using LLMs."""

import os, sys, logging, argparse, time, random, threading
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Style, init
import nbformat
import tiktoken
from tqdm import tqdm
import re
import subprocess
import os.path
import urllib.parse

# Initialize
load_dotenv()
init(autoreset=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
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

# Utility functions
log_msg = lambda msg, color=Fore.WHITE, emoji='': print(f"{emoji} {color}{msg}{Style.RESET_ALL}")
is_header_only = lambda text: all(line.strip().startswith('#') or not line.strip() for line in text.strip().split('\n'))

enc = tiktoken.get_encoding("cl100k_base")
tpm_lock = threading.Lock()
tokens_used_this_minute = 0
last_minute_start = time.time()

def tpm_throttle(token_count):
    """Throttles requests to stay within TPM limit."""
    global tokens_used_this_minute, last_minute_start
    with tpm_lock: # Use a lock to ensure thread safety
        now = time.time()
        if now - last_minute_start > 60: # Reset counters if a minute has passed
            tokens_used_this_minute = 0
            last_minute_start = now

        if tokens_used_this_minute + token_count > CONFIG['tpm_limit']: # Check if TPM limit is reached
            sleep_time = 60 - (now - last_minute_start)
            if sleep_time > 0:
                log_msg(f"TPM limit reached. Sleeping for {sleep_time:.2f} seconds.", Fore.YELLOW)
                time.sleep(sleep_time)
            tokens_used_this_minute = 0
            last_minute_start = time.time()

        tokens_used_this_minute += token_count

# --- Task Definitions ---
class Task(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def process_cell(self, cell, llm):
        """Process a single notebook cell."""
        pass

    def process_notebook(self, infile, outfile, llm, nb):
        """Process an entire notebook."""
        for cell in nb.cells:
            self.process_cell(cell, llm)

class CleanMarkdownTask(Task):
    """Task for cleaning markdown cells in a Jupyter notebook."""

    REWRITE_PROMPT = """Your task is to make existing educational content more concise and clear.
    Important rules:
    - DO NOT add new information or change meaning.
    - DO NOT modify section headers.
    - FOCUS on making the given text more concise while preserving all information.
    - ENSURE clarity and educational value.
    - MAINTAIN technical accuracy.
    - USE emojis where applicable to increase engagement, but err on the side of not using them.
    Return ONLY the rewritten markdown cell. Do not include any introductory or concluding remarks.
    """.strip()

    def __init__(self, config):
        super().__init__(config)
        self.rpm_limit = int(self.config.get('rpm_limit', '100'))
        self.max_retries = int(self.config.get('max_retries', '5'))
        self.backoff_factor = float(self.config.get('backoff_factor', '3'))
        self.error_throttle_time = int(self.config.get('error_throttle_time', '3'))
        self.num_workers = int(self.config.get('num_workers', '4'))
        self.tpm_limit = int(self.config.get('tpm_limit', '10000000'))

    @limits(calls=100, period=1) # Apply rate limiting
    def call_llm(self, llm, messages):
        """Call the LLM with rate limiting and error handling."""
        try:
            result = llm.invoke(messages) # Invoke the LLM with the given messages
            token_count = len(enc.encode(result.content)) # Estimate token count
            tpm_throttle(token_count) # Throttle based on token count
            return result.content.strip()
        except Exception as e:
            log_msg(f"LLM call failed: {e}. Throttling for {self.error_throttle_time} seconds.", Fore.YELLOW)
            time.sleep(self.error_throttle_time)
            raise e

    @on_exception(expo, RateLimitException, max_tries=5, factor=3, jitter=random.random) # Apply exponential backoff
    def rewrite_cell(self, llm, cell_source):
        """Rewrite a cell using Langchain with retry."""
        try:
            start_time = time.time()
            messages = [{"role": "system", "content": self.REWRITE_PROMPT},
                        {"role": "user", "content": f"Make this text more concise:\n\n{cell_source}"}]
            result = self.call_llm(llm, messages) # Call the LLM to rewrite the cell source
            end_time = time.time()
            logger.info(f"LLM call took {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            return cell_source

    def process_cell(self, cell, llm):
        """Process a single cell."""
        if cell.cell_type != 'markdown':
            return cell

        if is_header_only(cell.source):
            log_msg(f"\nSkipping header cell:\n{cell.source}", Fore.YELLOW, '📌')
            return cell

        log_msg(f"\nProcessing cell:", Fore.CYAN, '📝')
        try:
            new_text = self.rewrite_cell(llm, cell.source)
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, '✨')
            log_msg(new_text, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = new_text
        except Exception as e:
            log_msg(f"Error processing cell: {e}", Fore.RED, '❌')
        return cell

class EmojifyTask(Task):
    """Task for adding emojis to markdown cells using LLMs."""

    REWRITE_PROMPT = """Your task is to add emojis to existing text to make it more engaging.
    Important rules:
    - DO NOT add new information or change meaning.
    - Preserve the original content exactly.
    - Add emojis where they naturally fit to enhance readability and engagement.
    - Use emojis sparingly and appropriately.
    Return ONLY the emojified markdown cell. Do not include any introductory or concluding remarks.
    """.strip()

    def __init__(self, config):
        super().__init__(config)
        self.rpm_limit = int(self.config.get('rpm_limit', '100'))
        self.max_retries = int(self.config.get('max_retries', '5'))
        self.backoff_factor = float(self.config.get('backoff_factor', '3'))
        self.error_throttle_time = int(self.config.get('error_throttle_time', '3'))
        self.num_workers = int(self.config.get('num_workers', '4'))
        self.tpm_limit = int(self.config.get('tpm_limit', '10000000'))

    @limits(calls=100, period=1) # Apply rate limiting
    def call_llm(self, llm, messages):
        """Call the LLM with rate limiting and error handling."""
        try:
            result = llm.invoke(messages) # Invoke the LLM with the given messages
            token_count = len(enc.encode(result.content)) # Estimate token count
            tpm_throttle(token_count) # Throttle based on token count
            return result.content.strip()
        except Exception as e:
            log_msg(f"LLM call failed: {e}. Throttling for {self.error_throttle_time} seconds.", Fore.YELLOW)
            time.sleep(self.error_throttle_time)
            raise e

    @on_exception(expo, RateLimitException, max_tries=5, factor=3, jitter=random.random) # Apply exponential backoff
    def emojify_cell(self, llm, cell_source):
        """Emojify a cell using Langchain with retry."""
        try:
            start_time = time.time()
            messages = [{"role": "system", "content": self.REWRITE_PROMPT},
                        {"role": "user", "content": f"Add emojis to this text:\n\n{cell_source}"}]
            result = self.call_llm(llm, messages) # Call the LLM to emojify the cell source
            end_time = time.time()
            logger.info(f"LLM call took {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Emojify failed: {e}")
            return cell_source

    def process_cell(self, cell, llm):
        """Process a single cell."""
        if cell.cell_type != 'markdown':
            return cell

        log_msg(f"\nEmojifying cell:", Fore.CYAN, '📝')
        try:
            new_text = self.emojify_cell(llm, cell.source)
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, '✨')
            log_msg(new_text, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = new_text
        except Exception as e:
            log_msg(f"Error processing cell: {e}", Fore.RED, '❌')
        return cell

class FixColabLinks(Task):
    """Task for fixing 'Open in Colab' links to point to the correct GitHub repository."""

    def __init__(self, config):
        super().__init__(config)
        self.notebook_path = None
        self.repo_info = None

    def find_git_root(self, path):
        """Find the root directory of the Git repository containing the given path."""
        current = os.path.abspath(path)
        while current != '/':
            if os.path.exists(os.path.join(current, '.git')):
                return current
            current = os.path.dirname(current)
        return None

    def get_repo_info(self, notebook_path):
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

            # Extract username and repo name from URL
            # Handle different URL formats (SSH or HTTPS)
            if remote_url.startswith('git@'):
                # SSH format: git@github.com:username/repo.git
                match = re.match(r'git@github\.com:([^/]+)/([^.]+)\.?.*', remote_url)
                if match:
                    username, repo = match.groups()
            else:
                # HTTPS format: https://github.com/username/repo.git
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

    def create_colab_url(self, notebook_path):
        """Create a correct 'Open in Colab' URL for a notebook file."""
        repo_info = self.get_repo_info(notebook_path)
        if not repo_info:
            return None
        
        # Create the Colab URL with the format:
        # https://colab.research.google.com/github/{username}/{repo}/blob/main/{path}
        encoded_path = urllib.parse.quote(repo_info['rel_path'])
        url = f"https://colab.research.google.com/github/{repo_info['username']}/{repo_info['repo']}/blob/main/{encoded_path}"
        return url

    def fix_colab_links(self, cell_source, notebook_path):
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
                if "![" in full_match:  # Image link pattern
                    old_link = match.group(2)
                else:  # Simple link pattern
                    old_link = match.group(1)
                    
                # Create replacement with same format but updated URL
                replacement = full_match.replace(old_link, colab_url)
                modified_source = modified_source.replace(full_match, replacement)
                
        return modified_source

    def process_cell(self, cell, llm=None):
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

    def process_notebook(self, infile, outfile, llm, nb):
        """Process an entire notebook."""
        self.notebook_path = os.path.abspath(infile)
        super().process_notebook(infile, outfile, llm, nb)

# --- Utility functions ---
def find_notebooks(directory):
    """Find all .ipynb files in a directory, recursively."""
    notebooks = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb'):
                notebooks.append(os.path.join(root, file))
    return notebooks

def process_notebook(infile, task_name, output=None):
    """Process a single notebook with the given task."""
    # Determine output file path
    base, ext = os.path.splitext(infile)
    if output:
        if os.path.isdir(output):
            # If output is a directory, place the file there with original name
            outfile = os.path.join(output, os.path.basename(infile))
        else:
            # Otherwise use the specified output file
            outfile = output
    else:
        # Default behavior: append .clean to filename
        outfile = f"{base}.clean{ext}"
    
    # Initialize LLM (only if needed)
    llm = None
    if task_name == "clean_markdown" or task_name == "emojify":
        model_name = CONFIG['model']
        model_params = {
            'temperature': float(CONFIG['temp']),
            'max_tokens': int(CONFIG['max_tokens'])
        }

        if "gemini" in model_name:
            llm = ChatVertexAI(model_name=model_name, convert_system_message_to_human=True, **model_params)
        else:
            llm = ChatOpenAI(model_name=model_name, **model_params)

    # Load the task
    if task_name == "clean_markdown":
        task = CleanMarkdownTask(CONFIG)
    elif task_name == "emojify":
        task = EmojifyTask(CONFIG)
    elif task_name == "fix_colab_links":
        task = FixColabLinks(CONFIG)
    else:
        print(f"Error: Unknown task '{task_name}'.")
        return

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
        task.process_notebook(infile, outfile, llm, nb)
    else:
        with ThreadPoolExecutor(max_workers=int(CONFIG['num_workers'])) as executor:
            futures = [executor.submit(task.process_cell, cell, llm) for cell in cells]
            results = [future.result() for future in tqdm(futures, total=len(cells))]
        # Update notebook with processed cells
        nb.cells = results

    # Write the notebook
    try:
        nbformat.write(nb, outfile)
        log_msg(f"Processed notebook saved to {outfile}", Fore.GREEN, '💾')
    except Exception as e:
        log_msg(f"Error saving notebook {outfile}: {e}", Fore.RED, '❌')

# --- Main Program ---
def main():
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
