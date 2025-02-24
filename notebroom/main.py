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

# --- Main Program ---
def main():
    available_tasks = ["clean_markdown", "emojify"]
    parser = argparse.ArgumentParser(
        description="Jupyter notebook tool with task-based processing using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("notebook", help="Path to the input notebook file.")
    parser.add_argument(
        "task",
        help="Name of the task to execute. Available tasks:\n" + "\n".join(available_tasks),
        nargs='?',
        default=None
    )
    parser.add_argument("-o", "--output", help="Path to the output notebook file.", default=None)
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
    base, ext = os.path.splitext(infile)
    outfile = args.output if args.output else f"{base}.clean{ext}"

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
    else:
        print(f"Error: Unknown task '{task_name}'.")
        sys.exit(1)

    # Load notebook
    nb = nbformat.read(infile, as_version=4)
    cells = nb.cells

    # Process cells in parallel
    with ThreadPoolExecutor(max_workers=int(CONFIG['num_workers'])) as executor:
        futures = [executor.submit(task.process_cell, cell, llm) for cell in cells]
        results = [future.result() for future in tqdm(futures, total=len(cells))]

    # Update notebook with processed cells
    nb.cells = results

    # Write the notebook
    nbformat.write(nb, outfile)
    log_msg(f"Processed notebook saved to {outfile}", Fore.GREEN, '💾')

if __name__ == "__main__":
    main()
