"""Notebroom - Jupyter notebook tool with task-based processing using LLMs."""

import os
import sys
import logging
import argparse
import time
import threading
import requests
from typing import Dict, List, Optional, Any
import yaml

from dotenv import load_dotenv
from colorama import Fore, init
import nbformat
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from notebroom.tasks import TASK_MAP, AVAILABLE_TASKS, LLM_REQUIRED_TASKS
from notebroom.utils import log_msg, find_notebooks

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
            'model': os.getenv('MODEL', 'anthropic/claude-3-7-sonnet'),
            'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
            'keep_recent': int(os.getenv('KEEP_RECENT', '3')),
            'temp': float(os.getenv('TEMPERATURE', '0.2')),
            'tpm_limit': int(os.getenv('TPM_LIMIT', '100000')),
            'rpm_limit': int(os.getenv('RPM_LIMIT', '60')),
            'max_retries': int(os.getenv('MAX_RETRIES', '5')),
            'backoff_factor': float(os.getenv('BACKOFF_FACTOR', '3')),
            'error_throttle_time': int(os.getenv('ERROR_THROTTLE_TIME', '3')),
            'num_workers': int(os.getenv('NUM_WORKERS', '8')),
            'batch_size': int(os.getenv('BATCH_SIZE', '4')),
            'openrouter_api_base': os.getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1'),
            'openrouter_api_key': os.getenv('OPENROUTER_API_KEY', '')
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
        self.rpm_lock = threading.Lock()
        self.tokens_used_this_minute = 0
        self.requests_this_minute = 0
        self.last_minute_start = time.time()
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def throttle(self, token_count: int) -> None:
        """Throttle requests to stay within TPM and RPM limits."""
        now = time.time()
        
        # Check if we're in a new minute
        if now - self.last_minute_start > 60:
            with self.tpm_lock:
                self.tokens_used_this_minute = 0
                self.last_minute_start = now
            with self.rpm_lock:
                self.requests_this_minute = 0
        
        # Check TPM limit
        with self.tpm_lock:
            if self.tokens_used_this_minute + token_count > self.config['tpm_limit']:
                sleep_time = 60 - (now - self.last_minute_start)
                if sleep_time > 0:
                    log_msg(f"TPM limit reached. Sleeping for {sleep_time:.2f} seconds.", Fore.YELLOW)
                    time.sleep(sleep_time)
                self.tokens_used_this_minute = 0
                self.last_minute_start = time.time()
            self.tokens_used_this_minute += token_count
        
        # Check RPM limit
        with self.rpm_lock:
            if self.requests_this_minute + 1 > self.config['rpm_limit']:
                sleep_time = 60 - (now - self.last_minute_start)
                if sleep_time > 0:
                    log_msg(f"RPM limit reached. Sleeping for {sleep_time:.2f} seconds.", Fore.YELLOW)
                    time.sleep(sleep_time)
                self.requests_this_minute = 0
                self.last_minute_start = time.time()
            self.requests_this_minute += 1
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        return len(self.encoder.encode(text))


class OpenRouterLLMService:
    """Manages LLM interactions via OpenRouter with rate limiting and error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.rate_limiter = TokenRateLimiter(config)
        
        # Check if OpenRouter API key is configured
        if not config['openrouter_api_key']:
            log_msg("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.", 
                   Fore.RED, '❌')
            raise ValueError("OpenRouter API key not configured")
        
        self.api_base = config['openrouter_api_base']
        self.model = config['model']
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['openrouter_api_key']}",
            "HTTP-Referer": "https://github.com/tsilva/notebroom",  # Identify your application
            "X-Title": "Notebroom"  # Application name
        }
        
        # Create a session for connection pooling
        self.session = requests.Session()
        for adapter in self.session.adapters.values():
            # Increase connection pool size for better throughput
            adapter.max_retries = 3
            adapter.pool_connections = 20
            adapter.pool_maxsize = 20

        # Initialize the semaphore for parallel request limiting
        self.semaphore = threading.Semaphore(min(config['batch_size'], 8))
    
    def call_llm(self, messages: List[Dict[str, str]], retry_count=0) -> str:
        """Call the LLM with rate limiting and error handling."""
        # Estimate token usage for rate limiting
        message_text = " ".join([msg["content"] for msg in messages])
        input_tokens = self.rate_limiter.estimate_tokens(message_text)
        
        # Apply rate limiting before making the request
        self.rate_limiter.throttle(input_tokens)
        
        try:
            with self.semaphore:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.config['max_tokens'],
                    "temperature": self.config['temp']
                }
                
                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30  # Add timeout to prevent hanging requests
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter API error: HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg += f" - {error_data['error'].get('message', '')}"
                    except:
                        error_msg += f" - {response.text}"
                    
                    # Handle rate limiting with exponential backoff
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', self.config['error_throttle_time']))
                        log_msg(f"Rate limited. Waiting for {retry_after} seconds.", Fore.YELLOW)
                        time.sleep(retry_after)
                        
                        # Retry with exponential backoff
                        if retry_count < self.config['max_retries']:
                            backoff_time = self.config['backoff_factor'] ** retry_count
                            time.sleep(backoff_time)
                            return self.call_llm(messages, retry_count + 1)
                    
                    raise ValueError(error_msg)
                
                response_data = response.json()
                content = response_data['choices'][0]['message']['content'].strip()
                
                # Track output tokens for rate limiting
                output_tokens = self.rate_limiter.estimate_tokens(content)
                self.rate_limiter.throttle(output_tokens)
                
                return content
                
        except Exception as e:
            if retry_count < self.config['max_retries']:
                backoff_time = self.config['backoff_factor'] ** retry_count
                log_msg(f"LLM call failed: {e}. Retrying in {backoff_time} seconds.", Fore.YELLOW)
                time.sleep(backoff_time)
                return self.call_llm(messages, retry_count + 1)
            else:
                log_msg(f"LLM call failed after {self.config['max_retries']} retries: {e}", Fore.RED)
                raise e
    
    def process_text(self, system_prompt: str, user_text: str) -> str:
        """Process text using the LLM."""
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
    
    def process_batch(self, tasks: List[Dict[str, str]]) -> List[str]:
        """Process a batch of text prompts in parallel for improved throughput."""
        with ThreadPoolExecutor(max_workers=self.config['batch_size']) as executor:
            futures = [
                executor.submit(self.process_text, task['system_prompt'], task['user_text']) 
                for task in tasks
            ]
            return [future.result() for future in futures]


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Validate the config file has a tasks list
        if not config_data or not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a valid YAML document")
        
        if 'tasks' not in config_data or not isinstance(config_data['tasks'], list):
            raise ValueError("Configuration file must contain a 'tasks' list")
        
        return config_data
    except Exception as e:
        log_msg(f"Error loading configuration file: {e}", Fore.RED, '❌')
        sys.exit(1)

def process_notebook_with_tasks(notebook_path: str, tasks: List[Dict[str, Any]], output: Optional[str] = None) -> None:
    """Process a notebook with a sequence of tasks from a configuration file."""
    # Determine output file path
    outfile = output if output else notebook_path
    
    # Initialize configuration
    config = Config()
    
    # Initialize LLM service if needed for any task
    llm_service = None
    if any(task['name'] in LLM_REQUIRED_TASKS for task in tasks):
        try:
            llm_service = OpenRouterLLMService(config)
        except ValueError as e:
            log_msg(f"Cannot run LLM-based tasks: {e}", Fore.RED, '❌')
            log_msg("Set OPENROUTER_API_KEY environment variable", Fore.YELLOW)
            sys.exit(1)
        except Exception as e:
            log_msg(f"Cannot run LLM-based tasks: {e}", Fore.RED, '❌')
            sys.exit(1)
    
    # Track executed tasks for summary
    executed_tasks = []
    
    # Load the notebook initially
    try:
        nb = nbformat.read(notebook_path, as_version=4)
    except Exception as e:
        log_msg(f"Error loading notebook {notebook_path}: {e}", Fore.RED, '❌')
        return
    
    log_msg(f"Processing {notebook_path} with {len(tasks)} tasks...", Fore.CYAN, '🔄')
    
    # Process each task in sequence
    for task_config in tasks:
        task_name = task_config['name']
        
        # Check if task exists
        if task_name not in TASK_MAP:
            log_msg(f"Unknown task '{task_name}', skipping", Fore.YELLOW, '⚠️')
            continue
        
        log_msg(f"Executing task '{task_name}'...", Fore.CYAN, '🔍')
        task = TASK_MAP[task_name](config)
        
        # Process notebook with current task
        if task_name in ["fix_colab_links", "dump_markdown"]:
            # These tasks need special handling for output
            if task_name == "dump_markdown":
                # For dump_markdown, change extension to .md
                md_outfile = os.path.splitext(outfile)[0] + ".md"
                task.process_notebook(notebook_path, md_outfile, llm_service, nb)
            else:
                task.process_notebook(notebook_path, outfile, llm_service, nb)
        elif hasattr(task, 'process_batch') and llm_service is not None:
            # Use batch processing for better throughput
            try:
                batch_size = config['batch_size']
                cells = nb.cells
                for i in range(0, len(cells), batch_size):
                    batch_cells = cells[i:i+batch_size]
                    task.process_batch(batch_cells, llm_service)
            except Exception as e:
                log_msg(f"Error during batch processing: {e}", Fore.RED, '❌')
                # Fall back to individual processing
                log_msg("Falling back to individual cell processing", Fore.YELLOW)
                for cell in nb.cells:
                    task.process_cell(cell, llm_service)
        else:
            # Process cells individually
            for cell in nb.cells:
                task.process_cell(cell, llm_service)
        
        # Save the notebook after each task (except for dump_markdown which outputs to a different file)
        if task_name != "dump_markdown":
            try:
                nbformat.write(nb, outfile)
                log_msg(f"Saved notebook after task '{task_name}'", Fore.GREEN, '💾')
            except Exception as e:
                log_msg(f"Error saving notebook after task '{task_name}': {e}", Fore.RED, '❌')
                continue
        
        # Add to executed tasks list
        executed_tasks.append(task_name)
    
    # Print summary of executed tasks
    if executed_tasks:
        log_msg("\nTasks completed:", Fore.CYAN, '📋')
        for task_name in executed_tasks:
            log_msg(f"  {task_name}", Fore.GREEN, '✅')
    else:
        log_msg("\nNo tasks were executed successfully.", Fore.YELLOW, '⚠️')

def process_notebook(infile: str, task_name: str, output: Optional[str] = None) -> None:
    """Process a single notebook with the given task."""
    # Determine output file path
    if output:
        if os.path.isdir(output):
            # Keep original notebook name but change extension for dump_markdown task
            base_name = os.path.basename(infile)
            if task_name == "dump_markdown":
                base_name = os.path.splitext(base_name)[0] + ".md"
            outfile = os.path.join(output, base_name)
        else:
            outfile = output
    else:
        if task_name == "dump_markdown":
            # For dump_markdown, change extension to .md by default
            base_name = os.path.splitext(os.path.basename(infile))[0] + ".md"
            outfile = os.path.join(os.path.dirname(infile), base_name)
        else:
            # For other tasks, use the original file name
            outfile = infile
    
    # Initialize configuration
    config = Config()
    
    # Initialize LLM service if needed
    llm_service = None
    if task_name in LLM_REQUIRED_TASKS:
        try:
            llm_service = OpenRouterLLMService(config)
        except ValueError as e:
            log_msg(f"Cannot run '{task_name}' task: {e}", Fore.RED, '❌')
            log_msg("Set OPENROUTER_API_KEY environment variable", Fore.YELLOW)
            sys.exit(1)
        except Exception as e:
            log_msg(f"Cannot run '{task_name}' task: {e}", Fore.RED, '❌')
            sys.exit(1)
    
    # Load the task
    if task_name not in TASK_MAP:
        print(f"Error: Unknown task '{task_name}'.")
        return
        
    task = TASK_MAP[task_name](config)
    
    # Load notebook
    try:
        nb = nbformat.read(infile, as_version=4)
        cells = nb.cells
    except Exception as e:
        log_msg(f"Error loading notebook {infile}: {e}", Fore.RED, '❌')
        return
        
    log_msg(f"Processing {infile} with task '{task_name}'...", Fore.CYAN)
    
    # Process cells
    if task_name in ["fix_colab_links", "dump_markdown"]:
        task.process_notebook(infile, outfile, llm_service, nb)
    elif hasattr(task, 'process_batch') and llm_service is not None:
        # Use batch processing for better throughput
        try:
            batch_size = config['batch_size']
            for i in range(0, len(cells), batch_size):
                batch_cells = cells[i:i+batch_size]
                task.process_batch(batch_cells, llm_service)
                log_msg(f"Processed batch {i//batch_size+1}/{(len(cells)+batch_size-1)//batch_size}", Fore.CYAN)
        except Exception as e:
            log_msg(f"Error during batch processing: {e}", Fore.RED, '❌')
            # Fall back to individual processing
            log_msg("Falling back to individual cell processing", Fore.YELLOW)
            for cell in tqdm(cells):
                task.process_cell(cell, llm_service)
    else:
        # Process cells individually
        for cell in tqdm(cells):
            task.process_cell(cell, llm_service)
        
    # Write the notebook (only for tasks that output notebooks)
    if task_name != "dump_markdown":
        try:
            nbformat.write(nb, outfile)
            log_msg(f"Processed notebook saved to {outfile}", Fore.GREEN, '💾')
        except Exception as e:
            log_msg(f"Error saving notebook {outfile}: {e}", Fore.RED, '❌')


def main():
    """Main entry point for the Notebroom CLI."""
    # Create a more intuitive command-line interface
    parser = argparse.ArgumentParser(
        description="🧹 Notebroom - A CLI tool for cleaning and processing Jupyter notebooks with LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix Colab links in a notebook
  notebroom fix_colab_links path/to/notebook.ipynb
  
  # Clean markdown cells in all notebooks in a directory
  notebroom clean_markdown path/to/notebooks/ -o path/to/output/
  
  # Export notebook to markdown for LLM processing
  notebroom dump_markdown notebook.ipynb -o notebook_for_llm.md
  
  # Run a sequence of tasks from a config file
  notebroom task_config.yaml path/to/notebook.ipynb
        """
    )
    
    # Add task argument - can be either a task name or a path to a config file
    parser.add_argument(
        "task",
        metavar="TASK",
        help="Task to execute or path to a YAML configuration file with tasks. Available tasks: " + ", ".join(AVAILABLE_TASKS),
    )
    
    # Notebook path is always required
    parser.add_argument(
        "notebook",
        metavar="NOTEBOOK_PATH",
        help="Path to the input notebook file or directory containing notebooks"
    )
    
    parser.add_argument(
        "-o", "--output",
        metavar="OUTPUT_PATH",
        help="Path to the output file or directory. If not provided, input files will be modified in-place.",
        default=None
    )
    
    # Parse arguments
    args = parser.parse_args()
    infile = args.notebook
    
    # Check if the task is a YAML config file or a task name
    is_config_file = False
    if os.path.isfile(args.task) and (args.task.endswith('.yaml') or args.task.endswith('.yml')):
        is_config_file = True
    elif args.task not in AVAILABLE_TASKS:
        parser.error(f"'{args.task}' is not a recognized task or a valid YAML config file. Available tasks: {', '.join(AVAILABLE_TASKS)}")
    
    # Handle directory or file input
    if os.path.isdir(infile):
        notebooks = find_notebooks(infile)
        if not notebooks:
            print(f"Error: No .ipynb files found in {infile}")
            sys.exit(1)
            
        # Prompt for confirmation
        print(f"Found {len(notebooks)} notebook files in {infile}:")
        for nb in notebooks[:5]:  # Show first 5 notebooks
            print(f" - {os.path.basename(nb)}")
        if len(notebooks) > 5:
            print(f" ... and {len(notebooks) - 5} more")
        
        # Check output option for multiple notebooks
        if args.output and not os.path.isdir(args.output):
            print("Error: When processing multiple notebooks, output (-o) must be a directory.")
            sys.exit(1)
        
        if is_config_file:
            # Process with config file
            config_data = load_config_file(args.task)
            tasks = config_data['tasks']
            
            confirm = input(f"Process all {len(notebooks)} notebooks with {len(tasks)} tasks from config? [y/N] ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
                
            # Process each notebook with all tasks in sequence
            for nb_file in notebooks:
                output_path = os.path.join(args.output, os.path.basename(nb_file)) if args.output else None
                process_notebook_with_tasks(nb_file, tasks, output_path)
        else:
            # Process with single task
            task_name = args.task
            
            # Check for OpenRouter API key if using LLM tasks
            if not os.getenv("OPENROUTER_API_KEY") and task_name in LLM_REQUIRED_TASKS:
                print(f"Error: OPENROUTER_API_KEY environment variable must be set for the {task_name} task.")
                sys.exit(1)
                
            confirm = input(f"Process all {len(notebooks)} notebooks with task '{task_name}'? [y/N] ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
                
            # Process each notebook with the single task
            for nb_file in notebooks:
                process_notebook(nb_file, task_name, args.output)
    else:
        # Check if file exists
        if not os.path.exists(infile):
            print(f"Error: Notebook file not found: {infile}")
            sys.exit(1)
            
        if is_config_file:
            # Process with config file
            config_data = load_config_file(args.task)
            tasks = config_data['tasks']
            process_notebook_with_tasks(infile, tasks, args.output)
        else:
            # Process with single task
            task_name = args.task
            
            # Check for OpenRouter API key if using LLM tasks
            if not os.getenv("OPENROUTER_API_KEY") and task_name in LLM_REQUIRED_TASKS:
                print(f"Error: OPENROUTER_API_KEY environment variable must be set for the {task_name} task.")
                sys.exit(1)
                
            # Process single notebook with the single task
            process_notebook(infile, task_name, args.output)

if __name__ == "__main__":
    main()

