"""Notebroom - Jupyter notebook tool with task-based processing using LLMs."""

import os
import sys
import logging
import argparse
import time
import threading
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
import re
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import yaml

from dotenv import load_dotenv
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
    def process_cell(self, cell: NotebookNode, llm_service: Optional[OpenRouterLLMService] = None) -> NotebookNode:
        """Process a single notebook cell."""
        pass
    
    def process_notebook(self, infile: str, outfile: str, 
                         llm_service: Optional[OpenRouterLLMService], nb: NotebookNode) -> None:
        """Process an entire notebook."""
        for cell in nb.cells:
            self.process_cell(cell, llm_service)


class TextProcessingTask(Task):
    """Base class for text-processing tasks that use LLMs."""
    
    def __init__(self, config: Config, system_prompt: str):
        super().__init__(config)
        self.system_prompt = system_prompt
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional[OpenRouterLLMService] = None) -> NotebookNode:
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
    
    def process_batch(self, cells: List[NotebookNode], llm_service: OpenRouterLLMService) -> List[NotebookNode]:
        """Process multiple markdown cells in a batch for better throughput."""
        # Filter cells that need processing
        markdown_cells = []
        cell_indices = []
        
        for i, cell in enumerate(cells):
            if cell.cell_type == 'markdown' and not is_header_only(cell.source):
                markdown_cells.append(cell)
                cell_indices.append(i)
        
        if not markdown_cells:
            return cells
        
        # Create batch tasks
        tasks = [
            {"system_prompt": self.system_prompt, "user_text": cell.source}
            for cell in markdown_cells
        ]
        
        # Process in batch
        log_msg(f"\nBatch processing {len(tasks)} markdown cells", Fore.CYAN, '📦')
        results = llm_service.process_batch(tasks)
        
        # Update cells with results
        for i, (cell, result) in enumerate(zip(markdown_cells, results)):
            log_msg(f"\nCell {i+1}/{len(markdown_cells)}", Fore.CYAN, '📝')
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, '✨')
            log_msg(result, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = result
        
        # Return the updated cells
        return cells


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
            # Markdown style links
            r'\[(?:Open|Run|View) (?:in|on) Colab\]\((https?://colab\.research\.google\.com/[^\)]+)\)',
            r'\[\!\[(?:Open|Run|View) (?:in|on) Colab\]\(https?://[^\)]+\)\]\((https?://colab\.research\.google\.com/[^\)]+)\)',
            # HTML style links with image
            r'<a href="(https?://colab\.research\.google\.com/[^"]+)"[^>]*>.*?</a>'
        ]
        
        colab_url = self.create_colab_url(notebook_path)
        if not colab_url:
            return cell_source
            
        modified_source = cell_source
        
        # Handle different link patterns
        for i, pattern in enumerate(patterns):
            matches = re.finditer(pattern, cell_source)
            for match in matches:
                full_match = match.group(0)
                
                # Extract the URL based on pattern type
                if i == 0:  # Simple markdown link
                    old_link = match.group(1)
                    replacement = full_match.replace(old_link, colab_url)
                elif i == 1:  # Markdown link with image
                    old_link = match.group(2)
                    replacement = full_match.replace(old_link, colab_url)
                elif i == 2:  # HTML link
                    old_link = match.group(1)
                    # For HTML links, we need to construct a proper HTML replacement
                    replacement = f'<a href="{colab_url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
                
                modified_source = modified_source.replace(full_match, replacement)
                
        return modified_source
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional[OpenRouterLLMService] = None) -> NotebookNode:
        """Process a single cell."""
        if cell.cell_type != 'markdown':
            return cell
            
        # Improve detection of colab links with broader patterns
        if not any(pattern in cell.source.lower() for pattern in ['colab', 'href', '<a ']):
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
                         llm_service: Optional[OpenRouterLLMService], nb: NotebookNode) -> None:
        """Process an entire notebook."""
        self.notebook_path = os.path.abspath(infile)
        super().process_notebook(infile, outfile, llm_service, nb)


class DumpNotebookTask(Task):
    """Task for dumping notebooks as specially formatted markdown for LLM processing."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.cell_number = 0
        self.markdown_content = []
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional[OpenRouterLLMService] = None) -> NotebookNode:
        """Process a single cell by converting it to markdown format with special markers."""
        self.cell_number += 1
        
        # Get cell type and content
        cell_type = cell.cell_type.upper()
        cell_content = cell.source
        
        # Create cell marker with cell type and number
        start_marker = f"<!-- CELL:{cell_type}:{self.cell_number} -->"
        end_marker = f"<!-- CELL:{cell_type}:{self.cell_number}:END -->"
        
        # Format content based on cell type
        if cell_type == "CODE":
            formatted_content = f"```python\n{cell_content}\n```"
        else:  # MARKDOWN or other types
            formatted_content = cell_content
        
        # Append formatted cell to markdown content
        self.markdown_content.append(f"{start_marker}\n{formatted_content}\n{end_marker}\n")
        
        return cell
    
    def process_notebook(self, infile: str, outfile: str, 
                         llm_service: Optional[OpenRouterLLMService], nb: NotebookNode) -> None:
        """Process entire notebook and output as markdown."""
        # Reset state
        self.cell_number = 0
        self.markdown_content = []
        
        # Get notebook metadata
        notebook_name = os.path.basename(infile)
        metadata = {
            "filename": notebook_name,
            "cell_count": len(nb.cells),
            "kernelspec": nb.metadata.get("kernelspec", {}).get("name", "unknown")
        }
        
        # Create header with notebook info
        header = (
            f"<!-- NOTEBOOK:{notebook_name} -->\n"
            f"# Notebook: {notebook_name}\n\n"
            f"*This is a generated markdown representation of a Jupyter notebook optimized for LLM processing.*\n\n"
            f"**Metadata**: {metadata}\n\n"
        )
        
        # Process all cells
        for cell in nb.cells:
            self.process_cell(cell, llm_service)
            
        # Combine all parts
        full_content = header + "\n".join(self.markdown_content)
        
        # Write to output file
        try:
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write(full_content)
            log_msg(f"Notebook dumped as markdown to {outfile}", Fore.GREEN, '💾')
        except Exception as e:
            log_msg(f"Error writing markdown file {outfile}: {e}", Fore.RED, '❌')


class StandardizeIndentationTask(Task):
    """Task for standardizing Python code indentation to 2 spaces."""
    
    def __init__(self, config: Config):
        super().__init__(config)
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional[OpenRouterLLMService] = None) -> NotebookNode:
        """Process a code cell to standardize indentation."""
        if cell.cell_type != 'code':
            return cell
            
        source_lines = cell.source.split('\n')
        standardized_lines = []
        
        for line in source_lines:
            if not line.strip():  # Preserve empty lines
                standardized_lines.append(line)
                continue
                
            # Count leading whitespace and determine indentation level
            original_line = line
            leading_whitespace = len(line) - len(line.lstrip())
            
            if leading_whitespace == 0:
                # No indentation, keep the line as is
                standardized_lines.append(line)
                continue
                
            # Extract the indentation part and content part
            indentation = line[:leading_whitespace]
            content = line[leading_whitespace:]
            
            # Calculate indentation level
            if '\t' in indentation:
                # Replace tabs with 4 spaces to count indentation level
                expanded_indent = indentation.replace('\t', ' ' * 4)
                indent_level = len(expanded_indent) // 4
            else:
                # For spaces, estimate the level based on common patterns (2, 4, or 8 spaces)
                if leading_whitespace % 4 == 0:
                    indent_level = leading_whitespace // 4
                elif leading_whitespace % 2 == 0:
                    indent_level = leading_whitespace // 2
                else:
                    # Odd number of spaces, use a best guess
                    indent_level = round(leading_whitespace / 4)
            
            # Create standardized line with 2 spaces per indentation level
            standardized_line = ' ' * (2 * indent_level) + content
            standardized_lines.append(standardized_line)
        
        # Update cell source with standardized indentation
        new_source = '\n'.join(standardized_lines)
        
        if new_source != cell.source:
            log_msg("\nStandardized indentation in code cell:", Fore.CYAN, '🔧')
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Standardized:", Fore.GREEN, '✨')
            log_msg(new_source, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = new_source
            
        return cell


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
    llm_required_tasks = ["clean_markdown", "emojify"]
    if any(task['name'] in llm_required_tasks for task in tasks):
        try:
            llm_service = OpenRouterLLMService(config)
        except ValueError as e:
            log_msg(f"Cannot run LLM-based tasks: {e}", Fore.RED, '❌')
            log_msg("Set OPENROUTER_API_KEY environment variable", Fore.YELLOW)
            sys.exit(1)
        except Exception as e:
            log_msg(f"Cannot run LLM-based tasks: {e}", Fore.RED, '❌')
            sys.exit(1)
    
    # Task mapping
    task_map = {
        "clean_markdown": CleanMarkdownTask(config),
        "emojify": EmojifyTask(config),
        "fix_colab_links": FixColabLinks(config),
        "dump_markdown": DumpNotebookTask(config),
        "standardize_indentation": StandardizeIndentationTask(config)
    }
    
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
        if task_name not in task_map:
            log_msg(f"Unknown task '{task_name}', skipping", Fore.YELLOW, '⚠️')
            continue
        
        log_msg(f"Executing task '{task_name}'...", Fore.CYAN, '🔍')
        task = task_map[task_name]
        
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
    if task_name in ("clean_markdown", "emojify"):
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
    task_map = {
        "clean_markdown": CleanMarkdownTask(config),
        "emojify": EmojifyTask(config),
        "fix_colab_links": FixColabLinks(config),
        "dump_markdown": DumpNotebookTask(config),
        "standardize_indentation": StandardizeIndentationTask(config)
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
    available_tasks = ["clean_markdown", "emojify", "fix_colab_links", "dump_markdown", "standardize_indentation"]
    
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
  notebroom --config task_config.yaml path/to/notebook.ipynb
        """
    )
    
    # Create an argument group for the main action (task or config)
    action_group = parser.add_mutually_exclusive_group(required=True)
    
    # Add task argument as optional now (mutually exclusive with config)
    action_group.add_argument(
        "task",
        metavar="TASK",
        choices=available_tasks,
        help="Task to execute. Available tasks: " + ", ".join(available_tasks),
        nargs="?",
    )
    
    # Add config file argument
    action_group.add_argument(
        "--config",
        "-c",
        metavar="CONFIG_FILE",
        help="Path to a YAML configuration file with tasks to execute"
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
        
        if args.config:
            # Process with config file
            config_data = load_config_file(args.config)
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
            if not os.getenv("OPENROUTER_API_KEY") and task_name in ["clean_markdown", "emojify"]:
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
            
        if args.config:
            # Process with config file
            config_data = load_config_file(args.config)
            tasks = config_data['tasks']
            process_notebook_with_tasks(infile, tasks, args.output)
        else:
            # Process with single task
            task_name = args.task
            
            # Check for OpenRouter API key if using LLM tasks
            if not os.getenv("OPENROUTER_API_KEY") and task_name in ["clean_markdown", "emojify"]:
                print(f"Error: OPENROUTER_API_KEY environment variable must be set for the {task_name} task.")
                sys.exit(1)
                
            # Process single notebook with the single task
            process_notebook(infile, task_name, args.output)

if __name__ == "__main__":
    main()
