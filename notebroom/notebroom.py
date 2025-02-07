"""Notebroom - Jupyter notebook markdown cleaner using LLMs."""

import os, sys, re, logging, nbformat, tiktoken, time, random, threading
from functools import partial
from tqdm import tqdm
from colorama import Fore, Style, init
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo
from concurrent.futures import ThreadPoolExecutor

# Initialize
load_dotenv()
init(autoreset=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'model': os.getenv('MODEL', 'gpt-4o-mini'),
    'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
    'keep_recent': int(os.getenv('KEEP_RECENT', '3')),
    'temp': float(os.getenv('TEMPERATURE', '0.2')),
    'tpm_limit': int(os.getenv('TPM_LIMIT', '10000000')),
    'rpm_limit': int(os.getenv('RPM_LIMIT', '100')),
    'max_retries': int(os.getenv('MAX_RETRIES', '5')),
    'backoff_factor': float(os.getenv('BACKOFF_FACTOR', '3')),
    'error_throttle_time': int(os.getenv('ERROR_THROTTLE_TIME', '3')),
    'num_workers': int(os.getenv('NUM_WORKERS', '4'))
}

# Prompts
REWRITE_PROMPT = """Your task is to make existing educational content more concise and clear.
Important rules:
- DO NOT add new information or change meaning.
- DO NOT modify section headers.
- FOCUS on making the given text more concise while preserving all information.
- ENSURE clarity and educational value.
- MAINTAIN technical accuracy.
- USE emojis where applicable to increase engagement, but err on the side of not using them.
Return ONLY the rewritten markdown cell. Do not include any introductory or concluding remarks."""

SUMMARY_PROMPT = """Summarize these notebook cells concisely, preserving key concepts and progression.
Important aspects:
- Maintain the educational flow
- Keep critical code examples and their purpose
- Preserve technical accuracy and terminology
- Create a flowing narrative that connects concepts
- Focus on relationships between ideas"""

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

@limits(calls=CONFIG['rpm_limit'], period=1) # Apply rate limiting
def call_llm(llm, messages):
    """Call the LLM with rate limiting and error handling."""
    try:
        result = llm.invoke(messages) # Invoke the LLM with the given messages
        token_count = len(enc.encode(result.content)) # Estimate token count
        tpm_throttle(token_count) # Throttle based on token count
        return result.content.strip()
    except Exception as e:
        log_msg(f"LLM call failed: {e}. Throttling for {CONFIG['error_throttle_time']} seconds.", Fore.YELLOW)
        time.sleep(CONFIG['error_throttle_time'])
        raise e

@on_exception(expo, RateLimitException, max_tries=CONFIG['max_retries'], factor=CONFIG['backoff_factor'], jitter=random.random) # Apply exponential backoff
def rewrite_cell(llm, cell_source):
    """Rewrite a cell using Langchain with retry."""
    try:
        start_time = time.time()
        messages = [{"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": f"Make this text more concise:\n\n{cell_source}"}]
        result = call_llm(llm, messages) # Call the LLM to rewrite the cell source
        end_time = time.time()
        logger.info(f"LLM call took {end_time - start_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        return cell_source

def process_cell(idx, cell, llm):
    """Process a single cell."""
    if is_header_only(cell.source):
        log_msg(f"\nSkipping header cell #{idx}:\n{cell.source}", Fore.YELLOW, '📌')
        return cell

    log_msg(f"\nProcessing cell {idx}:", Fore.CYAN, '📝')
    try:
        new_text = rewrite_cell(llm, cell.source)
        log_msg("Original:", Fore.RED, '📄')
        log_msg(cell.source, Fore.RED)
        log_msg("Rewritten:", Fore.GREEN, '✨')
        log_msg(new_text, Fore.GREEN)
        log_msg("-" * 80)
        cell.source = new_text
    except Exception as e:
        log_msg(f"Error processing cell {idx}: {e}", Fore.RED, '❌')
    return cell

def process_notebook(infile, outfile):
    """Process notebook markdown cells in parallel."""
    nb = nbformat.read(infile, as_version=4)

    # Initialize LLM
    model_name = CONFIG['model']
    model_params = {
        'temperature': CONFIG['temp'],
        'max_tokens': CONFIG['max_tokens']
    }

    if "gemini" in model_name:
        llm = ChatVertexAI(model_name=model_name, convert_system_message_to_human=True, **model_params)
    else:
        llm = ChatOpenAI(model_name=model_name, **model_params)

    markdown_cells = [(idx, cell) for idx, cell in enumerate(nb.cells) if cell.cell_type == 'markdown']

    with ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
        futures = [executor.submit(process_cell, idx, cell, llm) for idx, cell in markdown_cells]
        
        # Collect results in order
        results = [future.result() for future in tqdm(futures, total=len(markdown_cells))]

    # Update notebook cells with processed results
    for i, (idx, _) in enumerate(markdown_cells):
        nb.cells[idx] = results[i]

    nbformat.write(nb, outfile)
    log_msg(f"Saved to {outfile}", Fore.GREEN, '💾')

def main():
    if len(sys.argv) != 2 or not os.getenv("OPENAI_API_KEY"):
        print("Usage: OPENAI_API_KEY=<key> notebroom <notebook.ipynb>")
        sys.exit(1)

    process_notebook(sys.argv[1], f"{os.path.splitext(sys.argv[1])[0]}.clean.ipynb")

if __name__ == "__main__":
    main()
