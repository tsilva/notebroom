from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import sys
import subprocess
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# Load environment variables for API access
def colored_text(text, color):
    """Return text with ANSI color codes for terminal display."""
    color_codes = {'red': '\033[31m', 'green': '\033[32m'}
    reset = '\033[0m'
    return f"{color_codes[color]}{text}{reset}"

def check_env_vars():
    """Check if required environment variables are set."""
    required_vars = ["OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "MODEL_ID"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(colored_text(f"Error: Missing environment variables: {', '.join(missing)}", 'red'))
        sys.exit(1)

def convert_notebook_to_markdown(notebook_path):
    """
    Convert notebook to markdown using the external notebook2md command
    """
    try:
        result = subprocess.run(['notebook2md', str(notebook_path)], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error converting notebook to markdown: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: notebook2md command not found. Please make sure it's installed and in your PATH.")
        sys.exit(1)

def improve_notebook(notebook_path, verbose=True):
    """Improve markdown cells in a Jupyter notebook using an AI model."""
    # Validate input file
    notebook_path = Path(notebook_path)
    if not notebook_path.exists() or notebook_path.suffix != '.ipynb':
        print(colored_text(f"Error: '{notebook_path}' does not exist or is not a .ipynb file", 'red'))
        sys.exit(1)
    
    # Convert notebook to markdown format
    notebook_md = convert_notebook_to_markdown(notebook_path)
    print(notebook_md)
    
    # Define the tool for the model to specify cell updates
    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_markdown_cells",
                "description": "Update the content of multiple markdown cells in the notebook",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "array",
                            "description": "Array of updates, each specifying a cell number and improved content",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cell_number": {
                                        "type": "integer",
                                        "description": "The number of the cell to update (starting from 1)"
                                    },
                                    "improved_content": {
                                        "type": "string",
                                        "description": "The improved markdown content for the cell"
                                    }
                                },
                                "required": ["cell_number", "improved_content"]
                            }
                        }
                    },
                    "required": ["updates"]
                }
            }
        }
    ]
    
    # Read system prompt from file
    system_prompt_path = Path(__file__).parent / "config" / "system_prompt.txt"
    try:
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        print(colored_text(f"Error: System prompt file not found at {system_prompt_path}", 'red'))
        sys.exit(1)

    # Initialize the OpenAI client
    check_env_vars()
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    # Make a single API call with the entire notebook
    try:
        print(f"Improving {notebook_path}...")
        completion = client.chat.completions.create(
            model=os.getenv("MODEL_ID"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": notebook_md}
            ],
            tools=tools,
            temperature=0.0,
            max_tokens=128_000
        )
        print("API call completed.")
    except Exception as e:
        print(colored_text(f"Error during API call: {str(e)}", 'red'))
        sys.exit(1)
    
    # Load the notebook for updating
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Process the model's response
    updated_cell_count = 0
    tool_calls = completion.choices[0].message.tool_calls or []
    for tool_call in tool_calls:
        if tool_call.function.name == "update_markdown_cells":
            # Parse the function arguments
            arguments = json.loads(tool_call.function.arguments)
            updates = arguments.get("updates", [])
            
            for update in updates:
                cell_number = int(update["cell_number"])
                improved_content = update["improved_content"]
                cell_index = cell_number
                
                print(f"Updating cell {cell_number}...")
                print(improved_content)
                
                # Skip if cell doesn't exist, is the first cell, or is not a markdown cell
                if cell_index >= len(notebook['cells']) or cell_index < 0:
                    if verbose:
                        tqdm.write(colored_text(f"Skipping update for cell {cell_number}: Invalid cell index", 'red'))
                    continue
                if cell_number == 1:
                    if verbose:
                        tqdm.write(colored_text(f"Skipping update for cell {cell_number}: First cell cannot be modified", 'red'))
                    continue
                if notebook['cells'][cell_index]['cell_type'] != 'markdown':
                    if verbose:
                        tqdm.write(colored_text(f"Skipping update for cell {cell_number}: Only markdown cells can be modified", 'red'))
                    continue
                
                # Log original and improved content if verbose
                if verbose:
                    original_source = ''.join(notebook['cells'][cell_index]['source']).strip()
                    tqdm.write(colored_text(f"--- Original cell {cell_number} ---", 'red'))
                    tqdm.write(colored_text(original_source, 'red'))
                    tqdm.write(colored_text(f"--- Improved cell {cell_number} ---", 'green'))
                    tqdm.write(colored_text(improved_content, 'green'))
                
                # Update the cell source
                lines = improved_content.split('\n')
                source = [line + '\n' for line in lines]  # Preserve newlines
                notebook['cells'][cell_index]['source'] = source
                print(f"Updated cell {cell_number}")
                updated_cell_count += 1
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"\nUpdated {updated_cell_count} markdown cells. ({updated_cell_count/len(notebook['cells'])*100:.2f}% of total)")
    print(f"\nCompleted! Improved notebook saved to {notebook_path}")
    return notebook_path

def main():
    """Entry point for the command line tool."""
    if len(sys.argv) < 2:
        print("Usage: notebroom notebook.ipynb [repeat_count]")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    repeat_count = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    
    for i in range(repeat_count):
        print(f"\n--- Iteration {i + 1} of {repeat_count} ---")
        notebook_path = improve_notebook(notebook_path, verbose=True)

if __name__ == "__main__":
    main()
