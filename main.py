import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables for API access
load_dotenv()

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

def ipynb_to_markdown(ipynb_path):
    """Convert a Jupyter notebook to a markdown string for processing."""
    with open(ipynb_path, 'r', encoding='utf-8') as f: 
        notebook = json.load(f)

    markdown_content = []
    for i, cell in enumerate(notebook.get('cells', [])):
        cell_type = cell.get('cell_type', 'unknown')
        type_abbrev = {'markdown': 'md', 'code': 'code'}.get(cell_type, 'unk')
        cell_num = str(i + 1)
        start_delim = f"<-- START:{cell_num}:{type_abbrev} -->\n"
        end_delim = f"<-- END:{cell_num}:{type_abbrev} -->\n"
        
        # Preserve the exact source, including trailing newlines
        cell_source = ''.join(cell.get('source', []))
        
        if cell_type == 'markdown':
            content = cell_source
        elif cell_type == 'code':
            outputs = [f"> Output:\n{''.join(output.get('text', [])).rstrip()}\n" 
                       for output in cell.get('outputs', []) 
                       if output.get('output_type') == 'stream' and ''.join(output.get('text', [])).rstrip()]
            content = cell_source + ''.join(outputs)
        else:
            content = ""

        markdown_content.append(start_delim + content + end_delim)

    return ''.join(markdown_content)

def improve_notebook(notebook_path, verbose=True):
    """Improve markdown cells in a Jupyter notebook using an AI model."""
    # Validate input file
    notebook_path = Path(notebook_path)
    if not notebook_path.exists() or notebook_path.suffix != '.ipynb':
        print(colored_text(f"Error: '{notebook_path}' does not exist or is not a .ipynb file", 'red'))
        sys.exit(1)
    
    # Convert notebook to markdown format
    notebook_md = ipynb_to_markdown(notebook_path)
    
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
    
    # System prompt with instructions for the model
    system_prompt = """You are an expert educator 📚 enhancing markdown cells in Jupyter notebooks for AI/ML tutorials. You'll receive notebook markdown cells labeled:

`<-- START:cell_number:cell_type -->`
`<-- END:cell_number:cell_type -->`

Analyze markdown cells that:
- Are NOT the first cell (cell_number > 1)
- Do NOT start with '#'

Improve cells ONLY if the current content:
- Is unclear, confusing, verbose, or redundant.
- Contains errors or irrelevant details.
- Lacks engagement or disrupts flow.
- Doesn't follow AI/ML tutorial best practices (concise examples, clear style).

Your improved version MUST:
- Be clear ✨, concise ✂️, and engaging 😊.
- Correct inaccuracies or vagueness 🛠️.
- Remove redundancy 📉.
- Follow AI/ML tutorial best practices (concise examples, consistent style) 🎯.
- Preserve the original intent, improving transitions for flow.
- NOT add headings (e.g., `# Heading`).

**Important Instructions for Updates**:
- Only call `update_markdown_cells` for cells where you have made a meaningful improvement to the content (e.g., rephrased for clarity, corrected errors, removed redundancy, improved engagement).
- **Do NOT call `update_markdown_cells` if the improved content is effectively identical to the original**, even if there are minor formatting differences (e.g., extra spaces, trailing newlines, or line ending variations). "Effectively identical" means the rendered markdown output would look the same to a reader.
- If a cell does not need improvement (because it’s already clear, concise, and follows best practices), exclude it entirely from the updates array. Do not include it just to report “no change.”
- When providing improved content, preserve the original formatting structure as much as possible (e.g., do not add or remove trailing newlines unless it’s part of a meaningful improvement).

Never modify or suggest changes to code cells.
""".strip()

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
            tool_choice="required",
            temperature=0.0,
            max_tokens=128_000
        )
        print(completion)
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
                cell_index = cell_number - 1  # Convert to 0-based index
                
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

import sys
from notebroom.main import main

if __name__ == "__main__":
    main()