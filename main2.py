import os
import json
import shutil
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

def ipynb_to_markdown(ipynb_path):
    """Convert a Jupyter notebook to markdown format with cell delimiters."""
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    markdown_content = []
    for i, cell in enumerate(notebook.get('cells', [])):
        cell_type = cell.get('cell_type', 'unknown')
        cell_source = ''.join(cell.get('source', []))
        cell_num = f"{i+1}"
        markdown_content.append(f"<-- CELL[{cell_type}] {cell_num}:  START  -->\n")
        
        if cell_type == 'markdown':
            markdown_content.append(cell_source + "\n")
        elif cell_type == 'code':
            markdown_content.append(f"```python\n{cell_source}\n```\n")
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'stream':
                    markdown_content.append(f"```\n{''.join(output.get('text', []))}\n```\n")
        
        markdown_content.append(f"<-- CELL[{cell_type}] {cell_num}:  END  -->\n\n")
    
    return ''.join(markdown_content)

def improve_all_at_once(notebook_path, output_path=None):
    """Improve all eligible markdown cells in a notebook with a single API call."""
    # Set default output path if not provided
    if output_path is None:
        file_path = Path(notebook_path)
        output_path = str(file_path.parent / f"{file_path.stem}_improved{file_path.suffix}")
    
    # Copy the original notebook to the output path
    shutil.copy2(notebook_path, output_path)
    
    # Convert notebook to markdown format
    notebook_md = ipynb_to_markdown(output_path)
    
    # Define the tool for the model to specify cell updates
    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_markdown_cell",
                "description": "Update the content of a markdown cell in the notebook",
                "parameters": {
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
        }
    ]
    
    # System prompt with instructions for the model
    system_prompt = """
You are an expert educator 📚 tasked with enhancing markdown cells in Jupyter notebooks for AI/ML tutorials. You will be provided with the notebook in markdown format, with cells labeled:  
`<-- CELL[cell_type] cell_number: START -->`  
`<-- CELL[cell_type] cell_number: END -->`

Your task is to analyze each markdown cell that is not the first cell (cell_number > 1) and whose content does not start with '#', and provide an improved version of its content. For each such cell, call the function `update_markdown_cell` with the cell number (as an integer) and the improved markdown content.

When improving the content, consider the context of the surrounding cells to ensure coherence and flow 🌊. The improved content should:
1. Be clear ✨, concise ✂️, and engaging 😊
2. Correct errors or vague explanations 🛠️
3. Remove unnecessary fluff for maximum information density 📉
4. Follow AI/ML tutorial best practices: use clear headings, bite-sized examples, useful links, and consistent style 🎯
5. Preserve the cell’s core intent while enhancing transitions for better notebook flow

Use emojis sparingly to add light engagement, but prioritize accuracy and clarity over decoration. Do not include the cell delimiters in the improved content; provide only the markdown text that should replace the original cell’s source.
"""
    
    # Initialize the OpenAI client
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    print("Improving notebook...")
    # Make a single API call with the entire notebook
    completion = client.chat.completions.create(
        model="openai/gpt-4.5-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": notebook_md}
        ],
        tools=tools
    )
    print("API call completed.")
    
    # Load the notebook for updating
    with open(output_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(completion)

    # Process the model's response
    if completion.choices[0].message.tool_calls:
        for tool_call in completion.choices[0].message.tool_calls:
            if tool_call.function.name == "update_markdown_cell":
                try:
                    # Parse the function arguments
                    arguments = json.loads(tool_call.function.arguments)
                    cell_number = int(arguments["cell_number"])
                    improved_content = arguments["improved_content"]
                    cell_index = cell_number - 1  # Convert to 0-based index
                    
                    # Validate the cell
                    if (0 <= cell_index < len(notebook['cells']) and 
                        notebook['cells'][cell_index]['cell_type'] == 'markdown'):
                        original_source = ''.join(notebook['cells'][cell_index]['source']).strip()
                        if not original_source.startswith('#'):
                            # Log original and improved content
                            tqdm.write(colored_text(f"--- Original cell {cell_number} ---", 'red'))
                            tqdm.write(colored_text(original_source, 'red'))
                            tqdm.write(colored_text(f"--- Improved cell {cell_number} ---", 'green'))
                            tqdm.write(colored_text(improved_content, 'green'))
                            
                            # Update the cell source
                            lines = improved_content.split('\n')
                            source = [line + '\n' for line in lines]  # Preserve newlines
                            notebook['cells'][cell_index]['source'] = source
                            print(f"Updated cell {cell_number}")
                        else:
                            print(f"Skipped cell {cell_number} as it starts with '#'")
                    else:
                        print(f"Invalid cell number {cell_number}")
                except Exception as e:
                    print(f"Error processing tool call: {e}")
    else:
        print("No updates provided by the model.")
    
    # Save the updated notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"\nCompleted! Improved notebook saved to {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py notebook.ipynb [output.ipynb]")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    improve_all_at_once(notebook_path, output_path)