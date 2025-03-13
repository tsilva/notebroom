import os
import json
import shutil
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables for API access
load_dotenv()

def colored_text(text, color):
    color_codes = {'red': '\033[31m', 'green': '\033[32m'}
    reset = '\033[0m'
    return f"{color_codes[color]}{text}{reset}"

def ipynb_to_markdown(ipynb_path):
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

def improve_notebook_cell(notebook_path, cell_id):
    notebook_md = ipynb_to_markdown(notebook_path)
    system_prompt = f"""You are an expert educator 📚 tasked with enhancing markdown cells in Jupyter notebooks for AI/ML tutorials. Your goal is to transform the specified markdown cell into a clear ✨, concise ✂️, and engaging 😊 piece of content that fits seamlessly into the notebook’s learning journey.

I’ll provide the notebook in markdown format, with cells labeled:  
`<-- CELL[cell_type] cell_number: START -->`  
`<-- CELL[cell_type] cell_number: END -->`

For the specified cell number, analyze the markdown cell 📝 in context—considering prior and upcoming cells—and deliver an improved version that:  
1. Corrects errors or vague explanations 🛠️  
2. Removes unnecessary fluff for maximum information density 📉  
3. Enhances clarity with tight structure, crisp formatting, and precise wording 📋  
4. Follows AI/ML tutorial best practices: clear headings, bite-sized examples, useful links, and consistent style 🎯  
5. Preserves the cell’s core intent while improving transitions for better notebook flow 🌊  

Use emojis sparingly to add light engagement, but prioritize accuracy and clarity over decoration. Keep the cell’s structure intact if it’s critical to the notebook’s function (e.g., don’t convert prose to code or vice versa unless necessary).

**Important:** Your output must ONLY include the polished markdown content, ready to replace the original. Do NOT include any cell delimiters like `<-- CELL[markdown] X: START -->` or `<-- CELL[markdown] X: END -->`, no extra notes, no explanations—just the improved markdown text itself.

---

**Notebook content in markdown format below:**
---

{notebook_md}
""".strip()
        
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    completion = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{cell_id}"}
        ]
    )
    
    return completion.choices[0].message.content.strip()

def improve_all(notebook_path, output_path=None):
    if output_path is None:
        file_path = Path(notebook_path)
        output_path = str(file_path.parent / f"{file_path.stem}_improved{file_path.suffix}")
    
    shutil.copy2(notebook_path, output_path)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    markdown_cells = [(i, cell) for i, cell in enumerate(notebook.get('cells', []))
                    if i > 0 and cell.get('cell_type') == 'markdown' 
                    and not ''.join(cell.get('source', [])).strip().startswith('#')]
        
    print(f"Found {len(markdown_cells)} markdown cells to improve")
    
    for cell_index, _ in tqdm(markdown_cells, desc="Improving markdown cells"):
        try:
            cell_number = cell_index + 1
            # Get the original content and log it in red
            original_content = '\n'.join(notebook['cells'][cell_index]['source'])
            tqdm.write(colored_text(f"--- Original cell {cell_number} ---", 'red'))
            tqdm.write(colored_text(original_content, 'red'))
            
            # Improve the cell content
            improved_content = improve_notebook_cell(output_path, str(cell_number))
            
            # Log the improved content in green
            tqdm.write(colored_text(f"--- Improved cell {cell_number} ---", 'green'))
            tqdm.write(colored_text(improved_content, 'green'))
            
            # Update the notebook with the improved content
            lines = improved_content.split('\n')
            if lines and lines[-1] == '':
                lines = lines[:-1]  # Remove trailing empty line if present
            notebook['cells'][cell_index]['source'] = lines
            
            # Save the updated notebook
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2)
            
            time.sleep(1)  # Brief pause between improvements
        
        except Exception as e:
            print(f"Error improving cell {cell_index + 1}: {str(e)}")
    
    print(f"\nCompleted! Improved notebook saved to {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py notebook.ipynb [output.ipynb]")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    improve_all(notebook_path, output_path)