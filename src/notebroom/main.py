import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# === Constants ===
RED, GREEN, RESET = '\033[31m', '\033[32m', '\033[0m'
CONFIG_DIR = Path.home() / ".notebroom"
ENV_PATH = CONFIG_DIR / ".env"
REQUIRED_VARS = ["OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "MODEL_ID"]

# === Logging ===
def log(msg, color=None):
    colors = {'red': RED, 'green': GREEN}
    print(f"{colors.get(color, '')}{msg}{RESET if color else ''}")

def fatal(msg):
    log(msg, 'red')
    sys.exit(1)

# === Setup ===
def setup_env():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not ENV_PATH.exists():
        try:
            shutil.copy(Path(__file__).parent / "configs" / ".env.example", ENV_PATH)
            log(f"✅ Created default env file at {ENV_PATH}", 'green')
            print(f"⚠️  Edit this file and rerun.\n🛠️  Use: nano {ENV_PATH}")
        except Exception as e:
            fatal(f"❌ Could not create .env: {e}")
        sys.exit(1)
    load_dotenv(dotenv_path=ENV_PATH, override=True)

def validate_env():
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        fatal(f"Missing env vars: {', '.join(missing)}")
    return {var: os.getenv(var) for var in REQUIRED_VARS}

# === Extract Notebook Cells (Improved Version) ===
def extract_notebook_cells(notebook_path):
    notebook = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    cells_text = []

    total_cells = len(notebook['cells'])
    separator = "<|CELL_SEPARATOR|>"

    cells_text.append("<|NOTEBOOK_START|>")

    for idx, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        content = ''.join(cell['source']).strip()
        execution_count = cell.get('execution_count', None)
        metadata = cell.get('metadata', {})

        # ✅ Use safe non-markdown cell header
        cell_header = f"<|CELL_HEADER|> Cell {idx + 1} of {total_cells} [{cell_type.upper()}]"
        if execution_count is not None:
            cell_header += f" | Execution Count: {execution_count}"
        if metadata.get('tags'):
            cell_header += f" | Tags: {', '.join(metadata['tags'])}"

        cells_text.append(cell_header)

        # Add content with appropriate formatting
        if cell_type == 'code':
            cells_text.append("```python")
            cells_text.append(content or "# (Empty code cell)")
            cells_text.append("```")
        else:
            cells_text.append(content or "(Empty markdown cell)")

        # Add outputs if applicable
        if cell_type == 'code' and 'outputs' in cell:
            outputs = cell['outputs']
            output_texts = []
            for output in outputs:
                if output.get('text'):
                    output_texts.append(''.join(output['text']).strip())
                elif output.get('data', {}).get('text/plain'):
                    output_texts.append(''.join(output['data']['text/plain']).strip())
            if output_texts:
                cells_text.append("*Output:*")
                cells_text.append("```")
                cells_text.extend(output_texts)
                cells_text.append("```")

        # Add explicit separator between cells
        cells_text.append(separator)

    cells_text.append("<|NOTEBOOK_END|>")

    # Prepare cleaned_cells for later tracking
    cleaned_cells = [
        {
            "cell_number": idx,
            "cell_type": cell['cell_type'],
            "content": ''.join(cell['source']).strip()
        }
        for idx, cell in enumerate(notebook['cells'])
    ]

    return notebook, cleaned_cells, '\n'.join(cells_text)

# === Notebook Improvement Pass ===
def run_improvement_pass(notebook, cleaned_cells, env_vars, pass_name, prompt_addition, notebook_text, verbose=True):
    system_prompt_path = Path(__file__).parent / "configs" / "system_prompt.txt"
    try:
        system_prompt = system_prompt_path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        fatal(f"System prompt not found at {system_prompt_path}")

    system_prompt = system_prompt.replace("{{CURRENT_PASS_INSTRUCTIONS}}", prompt_addition)

    client = OpenAI(base_url=env_vars["OPENROUTER_BASE_URL"], api_key=env_vars["OPENROUTER_API_KEY"])

    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_markdown_cells",
                "description": "Update markdown cells in the notebook",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cell_number": {"type": "integer"},
                                    "improved_content": {"type": "string"}
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

    try:
        log(f"🔍 Running {pass_name} pass...", 'green')
        response = client.chat.completions.create(
            model=env_vars["MODEL_ID"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": notebook_text}
            ],
            tools=tools,
            temperature=0.0,
            max_tokens=128_000
        )
    except Exception as e:
        fatal(f"API error during {pass_name} pass: {e}")

    tool_calls = response.choices[0].message.tool_calls or []

    # Build cell number to cleaned_cells index map
    cell_index_map = {cell['cell_number']: idx for idx, cell in enumerate(cleaned_cells)}

    updated = 0
    for call in tool_calls:
        if call.function.name != "update_markdown_cells":
            continue
        updates = json.loads(call.function.arguments).get("updates", [])
        for upd in updates:
            cell_num = upd["cell_number"]
            if cell_num < 0 or cell_num >= len(notebook['cells']):
                log(f"⚠️  Skipped update for invalid cell number {cell_num}.", 'red')
                continue
            cell = notebook['cells'][cell_num]
            if cell['cell_type'] != 'markdown':
                log(f"⚠️  Skipped update for non-markdown cell {cell_num}.", 'red')
                continue
            if verbose:
                tqdm.write(f"\nUpdating cell {cell_num}...")
                tqdm.write(f"Before:\n{''.join(cell['source']).strip()}")
                tqdm.write(f"After:\n{upd['improved_content']}")
            cell['source'] = [line + '\n' for line in upd['improved_content'].split('\n')]
            cleaned_idx = cell_index_map.get(cell_num)
            if cleaned_idx is not None:
                cleaned_cells[cleaned_idx]['content'] = upd['improved_content']  # Update for next pass
            updated += 1

    if updated:
        log(f"✅ {pass_name} pass: Updated {updated} markdown cells ({updated / len(notebook['cells']) * 100:.2f}%)", 'green')
    else:
        log(f"⚠️  {pass_name} pass: No cells updated.", 'red')

    return notebook, cleaned_cells

# === Multi-Pass Refinement ===
def improve_notebook_multi_pass(path, env_vars, verbose=True):
    notebook_path = Path(path)
    if not notebook_path.exists() or notebook_path.suffix != ".ipynb":
        fatal(f"Invalid notebook file: {notebook_path}")

    notebook, cleaned_cells, notebook_text = extract_notebook_cells(notebook_path)

    passes = [
        # === EXPANSIVE PASSES ===
        ("Conceptual Expansion", 
        "Ensure each step in the notebook is fully explained with minimal but sufficient context. Add brief explanations of *why* each step matters to understanding AI/ML concepts. Do not worry about length yet — focus on educational completeness."),
        
        ("Educational Enhancements", 
        "Strengthen explanations for clarity. Add minimal but helpful clarifications, practical tips, or reminders for learners. Use precise technical language. Make sure the purpose of each step is clear."),
        
        # === BALANCING PASSES ===
        ("Flow & Transitions", 
        "Improve logical flow between steps. Add short, effective transition phrases to guide the learner smoothly from one concept to the next. Keep transitions tight and efficient."),
        
        # === CONTRACTIVE PASSES ===
        ("Conciseness & Redundancy", 
        "Eliminate redundant phrases and unnecessary explanations. Sharpen sentences to be direct and compact while retaining all valuable teaching content added earlier."),
        
        ("Engagement & Style", 
        "Refine tone to be professional, clear, and approachable. Use markdown formatting and minimal, well-placed emojis to improve readability. Do not use humor. Make sure explanations feel motivating but efficient."),
        
        ("Final Polish", 
        "Ensure final tone and style consistency. Verify that the notebook is concise, highly educational, technically accurate, and visually engaging with well-applied markdown and emojis. Remove any leftover verbosity.")
    ]


    for idx, (pass_name, prompt_addition) in enumerate(passes, start=1):
        log(f"\n🎯 Pass {idx}/{len(passes)}: {pass_name}", 'green')
        notebook, cleaned_cells = run_improvement_pass(
            notebook, cleaned_cells, env_vars, pass_name, prompt_addition, notebook_text, verbose
        )

    output_path = notebook_path.with_name(f"{notebook_path.stem}-improved.ipynb")
    output_path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    log(f"\n✅ Multi-pass improvement complete! Output: {output_path}", 'green')
    return str(output_path)

# === CLI Entrypoint ===
def main():
    if len(sys.argv) < 2:
        print("Usage: notebroom notebook.ipynb")
        sys.exit(1)

    setup_env()
    env_vars = validate_env()
    notebook_path = sys.argv[1]

    improve_notebook_multi_pass(notebook_path, env_vars, verbose=True)

if __name__ == "__main__":
    main()
