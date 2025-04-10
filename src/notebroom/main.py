import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import textwrap
import autopep8
import argparse
import nbformat
from typing import List, Tuple, Dict, Any, Optional

# === Constants ===
RED, GREEN, RESET = '\033[31m', '\033[32m', '\033[0m'
CONFIG_DIR = Path.home() / ".notebroom"
ENV_PATH = CONFIG_DIR / ".env"
REQUIRED_VARS = ["OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "MODEL_ID"]

# Map of pass IDs to their display names and corresponding system prompt files
PASS_MAP = {
    "expand": ("Conceptual Expansion", "expand_prompt.txt"),
    "educate": ("Educational Enhancements", "educate_prompt.txt"),
    "flow": ("Flow & Transitions", "flow_prompt.txt"),
    "contract": ("Conciseness & Redundancy", "contract_prompt.txt"),
    "style": ("Engagement & Style", "style_prompt.txt"),
    "polish": ("Final Polish", "polish_prompt.txt"),
    "format-code": ("Code Formatter", None),  # No prompt file needed for code formatting
}

# === Helper Functions ===
def normalize_indentation(text: str, spaces: int = 4) -> str:
    return textwrap.indent(textwrap.dedent(text), ' ' * spaces).rstrip()

def format_code_cell(code: str) -> str:
    try:
        # Use autopep8 with minimal settings - focus on indentation only
        return autopep8.fix_code(
            code,
            options={
                'aggressive': 0,  # Minimal changes
                'select': ['E101', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117'],  # Only indentation
                'ignore': ['E2', 'E3', 'E4', 'E5', 'W'],  # Ignore everything else
            }
        )
    except Exception as e:
        log(f"⚠️ Failed to format code: {e}", 'red')
        return code

# === Logging ===
def log(msg: str, color: Optional[str] = None) -> None:
    colors = {'red': RED, 'green': GREEN}
    print(f"{colors.get(color, '')}{msg}{RESET if color else ''}")

def fatal(msg: str) -> None:
    log(msg, 'red')
    sys.exit(1)

# === Setup ===
def setup_env() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    example_env = Path(__file__).parent / "configs" / ".env.example"
    if not ENV_PATH.exists():
        if not example_env.exists():
            fatal(f"❌ Example .env file not found at {example_env}")
        try:
            shutil.copy(example_env, ENV_PATH)
            log(f"✅ Created default env file at {ENV_PATH}", 'green')
            print(f"⚠️  Edit this file and rerun.\n🛠️  Use: nano {ENV_PATH}")
        except Exception as e:
            fatal(f"❌ Could not create .env: {str(e)}")
        sys.exit(1)
    load_dotenv(dotenv_path=ENV_PATH, override=True)

def validate_env() -> Dict[str, str]:
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        fatal(f"Missing env vars: {', '.join(missing)}")
    return {var: os.getenv(var) for var in REQUIRED_VARS}

# === Extract Notebook Cells ===
def extract_notebook_cells(notebook_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    cells_text = []

    total_cells = len(notebook['cells'])
    separator = "<|CELL_SEPARATOR|>"

    cells_text.append("<|NOTEBOOK_START|>")

    for idx, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        content = ''.join(cell['source']).strip()
        execution_count = cell.get('execution_count', None)
        metadata = cell.get('metadata', {})

        cell_header = f"<|CELL_HEADER|> Cell {idx + 1} of {total_cells} [{cell_type.upper()}]"
        if execution_count is not None:
            cell_header += f" | Execution Count: {execution_count}"
        if metadata.get('tags'):
            cell_header += f" | Tags: {', '.join(metadata['tags'])}"

        cells_text.append(cell_header)

        if cell_type == 'code':
            cells_text.append("```python")
            normalized_content = normalize_indentation(content or "# (Empty code cell)")
            cells_text.append(normalized_content)
            cells_text.append("```")
        else:
            cells_text.append(content or "(Empty markdown cell)")

        if cell_type == 'code' and 'outputs' in cell:
            outputs = cell['outputs']
            output_texts = []
            for output in outputs:
                if output.get('text'):
                    output_texts.append(''.join(output['text']).strip())
                elif output.get('data', {}).get('text/plain'):
                    output_texts.append(''.join(output['data']['text/plain']).strip())
                elif output.get('data', {}).get('text/html'):
                    output_texts.append(''.join(output['data']['text/html']).strip())
            if output_texts:
                cells_text.append("*Output:*")
                cells_text.append("```")
                normalized_output = "\n".join(normalize_indentation(output) for output in output_texts)
                cells_text.append(normalized_output)
                cells_text.append("```")

        cells_text.append(separator)

    cells_text.append("<|NOTEBOOK_END|>")

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
def run_improvement_pass(
    notebook: Dict[str, Any],
    cleaned_cells: List[Dict[str, Any]],
    env_vars: Dict[str, str],
    pass_name: str,
    prompt_file: str,
    notebook_text: str,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Load base system prompt
    base_prompt_path = Path(__file__).parent / "configs" / "base_system_prompt.txt"
    try:
        system_prompt = base_prompt_path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        fatal(f"Base system prompt not found at {base_prompt_path}")

    # Load pass-specific prompt if provided
    pass_prompt = ""
    if prompt_file:
        pass_prompt_path = Path(__file__).parent / "configs" / prompt_file
        try:
            pass_prompt = pass_prompt_path.read_text(encoding='utf-8').strip()
        except FileNotFoundError:
            fatal(f"Pass-specific prompt not found at {pass_prompt_path}")

    # Combine prompts if we have a pass-specific prompt
    if pass_prompt:
        system_prompt = f"{system_prompt}\n\n{pass_prompt}"

    client = OpenAI(base_url=env_vars["OPENROUTER_BASE_URL"], api_key=env_vars["OPENROUTER_API_KEY"])

    tools = [{
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
    }]

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
        fatal(f"API error during {pass_name} pass: {str(e).splitlines()[0]}")  # Sanitize output

    tool_calls = getattr(getattr(response.choices[0].message, 'tool_calls', None), '__iter__', lambda: [])()

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

            original_content = ''.join(cell['source']).strip()
            improved_content = upd['improved_content'].strip()

            if original_content.startswith('#') and all(line.strip().startswith('#') for line in original_content.splitlines() if line.strip()):
                log(f"⏭️  Skipping section header-only cell {cell_num}.", 'red')
                continue

            if verbose:
                tqdm.write(f"\nUpdating cell {cell_num}...")
                tqdm.write(f"Before:\n{original_content}")
                tqdm.write(f"After:\n{improved_content}")

            cell['source'] = [line + '\n' for line in improved_content.split('\n')]
            cleaned_idx = cell_index_map.get(cell_num)
            if cleaned_idx is not None:
                cleaned_cells[cleaned_idx]['content'] = improved_content
            updated += 1

    if updated:
        log(f"✅ {pass_name} pass: Updated {updated} markdown cells ({updated / len(notebook['cells']) * 100:.2f}%)", 'green')
    else:
        log(f"⚠️  {pass_name} pass: No cells updated.", 'red')

    return notebook, cleaned_cells

# === Main Improvement ===
def improve_notebook(path: str, env_vars: Dict[str, str], tasks: List[str], verbose: bool = True) -> str:
    notebook_path = Path(path)
    if not notebook_path.exists() or notebook_path.suffix != ".ipynb":
        fatal(f"Invalid notebook file: {notebook_path}")

    notebook, cleaned_cells, notebook_text = extract_notebook_cells(notebook_path)

    for idx, task_id in enumerate(tasks, start=1):
        if task_id == "format-code":
            formatted = 0
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    original_code = ''.join(cell['source'])
                    formatted_code = format_code_cell(original_code)
                    if formatted_code != original_code:
                        cell['source'] = [line + '\n' for line in formatted_code.splitlines()]
                        formatted += 1
            if formatted:
                log(f"✅ Formatted {formatted} code cells with Black", 'green')
            else:
                log(f"ℹ️  Code cells already properly formatted", 'green')
            continue

        pass_name, prompt_file = PASS_MAP.get(task_id, (None, None))
        if not pass_name:
            log(f"⚠️  Unknown task ID '{task_id}'. Skipping.", 'red')
            continue

        log(f"\n🎯 Task {idx}/{len(tasks)}: {pass_name}", 'green')
        notebook, cleaned_cells = run_improvement_pass(
            notebook, cleaned_cells, env_vars, pass_name, prompt_file, notebook_text, verbose
        )

    output_path = notebook_path.with_name(f"{notebook_path.stem}.ipynb")
    nbformat.write(nbformat.from_dict(notebook), str(output_path))
    log(f"\n✅ Improvement complete! Output: {output_path}", 'green')
    return str(output_path)

# === CLI Entrypoint ===
def main() -> None:
    parser = argparse.ArgumentParser(description="Notebroom - Notebook Improver")
    parser.add_argument("notebook", help="Path to the notebook file (.ipynb)")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(PASS_MAP.keys()),
        help=f"Sequence of tasks to run (default: normal mode). Available: {', '.join(PASS_MAP.keys())}"
    )
    args = parser.parse_args()

    setup_env()
    env_vars = validate_env()

    improve_notebook(args.notebook, env_vars, tasks=args.tasks, verbose=True)

if __name__ == "__main__":
    main()
