import os
import sys
import json
import shutil
import textwrap
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import argparse

import autopep8
import nbformat
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# === Constants ===
CONFIG_DIR = Path.home() / ".notebroom"
ENV_PATH = CONFIG_DIR / ".env"
REQUIRED_VARS = ["OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "MODEL_ID"]
PASS_MAP = {
    "expand": ("Conceptual Expansion", "expand_prompt.txt"),
    "educate": ("Educational Enhancements", "educate_prompt.txt"),
    "flow": ("Flow & Transitions", "flow_prompt.txt"),
    "contract": ("Conciseness & Redundancy", "contract_prompt.txt"),
    "style": ("Engagement & Style", "style_prompt.txt"),
    "polish": ("Final Polish", "polish_prompt.txt"),
    "format-code": ("Code Formatter", None),
}
SEPARATOR = "<|CELL_SEPARATOR|>"

# === Helpers ===

def log(msg: str, color: Optional[str] = None) -> None:
    colors = {'red': '\033[31m', 'green': '\033[32m', None: ''}
    print(f"{colors.get(color, '')}{msg}\033[0m")

def fatal(msg: str) -> None:
    log(msg, 'red')
    sys.exit(1)

def normalize_indentation(text: str, spaces: int = 4) -> str:
    return textwrap.indent(textwrap.dedent(text), ' ' * spaces).rstrip()

def format_code(code: str) -> str:
    try:
        return autopep8.fix_code(code, options={
            'aggressive': 0,
            'select': ['E101', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117'],
            'ignore': ['E2', 'E3', 'E4', 'E5', 'W'],
        })
    except Exception as e:
        log(f"⚠️ Code format failed: {e}", 'red')
        return code

def load_prompt(filename: str) -> str:
    path = Path(__file__).parent / "configs" / filename
    if not path.exists():
        fatal(f"Prompt file missing: {path}")
    return path.read_text(encoding='utf-8').strip()

# === Environment ===

def setup_env() -> Dict[str, str]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    example_env = Path(__file__).parent / "configs" / ".env.example"
    if not ENV_PATH.exists():
        if not example_env.exists():
            fatal(f"Example .env missing: {example_env}")
        shutil.copy(example_env, ENV_PATH)
        log(f"✅ Created .env at {ENV_PATH}. Edit this file before rerunning.", 'green')
        sys.exit(0)

    load_dotenv(dotenv_path=ENV_PATH, override=True)
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        fatal(f"Missing env vars: {', '.join(missing)}")
    return {var: os.getenv(var) for var in REQUIRED_VARS}

# === Notebook Parsing ===

def extract_cells(notebook_path: Path) -> Tuple[Dict, List[Dict], str]:
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    cells_text, cleaned_cells = ["<|NOTEBOOK_START|>"], []
    total_cells = len(notebook['cells'])

    for idx, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        content = ''.join(cell['source']).strip() or f"(Empty {cell_type} cell)"

        header = f"<|CELL_HEADER|> Cell {idx} of {total_cells - 1} [{cell_type.upper()}]"
        if exec_count := cell.get('execution_count'):
            header += f" | Execution Count: {exec_count}"
        if tags := cell.get('metadata', {}).get('tags'):
            header += f" | Tags: {', '.join(tags)}"

        cells_text.append(header)
        cells_text.append("```python" if cell_type == 'code' else "")
        cells_text.append(normalize_indentation(content))
        if cell_type == 'code':
            cells_text.append("```")

        if outputs := cell.get('outputs'):
            outputs_text = []
            for output in outputs:
                output_data = output.get('text') or output.get('data', {}).get('text/plain') or output.get('data', {}).get('text/html')
                if output_data:
                    outputs_text.append(normalize_indentation(''.join(output_data).strip()))
            if outputs_text:
                cells_text.extend(["*Output:*", "```", "\n".join(outputs_text), "```"])

        cells_text.append(SEPARATOR)
        cleaned_cells.append({"cell_number": idx, "cell_type": cell_type, "content": content})

    cells_text.append("<|NOTEBOOK_END|>")
    return notebook, cleaned_cells, '\n'.join(cells_text)

# === AI Passes ===

def apply_pass(client, notebook, cleaned_cells, env_vars, task_id, notebook_text) -> Tuple[Dict, List[Dict]]:
    pass_name, prompt_file = PASS_MAP[task_id]
    log(f"\n🎯 Starting pass: {pass_name}", 'green')

    if task_id == "format-code":
        formatted = 0
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                original = ''.join(cell['source'])
                formatted_code = format_code(original)
                if formatted_code != original:
                    cell['source'] = [line + '\n' for line in formatted_code.splitlines()]
                    formatted += 1
        log(f"✅ Formatted {formatted} code cells" if formatted else "ℹ️ Code already formatted", 'green')
        return notebook, cleaned_cells

    system_prompt = load_prompt("base_system_prompt.txt")
    if prompt_file:
        system_prompt += f"\n\n{load_prompt(prompt_file)}"

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
        response = client.chat.completions.create(
            model=env_vars["MODEL_ID"],
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": notebook_text}],
            tools=tools,
            temperature=0.0,
            max_tokens=128_000
        )
    except Exception as e:
        fatal(f"API error during {pass_name}: {str(e).splitlines()[0]}")

    tool_calls = getattr(getattr(response.choices[0].message, 'tool_calls', None), '__iter__', lambda: [])()
    cell_map = {c['cell_number']: idx for idx, c in enumerate(cleaned_cells)}
    updated = 0

    for call in tool_calls:
        if call.function.name != "update_markdown_cells":
            continue

        updates = json.loads(call.function.arguments).get("updates", [])
        for upd in updates:
            idx = upd["cell_number"]
            if idx >= len(notebook['cells']) or notebook['cells'][idx]['cell_type'] != 'markdown':
                continue

            improved = upd['improved_content'].strip()
            notebook['cells'][idx]['source'] = [line + '\n' for line in improved.splitlines()]
            if (clean_idx := cell_map.get(idx)) is not None:
                cleaned_cells[clean_idx]['content'] = improved
            updated += 1

    log(f"✅ {pass_name} pass: Updated {updated} markdown cells", 'green')
    return notebook, cleaned_cells

# === Main ===

def improve_notebook(path: str, env_vars: Dict[str, str], tasks: List[str]) -> str:
    notebook_path = Path(path)
    if not notebook_path.exists() or notebook_path.suffix != ".ipynb":
        fatal(f"Invalid notebook file: {notebook_path}")

    notebook, cleaned_cells, notebook_text = extract_cells(notebook_path)
    client = OpenAI(base_url=env_vars["OPENROUTER_BASE_URL"], api_key=env_vars["OPENROUTER_API_KEY"])

    for task in tasks:
        if task not in PASS_MAP:
            log(f"⚠️ Unknown task: {task}", 'red')
            continue
        notebook, cleaned_cells = apply_pass(client, notebook, cleaned_cells, env_vars, task, notebook_text)

    output_path = notebook_path.with_name(f"{notebook_path.stem}.ipynb")
    nbformat.write(nbformat.from_dict(notebook), str(output_path))
    log(f"\n✅ Improvement complete! Output: {output_path}", 'green')
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Notebroom - Notebook Improver")
    parser.add_argument("notebook", help="Path to the notebook (.ipynb)")
    parser.add_argument("--tasks", nargs="+", default=list(PASS_MAP.keys()), help=f"Tasks to run: {', '.join(PASS_MAP.keys())}")
    args = parser.parse_args()

    env_vars = setup_env()
    improve_notebook(args.notebook, env_vars, args.tasks)