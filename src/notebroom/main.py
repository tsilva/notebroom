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

# === Extract Markdown Cells ===
def extract_markdown_cells(notebook_path):
    notebook = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    markdown_cells = []
    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            cell_content = ''.join(cell['source']).strip()
            markdown_cells.append({
                "cell_number": idx,
                "content": cell_content
            })
    return notebook, markdown_cells

# === Notebook Improvement ===
def improve_notebook(path, env_vars, verbose=True):
    notebook_path = Path(path)
    if not notebook_path.exists() or notebook_path.suffix != ".ipynb":
        fatal(f"Invalid notebook file: {notebook_path}")

    notebook, markdown_cells = extract_markdown_cells(notebook_path)

    system_prompt_path = Path(__file__).parent / "configs" / "system_prompt.txt"
    try:
        system_prompt = system_prompt_path.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        fatal(f"System prompt not found at {system_prompt_path}")

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
        log(f"Improving {notebook_path}...", 'green')
        response = client.chat.completions.create(
            model=env_vars["MODEL_ID"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(markdown_cells)}
            ],
            tools=tools,
            temperature=0.0,
            max_tokens=128_000
        )
    except Exception as e:
        fatal(f"API error: {e}")

    tool_calls = response.choices[0].message.tool_calls or []

    updated = 0
    for call in tool_calls:
        if call.function.name != "update_markdown_cells":
            continue
        updates = json.loads(call.function.arguments).get("updates", [])
        for upd in updates:
            idx = upd["cell_number"]
            if idx < 0 or idx >= len(notebook['cells']):
                continue
            cell = notebook['cells'][idx]
            if cell['cell_type'] != 'markdown':
                continue
            if verbose:
                tqdm.write(f"\nUpdating cell {idx}...")
                tqdm.write(f"Before:\n{''.join(cell['source']).strip()}")
                tqdm.write(f"After:\n{upd['improved_content']}")
            cell['source'] = [line + '\n' for line in upd['improved_content'].split('\n')]
            updated += 1

    if updated:
        notebook_path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
        log(f"\n✅ Updated {updated} markdown cells ({updated / len(notebook['cells']) * 100:.2f}%)", 'green')
    else:
        log("⚠️  No cells updated.", 'red')

    return str(notebook_path)

# === CLI Entrypoint ===
def main():
    if len(sys.argv) < 2:
        print("Usage: notebroom notebook.ipynb [repeat_count]")
        sys.exit(1)

    setup_env()
    env_vars = validate_env()
    notebook_path = sys.argv[1]
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1

    for i in range(repeat):
        log(f"\n--- Iteration {i + 1} of {repeat} ---", 'green')
        notebook_path = improve_notebook(notebook_path, env_vars, verbose=True)

if __name__ == "__main__":
    main()