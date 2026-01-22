# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**notebroom** is a Jupyter notebook enhancement tool that uses AI to improve markdown cells. It processes notebooks through multiple AI-powered passes to enhance clarity, conciseness, and engagement while preserving code cells and notebook structure.

The tool:
- Converts notebooks to structured markdown representation with cell metadata
- Uses OpenRouter API to apply improvement passes via function calling
- Updates markdown cells in-place, leaving code cells untouched
- Supports multiple improvement strategies: expansion, contraction, and code formatting

## Commands

### Development & Testing

```bash
# Install the package locally with pipx (recommended)
pipx install . --force

# Run the tool (single pass with all tasks)
notebroom path/to/notebook.ipynb

# Run specific tasks only
notebroom path/to/notebook.ipynb --tasks expand contract

# Run notebook quality tests
python test.py path/to/notebook.ipynb
```

### Available Tasks

The `--tasks` flag accepts these values (from `PASS_MAP` in main.py:20-24):
- `expand`: Conceptual Expansion pass
- `contract`: Conciseness & Redundancy reduction
- `format-code`: Python code formatting (autopep8)

## Architecture

### Core Pipeline

The improvement process follows this flow:

1. **Configuration (`setup_env`)**: Loads API credentials from `~/.notebroom/.env`, creates config directory on first run
2. **Notebook Parsing (`extract_cells`)**: Converts `.ipynb` to structured text format with delimiters:
   - Cell boundaries marked with `<|CELL_SEPARATOR|>`
   - Headers include cell number, type, execution count, tags
   - Code cells wrapped in markdown code blocks
   - Outputs included but ignored by AI
3. **AI Passes (`apply_pass`)**: Each task loads specific prompts from `configs/` and:
   - Combines `base_system_prompt.md` + task-specific prompt
   - Sends structured notebook text to OpenRouter model
   - Uses function calling (`update_markdown_cells`) to get structured updates
   - Updates notebook dict and cleaned_cells list
4. **Output (`improve_notebook`)**: Writes modified notebook back using nbformat

### Key Data Structures

- **notebook** (dict): Raw notebook JSON loaded from file, modified in-place during passes
- **cleaned_cells** (list): Simplified cell metadata `[{"cell_number", "cell_type", "content"}]` for tracking
- **notebook_text** (str): Structured markdown representation sent to AI, remains unchanged across passes

### Prompt System

Prompts in `src/notebroom/configs/`:
- `base_system_prompt.md`: Core instructions for cell selection, tool usage, constraints
- `expand.prompt.md`: Enhances conceptual depth and flow
- `contract.prompt.md`: Tightens wording and polishes style
- `.env.example`: Template copied to `~/.notebroom/.env` on first run

All prompts are loaded via `load_prompt()` which enforces the `configs/` directory structure.

### Cell Update Logging

When cells are modified, `log_cell_update()` (main.py:115-121) prints:
- Original content in red
- Updated content in green
This helps track what changes were made during each pass.

### Test Suite

`test.py` provides unittest-based quality checks:
- `test_no_consecutive_headers`: Prevents adjacent header-only markdown cells
- `test_no_html_comments`: Ensures no HTML comments in cells
- `test_first_cell_colab_badge`: Validates Colab badge format and path

Note: Tests are run manually via `python test.py <notebook>`, not integrated into a test runner.

## Important Implementation Details

- **Immutability of notebook_text**: The structured markdown representation is created once and never modified, even though notebook cells change across passes. This means later passes see stale content in the notebook_text parameter.
- **Function calling contract**: AI must use exact cell numbers from `<|CELL_HEADER|>` tags and only update markdown cells (enforced in main.py:192-193).
- **Code formatting**: The `format-code` task bypasses AI and uses autopep8 directly with conservative settings (only indentation fixes, main.py:42-46).
- **Cell source format**: Notebook cells store source as list of strings with `\n` suffixes, assembled via `''.join(cell['source'])`.
- **Temperature 0**: All AI calls use `temperature=0.0` for deterministic outputs (main.py:175).

## Configuration

Environment variables (loaded from `~/.notebroom/.env`):
- `OPENROUTER_API_KEY`: Required for OpenRouter API access
- `OPENROUTER_BASE_URL`: API endpoint (default: https://openrouter.ai/api/v1)
- `MODEL_ID`: AI model to use (example: anthropic/claude-3.7-sonnet:thinking)

The tool validates all three are present before proceeding (main.py:70-72).

## Project Maintenance

- **README.md must be kept up to date** with any significant project changes
- Code is auto-formatted via autopep8 when using the `format-code` task
- Prompt modifications should maintain the constraint that cells are never added/removed
