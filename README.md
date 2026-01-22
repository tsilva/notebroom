<div align="center">
  <img src="logo.png" alt="notebroom" width="256"/>

  # notebroom

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://python.org)
  [![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-blueviolet)](https://openrouter.ai)

  **🧹 AI-powered Jupyter notebook enhancement that makes your markdown cells clearer and more engaging**

  [Installation](#installation) · [Usage](#usage) · [Configuration](#configuration)
</div>

## Overview

notebroom analyzes the markdown cells in your Jupyter notebooks and uses AI (via OpenRouter) to enhance their clarity, conciseness, and engagement. It intelligently identifies cells that could benefit from improvement while leaving code cells untouched.

The tool converts notebooks to a structured markdown representation, sends relevant cells to the configured AI model using function calling, and updates the original notebook file with improved content.

## Features

- **Multiple improvement passes** - Expand concepts, reduce redundancy, or format code
- **Selective updates** - Only markdown cells are modified; code cells remain untouched
- **Visual feedback** - See original vs updated content with color-coded diff output
- **Configurable tasks** - Run all passes or pick specific ones
- **Code formatting** - Optional autopep8 integration for Python code cells

## Installation

```bash
pipx install . --force
```

Or with pip:

```bash
pip install .
```

## Usage

### First Run Setup

On first run, notebroom creates a configuration directory at `~/.notebroom/` with an example `.env` file. Edit this file to add your API credentials:

```bash
# ~/.notebroom/.env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_ID=anthropic/claude-3.7-sonnet:thinking
```

### Running notebroom

```bash
# Run all improvement passes (expand, contract, format-code)
notebroom path/to/notebook.ipynb

# Run specific tasks only
notebroom path/to/notebook.ipynb --tasks expand contract

# Just format code cells
notebroom path/to/notebook.ipynb --tasks format-code
```

The notebook file is updated in place.

### Available Tasks

| Task | Description |
|------|-------------|
| `expand` | Conceptual Expansion - enhances depth and flow |
| `contract` | Conciseness & Redundancy - tightens wording and style |
| `format-code` | Python code formatting via autopep8 (indentation only) |

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | Yes | API endpoint (default: `https://openrouter.ai/api/v1`) |
| `MODEL_ID` | Yes | AI model to use (e.g., `anthropic/claude-3.7-sonnet:thinking`) |

### How It Works

1. **Parse notebook** - Converts `.ipynb` to structured text with cell metadata
2. **AI analysis** - Sends notebook representation to configured model
3. **Function calling** - Model returns structured updates via `update_markdown_cells`
4. **Apply changes** - Updates are written back to the notebook file

## Testing

Run quality checks on processed notebooks:

```bash
python test.py path/to/notebook.ipynb
```

Tests include:
- No consecutive header-only markdown cells
- No HTML comments in cells
- Valid Colab badge format (if present)

## License

This project is licensed under the [MIT License](LICENSE).
