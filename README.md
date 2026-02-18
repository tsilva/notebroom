> [!WARNING]
> ## Archived
> This project is archived and no longer maintained.
>
> It has been deprecated — modern AI agents can now handle notebook enhancement tasks more effectively and flexibly than this tool.

<div align="center">
  <img src="logo.png" alt="notebroom" width="512"/>

  # notebroom

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://python.org)
  [![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-blueviolet)](https://openrouter.ai)

  **🧹 Polish your Jupyter notebooks with AI-powered markdown enhancement 📓**

  [Installation](#installation) · [Usage](#usage) · [Configuration](#configuration)
</div>

## Overview

Writing clear, engaging Jupyter notebooks is hard. Markdown cells accumulate jargon, inconsistent formatting, and redundant explanations over time. **notebroom** runs AI improvement passes over your notebook's markdown cells — expanding thin explanations, tightening wordy prose, and formatting code — while leaving code cells completely untouched.

The notebook file is updated in place. No manual editing. No cell juggling.

## ✨ Features

- **🔄 Multiple improvement passes** — Expand concepts, reduce redundancy, or format code
- **🛡️ Code cell safe** — Only markdown cells are modified; code cells remain untouched
- **🎨 Visual diff output** — See original vs updated content with color-coded output
- **🎯 Selective tasks** — Run all passes or pick specific ones
- **⚡ Code formatting** — Optional autopep8 integration for Python code cells

## 📦 Installation

```bash
pipx install . --force
```

Or with pip:

```bash
pip install .
```

## 🚀 Usage

### First Run Setup

On first run, notebroom creates `~/.notebroom/` with an example `.env` file. Add your API credentials:

```bash
# ~/.notebroom/.env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_ID=anthropic/claude-3.7-sonnet:thinking
```

### Running notebroom

```bash
# Run all improvement passes
notebroom path/to/notebook.ipynb

# Run specific tasks only
notebroom path/to/notebook.ipynb --tasks expand contract

# Just format code cells
notebroom path/to/notebook.ipynb --tasks format-code
```

### Available Tasks

| Task | Description |
|------|-------------|
| `expand` | Enhances depth and conceptual flow of thin explanations |
| `contract` | Tightens wordy prose and reduces redundancy |
| `format-code` | Formats Python code cells via autopep8 (indentation fixes only) |

## ⚙️ Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | ✅ | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | ✅ | API endpoint (default: `https://openrouter.ai/api/v1`) |
| `MODEL_ID` | ✅ | AI model (e.g., `anthropic/claude-3.7-sonnet:thinking`) |

### How It Works

1. **Parse** — Converts `.ipynb` to structured text with cell metadata
2. **Analyze** — Sends notebook representation to the configured AI model
3. **Update** — Model returns structured cell updates via function calling
4. **Write** — Changes are applied back to the original notebook file

## 🧪 Testing

Run quality checks on processed notebooks:

```bash
python test.py path/to/notebook.ipynb
```

Checks include:
- ✅ No consecutive header-only markdown cells
- ✅ No HTML comments in cells
- ✅ Valid Colab badge format (if present)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
