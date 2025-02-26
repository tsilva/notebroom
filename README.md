<div align="center">

# 🧹 Notebroom

**A CLI tool for cleaning and processing Jupyter notebooks with LLMs**

</div>

## 📋 Overview

Notebroom is a command-line tool that helps you clean, process, and transform Jupyter notebooks. It uses LLMs (Large Language Models) for text-based tasks and provides utilities for fixing notebook links and exporting notebooks to markdown format.

## ⚡ Features

- 🔄 **Fix Colab Links** - Automatically update GitHub repository links in Colab buttons
- 📝 **Clean Markdown** - Use LLMs to make notebook markdown cells more concise and clear
- 😀 **Emojify** - Add appropriate emojis to markdown cells to make content more engaging
- 📤 **Export to Markdown** - Convert notebooks to specially formatted markdown files for LLM processing

## 🚀 Installation

```bash
# Basic installation
pip install notebroom

# Or with LLM support (required for clean_markdown and emojify)
pip install 'notebroom[llm]'
```

## 🛠️ Usage

### Basic Command Format

```bash
notebroom <notebook-path> <task-name> [options]
```

### Tasks

#### 1. Fix Colab Links

```bash
notebroom path/to/notebook.ipynb fix_colab_links
```

This task detects Colab links in markdown cells and updates them to point to the correct GitHub repository based on your local Git configuration. It works with both Markdown-style and HTML-style Colab links.

#### 2. Clean Markdown

```bash
notebroom path/to/notebook.ipynb clean_markdown
```

Uses an LLM to make existing markdown cells more concise and clear while preserving all information and maintaining technical accuracy.

> ⚠️ Requires `OPENAI_API_KEY` environment variable to be set.

#### 3. Emojify

```bash
notebroom path/to/notebook.ipynb emojify
```

Adds appropriate emojis to markdown cells to make the content more engaging and readable.

> ⚠️ Requires `OPENAI_API_KEY` environment variable to be set.

#### 4. Export to Markdown

```bash
notebroom path/to/notebook.ipynb dump_markdown -o output.md
```

Converts the notebook to a specially formatted markdown file that is optimized for LLM processing. Each cell is enclosed in HTML comments with type and number markers, making it easy for LLMs to reference specific cells.

### Processing Multiple Notebooks

You can process all notebooks in a directory by specifying a directory path:

```bash
notebroom path/to/directory fix_colab_links
```

### Output Options

By default, most tasks modify the notebook in place. To save to a different location:

```bash
notebroom path/to/notebook.ipynb task-name -o path/to/output.ipynb
```

For `dump_markdown`, the default output is a markdown file with the same name as the notebook.

## ⚙️ Configuration

Notebroom can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NOTEBROOM_MODEL` | LLM model to use | `gpt-4o-mini` |
| `NOTEBROOM_MAX_TOKENS` | Maximum tokens for LLM responses | `1000` |
| `NOTEBROOM_TEMPERATURE` | Temperature for LLM responses | `0.2` |
| `NOTEBROOM_NUM_WORKERS` | Number of concurrent workers | `4` |
| `NOTEBROOM_TPM_LIMIT` | Tokens per minute limit | `10000000` |
| `NOTEBROOM_RPM_LIMIT` | Requests per minute limit | `100` |
| `NOTEBROOM_MAX_RETRIES` | Maximum retries for failed LLM calls | `5` |

## 📝 Examples

### Fix Colab links in an entire repository

```bash
notebroom path/to/notebooks fix_colab_links
```

### Clean markdown and save to a new file

```bash
notebroom notebook.ipynb clean_markdown -o cleaned_notebook.ipynb
```

### Export notebook to formatted markdown for LLM analysis

```bash
notebroom complex_notebook.ipynb dump_markdown -o for_llm_analysis.md
```

## 📚 Output Format for LLM Processing

When using `dump_markdown`, the output has special markers that help LLMs understand and reference notebook structure:

```markdown
<!-- NOTEBOOK:example.ipynb -->
# Notebook: example.ipynb

<!-- CELL:MARKDOWN:1 -->
# Example Header
<!-- CELL:MARKDOWN:1:END -->

<!-- CELL:CODE:2 -->
```python
import pandas as pd
```
<!-- CELL:CODE:2:END -->
```

This format makes it easy for LLMs to reference specific cells by type and number (e.g., "In CODE:2, you should add...").

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
