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
# Clone the repository
git clone https://github.com/tsilva/notebroom.git
cd notebroom

# Install the package
pip install -e .
```

## Setup with OpenRouter

Notebroom uses [OpenRouter](https://openrouter.ai/) to access Claude-3.7-Sonnet and other LLMs.

1. Create an account at [OpenRouter](https://openrouter.ai/)
2. Generate an API key from your dashboard
3. Set the API key as an environment variable:

```bash
# Add this to your .env file
OPENROUTER_API_KEY=your_api_key_here
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | (required) |
| `MODEL` | Model to use | `anthropic/claude-3-7-sonnet` |
| `MAX_TOKENS` | Max tokens in response | `1000` |
| `TEMPERATURE` | Temperature setting | `0.2` |
| `BATCH_SIZE` | Batch size for processing | `4` |
| `NUM_WORKERS` | Number of worker threads | `8` |

## 🛠️ Usage

### Basic Command Format

```bash
notebroom TASK NOTEBOOK_PATH [options]
```

For example:

```bash
notebroom fix_colab_links notebook.ipynb
```

Run `notebroom -h` to see all available options.

### Available Tasks

#### 1. Fix Colab Links (`fix_colab_links`)

```bash
notebroom fix_colab_links path/to/notebook.ipynb
```

This task detects Colab links in markdown cells and updates them to point to the correct GitHub repository based on your local Git configuration. It works with both Markdown-style and HTML-style Colab links.

#### 2. Clean Markdown (`clean_markdown`)

```bash
notebroom clean_markdown path/to/notebook.ipynb
```

Uses an LLM to make existing markdown cells more concise and clear while preserving all information and maintaining technical accuracy.

> ⚠️ Requires `OPENROUTER_API_KEY` environment variable to be set.

#### 3. Emojify (`emojify`)

```bash
notebroom emojify path/to/notebook.ipynb
```

Adds appropriate emojis to markdown cells to make the content more engaging and readable.

> ⚠️ Requires `OPENROUTER_API_KEY` environment variable to be set.

#### 4. Export to Markdown (`dump_markdown`)

```bash
notebroom dump_markdown path/to/notebook.ipynb -o output.md
```

Converts the notebook to a specially formatted markdown file that is optimized for LLM processing. Each cell is enclosed in HTML comments with type and number markers, making it easy for LLMs to reference specific cells.

### Processing Multiple Notebooks

You can process all notebooks in a directory by specifying a directory path:

```bash
notebroom fix_colab_links path/to/directory/
```

### Output Options

By default, most tasks modify the notebook in place. To save to a different location:

```bash
notebroom task-name path/to/notebook.ipynb -o path/to/output.ipynb
```

For `dump_markdown`, the default output is a markdown file with the same name as the notebook.

## ⚙️ Configuration

Notebroom can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL` | LLM model to use | `anthropic/claude-3-7-sonnet` |
| `MAX_TOKENS` | Maximum tokens for LLM responses | `1000` |
| `TEMPERATURE` | Temperature for LLM responses | `0.2` |
| `NUM_WORKERS` | Number of concurrent workers | `8` |
| `TPM_LIMIT` | Tokens per minute limit | `100000` |
| `RPM_LIMIT` | Requests per minute limit | `60` |
| `MAX_RETRIES` | Maximum retries for failed LLM calls | `5` |

## 📝 Examples

### Fix Colab links in an entire repository

```bash
notebroom fix_colab_links path/to/notebooks/
```

### Clean markdown and save to a new file

```bash
notebroom clean_markdown notebook.ipynb -o cleaned_notebook.ipynb
```

### Export notebook to formatted markdown for LLM analysis

```bash
notebroom dump_markdown complex_notebook.ipynb -o for_llm_analysis.md
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

## Performance Optimizations

This version is optimized for throughput when working with Claude-3.7-Sonnet:

- Batched cell processing for parallel execution
- Improved connection pooling and request handling
- Intelligent rate limiting for OpenRouter's API constraints
- Automatic retry with exponential backoff

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
