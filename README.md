# 🧹 notebroom
<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>


🔹 A tool to improve Jupyter notebook markdown cells using AI, making your notebooks clearer and more engaging.

## 📖 Overview

`notebroom` analyzes the markdown cells in your Jupyter notebooks and uses AI (via OpenRouter) to enhance their clarity, conciseness, and engagement. It intelligently identifies cells that could benefit from improvement, leaving code cells and well-written markdown untouched.

The tool converts the notebook to a markdown representation, sends relevant cells to the configured AI model, and then updates the original notebook file with the improved content. It provides feedback in the terminal, showing which cells were updated.

## 🚀 Installation

```bash
pipx install . --force
```

## 🛠️ Usage

Before running, you need to configure your API credentials:

1. `notebroom` will automatically create a configuration directory at `~/.notebroom/` on first run if it doesn't exist.
2. It will copy an example `.env` file into this directory.
3. Edit the `~/.notebroom/.env` file to add your `OPENROUTER_API_KEY` and customize `MODEL_ID` or `OPENROUTER_BASE_URL` if needed.

Once configured, you can run the tool:

```bash
# Improve a notebook (single pass)
notebroom path/to/your/notebook.ipynb

# Improve a notebook with multiple passes (e.g., 3 iterations)
notebroom path/to/your/notebook.ipynb 3
```

The notebook file will be updated in place.

## 📄 License

This project is licensed under the [MIT License](LICENSE).