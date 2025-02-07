# notebroom 📝✨

A tool to clean up Jupyter notebook markdown cells using LLMs. It processes each markdown cell to make text more concise while preserving the original meaning and formatting. 🪄

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/tsilva/notebroom.git
cd notebroom
```

2. Install with pip (will automatically use pyproject.toml):
```bash
pip install -e .
```

3. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > ~/.notebroom.env
```

## Configuration 🔧

Configure notebroom using environment variables in your `.env` file:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional (showing defaults)
NOTEBROOM_MODEL=gpt-4o           # LLM model to use
NOTEBROOM_MAX_TOKENS=4000        # Maximum tokens for context
NOTEBROOM_KEEP_RECENT=3          # Number of recent cells to keep in full
NOTEBROOM_TEMPERATURE=0.2        # LLM temperature
NOTEBROOM_WINDOW_SIZE=10         # Maximum number of cells to consider for context
```

## Usage 🚀

After installation, you can use notebroom from anywhere:

```bash
notebroom your_notebook.ipynb
```

The tool will process your notebook and create a new file with cleaned markdown cells. The output file will be saved in the same directory as your input file, with `.clean.ipynb` added before the extension.

For example:
- Input:  `lecture_notes.ipynb` 📓
- Output: `lecture_notes.clean.ipynb` ✨

Each markdown cell will be rewritten to be more concise while maintaining the original information and formatting. 🎯
