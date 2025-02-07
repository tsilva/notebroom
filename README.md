# notebroom 📝✨

A tool to clean up Jupyter notebook markdown cells using LLMs. It processes each markdown cell to make text more concise while preserving the original meaning and formatting. 🪄

## Setup 🛠️

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a new virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Usage 🚀

```bash
python main.py your_notebook.ipynb
```

The tool will process your notebook and create a new file with cleaned markdown cells. The output file will be saved in the same directory as your input file, with `.clean.ipynb` added before the extension.

For example:
- Input:  `lecture_notes.ipynb` 📓
- Output: `lecture_notes.clean.ipynb` ✨

Each markdown cell will be rewritten to be more concise while maintaining the original information and formatting. 🎯
