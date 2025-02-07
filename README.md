# notebroom 📝✨

<p align="center">
  <img src="logo.jpg" alt="notebroom logo" width="400"/>
</p>

A tool to process Jupyter notebook cells using LLMs. It supports various tasks to enhance your notebooks. 🪄

## Installation 🔧

1.  Clone the repository:

    ```bash
    git clone https://github.com/tsilva/notebroom.git
    cd notebroom
    ```
2.  Install with pip (uses `pyproject.toml` for dependencies):

    ```bash
    pip install .
    ```
3.  Copy the example `.env.example` file to your home directory or project directory and modify it with your OpenAI API key and any other desired configurations:

    ```bash
    cp .env.example ~/.notebroom.env
    # or
    cp .env.example ./.notebroom.env
    ```

    Then, edit the file to add your OpenAI API key and other missing configurations:

    ```bash
    nano ~/.notebroom.env
    # or
    nano ./.notebroom.env
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
