# Notebroom

A tool to improve Jupyter notebook markdown cells using AI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

Notebook Refiner takes your Jupyter notebooks and polishes the markdown cells to make them clearer, more concise, and more engaging. Powered by OpenAI’s API, it analyzes your explanatory text, suggests meaningful improvements, and updates your notebook—all while leaving code cells untouched. Perfect for AI/ML tutorials or any notebook where great documentation matters.

## Features

- **Smart Analysis**: Evaluates markdown cells for clarity, conciseness, and engagement.
- **AI-Powered Improvements**: Suggests enhancements that preserve your original intent.
- **Selective Refinement**: Skips the first cell, headings, and already well-written content.
- **Code-Safe**: Never modifies code cells—just the markdown.
- **Terminal Feedback**: Color-coded output shows original vs. improved content.

## Requirements

- **Python**: 3.6 or higher
- **OpenAI API Key**: Required for AI enhancements
- **Dependencies**: 
  - `openai`
  - `python-dotenv`
  - `tqdm`
  - `pathlib`
  - `shutil`

## Installation

```bash
# Install from the local directory
pip install -e .

# Or install directly from git
# pip install git+https://github.com/username/notebroom.git
```

## Usage

```bash
# Basic usage
notebroom path/to/notebook.ipynb

# Run multiple improvement iterations
notebroom path/to/notebook.ipynb 3

# Show verbose output
notebroom path/to/notebook.ipynb --verbose

# Combine options
notebroom path/to/notebook.ipynb 2 --verbose
```

## Environment Variables

Create a `.env` file with the following variables:

```
OPENROUTER_BASE_URL=your_base_url
OPENROUTER_API_KEY=your_api_key
MODEL_ID=your_model_id
```

Run the tool with a single command:

```bash
python main.py path/to/your/notebook.ipynb
```

- The script processes your notebook and updates it in place.
- Watch the terminal for colorful feedback on what’s being improved.

## How It Works

1. **Conversion**: Transforms your `.ipynb` file into a markdown-like format for analysis.
2. **Analysis**: Sends markdown cells (excluding the first cell and headings) to the AI model.
3. **Enhancement**: Applies AI-suggested improvements only to cells that need it—think clarity, brevity, or engagement boosts.
4. **Update**: Saves the refined notebook with updated markdown cells.

## Example Output

In your terminal, you’ll see something like this:

```
--- Original cell 2 ---
This is a long and confusing explanation about the code that follows it.
--- Improved cell 2 ---
Here’s a concise explanation of the upcoming code.
```

## Contributing

Love the idea? Want to make it better? Contributions are welcome!

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-idea`).
3. Commit your changes (`git commit -m "Add amazing idea"`).
4. Push to the branch (`git push origin feature/amazing-idea`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE)—free to use, modify, and share.
