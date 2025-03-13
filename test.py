import unittest
import nbformat
import re
from pathlib import Path
import argparse

class JupyterNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the notebook file path and load the notebook once for all tests."""
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Test a Jupyter Notebook for specific conditions.")
        parser.add_argument('notebook_path', type=str, help="Path to the Jupyter Notebook file to test")
        args = parser.parse_args()

        cls.notebook_path = Path(args.notebook_path).resolve()  # Get absolute path
        cls.notebook = cls.load_notebook()

    @classmethod
    def load_notebook(cls):
        """Load the Jupyter notebook from the specified path."""
        if not cls.notebook_path.exists():
            raise FileNotFoundError(f"Notebook file not found: {cls.notebook_path}")
        
        with open(cls.notebook_path, 'r', encoding='utf-8') as f:
            return nbformat.read(f, as_version=4)

    def test_no_consecutive_headers(self):
        """Test that no two header cells appear one after the other in markdown cells."""
        cells = self.notebook.cells
        markdown_cells = [cell for cell in cells if cell.cell_type == 'markdown']
        
        for i in range(len(markdown_cells) - 1):
            current_cell = markdown_cells[i].source.strip()
            next_cell = markdown_cells[i + 1].source.strip()
            
            # Check if current cell starts with a header
            current_is_header = bool(re.match(r'^#{1,6}\s', current_cell))
            next_is_header = bool(re.match(r'^#{1,6}\s', next_cell))
            
            # If both are headers and cells are contiguous in the notebook
            if current_is_header and next_is_header:
                # Find their indices in the original cell list to check if contiguous
                current_idx = cells.index(markdown_cells[i])
                next_idx = cells.index(markdown_cells[i + 1])
                if next_idx == current_idx + 1:
                    self.fail(f"Consecutive headers found at cells {current_idx} and {next_idx}")

    def test_no_html_comments(self):
        """Test that no cell contains HTML comments (<!-- -->)."""
        for idx, cell in enumerate(self.notebook.cells):
            source = cell.source
            if '<!--' in source:
                self.fail(f"HTML comment found in cell {idx}: {source}")

    def test_first_cell_colab_badge(self):
        """Test that the first cell is a markdown cell with a Colab badge and correct href path."""
        if not self.notebook.cells:
            self.fail("Notebook has no cells")

        first_cell = self.notebook.cells[0]
        
        # Check if the first cell is a markdown cell
        if first_cell.cell_type != 'markdown':
            self.fail("First cell must be a markdown cell")

        # Expected Colab badge format
        expected_badge = (
            '<a href="https://colab.research.google.com/github/tsilva/aiml-notebooks/blob/main/'
            'misc/wip-generate-mnist-with-vaes.ipynb" target="_parent">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
        )

        # Clean the source for comparison (remove extra spaces, newlines)
        actual_source = re.sub(r'\s+', ' ', first_cell.source.strip())

        # Check if the badge format matches (ignoring extra whitespace)
        expected_badge_clean = re.sub(r'\s+', ' ', expected_badge)
        if actual_source != expected_badge_clean:
            self.fail(f"First cell does not match expected Colab badge format.\nExpected: {expected_badge_clean}\nFound: {actual_source}")

        # Extract the href from the actual source
        href_match = re.search(r'href="([^"]+)"', first_cell.source)
        if not href_match:
            self.fail("No href found in the Colab badge")

        href = href_match.group(1)

        # Compute the expected relative path from /blob/main/
        # Assume the repo root is the directory containing 'blob/main'
        # Find the position of 'blob/main' in the path and extract the relative path from there
        repo_root = None
        path_parts = list(self.notebook_path.parts)
        
        # Look for 'blob/main' in the path or infer the repo structure
        try:
            main_idx = path_parts.index('blob')  # Look for 'blob' directory
            if path_parts[main_idx + 1] == 'main':
                repo_root = Path(*path_parts[:main_idx + 2]).resolve()
        except (ValueError, IndexError):
            # If 'blob/main' isn't found, assume the repo root is one level above the notebook
            repo_root = self.notebook_path.parent

        # Compute the relative path from /blob/main/
        try:
            relative_path = self.notebook_path.relative_to(repo_root)
        except ValueError:
            self.fail(f"Could not compute relative path of notebook from {repo_root}")

        expected_href_path = f"https://colab.research.google.com/github/tsilva/aiml-notebooks/blob/main/{relative_path.as_posix()}"
        
        if href != expected_href_path:
            self.fail(f"Colab badge href path does not match notebook path.\nExpected href: {expected_href_path}\nFound href: {href}")

if __name__ == '__main__':
    unittest.main(argv=[''])  # Pass empty argv to avoid unittest consuming script args