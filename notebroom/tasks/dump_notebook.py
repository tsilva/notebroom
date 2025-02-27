"""Task for dumping notebooks as specially formatted markdown for LLM processing."""

import os
from nbformat import NotebookNode
from typing import Optional
from colorama import Fore

from .base import Task
from notebroom.utils import log_msg


class DumpNotebookTask(Task):
    """Task for dumping notebooks as specially formatted markdown for LLM processing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.cell_number = 0
        self.markdown_content = []
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional = None) -> NotebookNode:
        """Process a single cell by converting it to markdown format with special markers."""
        self.cell_number += 1
        
        # Get cell type and content
        cell_type = cell.cell_type.upper()
        cell_content = cell.source
        
        # Create cell marker with cell type and number
        start_marker = f"<!-- CELL:{cell_type}:{self.cell_number} -->"
        end_marker = f"<!-- CELL:{cell_type}:{self.cell_number}:END -->"
        
        # Format content based on cell type
        if cell_type == "CODE":
            formatted_content = f"```python\n{cell_content}\n```"
        else:  # MARKDOWN or other types
            formatted_content = cell_content
        
        # Append formatted cell to markdown content
        self.markdown_content.append(f"{start_marker}\n{formatted_content}\n{end_marker}\n")
        
        return cell
    
    def process_notebook(self, infile: str, outfile: str, 
                         llm_service: Optional = None, nb: NotebookNode = None) -> None:
        """Process entire notebook and output as markdown."""
        # Reset state
        self.cell_number = 0
        self.markdown_content = []
        
        # Get notebook metadata
        notebook_name = os.path.basename(infile)
        metadata = {
            "filename": notebook_name,
            "cell_count": len(nb.cells),
            "kernelspec": nb.metadata.get("kernelspec", {}).get("name", "unknown")
        }
        
        # Create header with notebook info
        header = (
            f"<!-- NOTEBOOK:{notebook_name} -->\n"
            f"# Notebook: {notebook_name}\n\n"
            f"*This is a generated markdown representation of a Jupyter notebook optimized for LLM processing.*\n\n"
            f"**Metadata**: {metadata}\n\n"
        )
        
        # Process all cells
        for cell in nb.cells:
            self.process_cell(cell, llm_service)
            
        # Combine all parts
        full_content = header + "\n".join(self.markdown_content)
        
        # Write to output file
        try:
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write(full_content)
            log_msg(f"Notebook dumped as markdown to {outfile}", Fore.GREEN, '💾')
        except Exception as e:
            log_msg(f"Error writing markdown file {outfile}: {e}", Fore.RED, '❌')
