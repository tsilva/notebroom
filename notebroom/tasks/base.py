"""Base task classes for Notebroom."""

from abc import ABC, abstractmethod
from typing import Optional, List

from nbformat import NotebookNode


class Task(ABC):
    """Base class for notebook processing tasks."""
    
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def process_cell(self, cell: NotebookNode, llm_service: Optional = None) -> NotebookNode:
        """Process a single notebook cell."""
        pass
    
    def process_notebook(self, infile: str, outfile: str, 
                         llm_service: Optional = None, nb: NotebookNode = None) -> None:
        """Process an entire notebook."""
        for cell in nb.cells:
            self.process_cell(cell, llm_service)


class TextProcessingTask(Task):
    """Base class for text-processing tasks that use LLMs."""
    
    def __init__(self, config, system_prompt: str):
        super().__init__(config)
        self.system_prompt = system_prompt
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional = None) -> NotebookNode:
        """Process a markdown cell using the LLM service."""
        from notebroom.utils import is_header_only, log_msg
        from colorama import Fore
        
        if cell.cell_type != 'markdown' or llm_service is None:
            return cell
            
        if is_header_only(cell.source):
            log_msg(f"\nSkipping header cell:\n{cell.source}", Fore.YELLOW, '📌')
            return cell
        
        log_msg(f"\nProcessing cell:", Fore.CYAN, '📝')
        try:
            new_text = llm_service.process_text(self.system_prompt, cell.source)
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, '✨')
            log_msg(new_text, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = new_text
        except Exception as e:
            log_msg(f"Error processing cell: {e}", Fore.RED, '❌')
        
        return cell
    
    def process_batch(self, cells: List[NotebookNode], llm_service) -> List[NotebookNode]:
        """Process multiple markdown cells in a batch for better throughput."""
        from notebroom.utils import is_header_only, log_msg
        from colorama import Fore
        
        # Filter cells that need processing
        markdown_cells = []
        cell_indices = []
        
        for i, cell in enumerate(cells):
            if cell.cell_type == 'markdown' and not is_header_only(cell.source):
                markdown_cells.append(cell)
                cell_indices.append(i)
        
        if not markdown_cells:
            return cells
        
        # Create batch tasks
        tasks = [
            {"system_prompt": self.system_prompt, "user_text": cell.source}
            for cell in markdown_cells
        ]
        
        # Process in batch
        log_msg(f"\nBatch processing {len(tasks)} markdown cells", Fore.CYAN, '📦')
        results = llm_service.process_batch(tasks)
        
        # Update cells with results
        for i, (cell, result) in enumerate(zip(markdown_cells, results)):
            log_msg(f"\nCell {i+1}/{len(markdown_cells)}", Fore.CYAN, '📝')
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Rewritten:", Fore.GREEN, '✨')
            log_msg(result, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = result
        
        # Return the updated cells
        return cells
