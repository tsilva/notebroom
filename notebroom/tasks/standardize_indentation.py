"""Task for standardizing Python code indentation to 2 spaces."""

from nbformat import NotebookNode
from typing import Optional
from colorama import Fore

from .base import Task
from notebroom.utils import log_msg


class StandardizeIndentationTask(Task):
    """Task for standardizing Python code indentation to 2 spaces."""
    
    def __init__(self, config):
        super().__init__(config)
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional = None) -> NotebookNode:
        """Process a code cell to standardize indentation."""
        if cell.cell_type != 'code':
            return cell
            
        source_lines = cell.source.split('\n')
        standardized_lines = []
        
        for line in source_lines:
            if not line.strip():  # Preserve empty lines
                standardized_lines.append(line)
                continue
                
            # Count leading whitespace and determine indentation level
            original_line = line
            leading_whitespace = len(line) - len(line.lstrip())
            
            if leading_whitespace == 0:
                # No indentation, keep the line as is
                standardized_lines.append(line)
                continue
                
            # Extract the indentation part and content part
            indentation = line[:leading_whitespace]
            content = line[leading_whitespace:]
            
            # Calculate indentation level
            if '\t' in indentation:
                # Replace tabs with 4 spaces to count indentation level
                expanded_indent = indentation.replace('\t', ' ' * 4)
                indent_level = len(expanded_indent) // 4
            else:
                # For spaces, estimate the level based on common patterns (2, 4, or 8 spaces)
                if leading_whitespace % 4 == 0:
                    indent_level = leading_whitespace // 4
                elif leading_whitespace % 2 == 0:
                    indent_level = leading_whitespace // 2
                else:
                    # Odd number of spaces, use a best guess
                    indent_level = round(leading_whitespace / 4)
            
            # Create standardized line with 2 spaces per indentation level
            standardized_line = ' ' * (2 * indent_level) + content
            standardized_lines.append(standardized_line)
        
        # Update cell source with standardized indentation
        new_source = '\n'.join(standardized_lines)
        
        if new_source != cell.source:
            log_msg("\nStandardized indentation in code cell:", Fore.CYAN, '🔧')
            log_msg("Original:", Fore.RED, '📄')
            log_msg(cell.source, Fore.RED)
            log_msg("Standardized:", Fore.GREEN, '✨')
            log_msg(new_source, Fore.GREEN)
            log_msg("-" * 80)
            cell.source = new_source
            
        return cell
