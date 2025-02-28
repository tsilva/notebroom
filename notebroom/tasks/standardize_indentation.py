"""Task for standardizing Python code indentation to 2 spaces."""

from nbformat import NotebookNode
from typing import Optional
from colorama import Fore

from .base import Task, BaseTask
from notebroom.utils import log_msg


class StandardizeIndentationTask(BaseTask):
    """Task to standardize Python code indentation to 2 spaces."""
    
    task_name = "standardize_indentation"
    
    def run(self, notebook_content):
        """
        Standardize Python code indentation in notebook cells.
        
        Args:
            notebook_content: The notebook content to process
            
        Returns:
            Processed notebook content with standardized indentation
        """
        # Implementation for standardizing indentation
        # This is a placeholder - implement the actual logic based on your requirements
        for cell in notebook_content.get('cells', []):
            if cell['cell_type'] == 'code':
                cell['source'] = self._standardize_indentation(cell['source'])
        
        return notebook_content
    
    def _standardize_indentation(self, source):
        """
        Convert indentation to 2 spaces in Python code.
        
        Args:
            source: The source code as a string or list of lines
            
        Returns:
            Source code with standardized indentation
        """
        if not source:
            return source
            
        # Convert list of lines to a single string if needed
        if isinstance(source, list):
            content = ''.join(source)
        else:
            content = source
            
        # Split into lines for processing
        lines = content.split('\n')
        standardized_lines = []
        
        for line in lines:
            # Count leading spaces/tabs
            leading_whitespace = len(line) - len(line.lstrip())
            if leading_whitespace > 0:
                # Determine number of indentation levels (assuming 4 spaces or 1 tab per level)
                if '\t' in line[:leading_whitespace]:
                    # Handle tabs
                    indent_level = line[:leading_whitespace].count('\t')
                    new_indent = '  ' * indent_level
                else:
                    # Handle spaces, assuming original indent is 4 spaces
                    indent_level = leading_whitespace // 4
                    new_indent = '  ' * indent_level
                
                standardized_lines.append(new_indent + line.lstrip())
            else:
                standardized_lines.append(line)
        
        # Return in the same format as input (string or list)
        if isinstance(source, list):
            return [line + '\n' for line in standardized_lines[:-1]] + [standardized_lines[-1]]
        else:
            return '\n'.join(standardized_lines)

# Initialize the task to register it
StandardizeIndentationTask()
