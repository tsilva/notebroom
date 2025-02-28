"""Task for standardizing Python code indentation to 4 spaces per level."""

from nbformat import NotebookNode
from typing import Optional
from colorama import Fore

from .base import Task, BaseTask
from notebroom.utils import log_msg


class StandardizeIndentationTask(BaseTask):
    """Task to standardize Python code indentation to 4 spaces per level."""
    
    task_name = "standardize_indentation"
    
    def run(self, notebook_content):
        """
        Standardize Python code indentation in notebook cells to 4 spaces per level.
        
        Args:
            notebook_content: The notebook content to process
            
        Returns:
            Processed notebook content with standardized indentation
        """
        for cell in notebook_content.get('cells', []):
            if cell['cell_type'] == 'code':
                cell['source'] = self._standardize_indentation(cell['source'])
        
        return notebook_content
    
    def _standardize_indentation(self, source):
        """
        Adjust existing indentation to use 4 spaces per level.
        
        Args:
            source: The source code as a string or list of lines
            
        Returns:
            Source code with indentation standardized to 4 spaces per level
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
            # Count leading whitespace (spaces or tabs)
            leading_whitespace = len(line) - len(line.lstrip())
            if leading_whitespace > 0:
                # Count current indent level (assuming 2 spaces or 1 tab per level originally)
                current_indent = line[:leading_whitespace]
                if '\t' in current_indent:
                    indent_level = current_indent.count('\t')
                else:
                    # Assume 2 spaces per level if spaces are used
                    indent_level = leading_whitespace // 2
                
                # Convert to 4 spaces per level
                new_indent = '    ' * indent_level  # 4 spaces per level
                standardized_lines.append(new_indent + line.lstrip())
            else:
                standardized_lines.append(line)
        
        # Return in the same format as input
        if isinstance(source, list):
            return [line + '\n' for line in standardized_lines[:-1]] + [standardized_lines[-1]]
        else:
            return '\n'.join(standardized_lines)

# Initialize the task to register it
StandardizeIndentationTask()