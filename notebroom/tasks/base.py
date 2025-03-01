"""Base class for all notebroom tasks."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseTask(ABC):
    """Base class that all tasks should inherit from."""
    
    task_name = None  # Should be overridden by subclasses
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Register this task with the registry
        from . import registry
        if self.task_name:
            registry.register_task(self.task_name, self.__class__)
    
    @abstractmethod
    def run(self, notebook_content):
        """Run the task on a notebook. Must be implemented by subclasses."""
        pass


class Task(BaseTask):
    """Base abstract class for all notebook processing tasks."""
    
    # Class attributes that define task metadata
    task_id = None  # Must be defined by subclasses
    requires_llm = False  # Whether this task requires LLM services
    
    def __init__(self, config):
        """Initialize the task with configuration."""
        super().__init__(config)
        
        # Validate that task_id is defined
        if self.__class__.task_id is None:
            raise ValueError(f"Task {self.__class__.__name__} must define a task_id class attribute")
    
    @abstractmethod
    def process_cell(self, cell: Dict[str, Any], llm_service=None) -> None:
        """Process a single notebook cell."""
        pass
    
    def process_notebook(self, input_path: str, output_path: str, llm_service=None, notebook=None) -> None:
        """Process an entire notebook.
        
        Default implementation processes each cell individually.
        Override this method for tasks that need to process the notebook as a whole.
        """
        # Default implementation doesn't do anything special at the notebook level
        pass
    
    def run(self, notebook_content):
        """Run the task on a notebook."""
        # Process each cell in the notebook
        for cell in notebook_content.get('cells', []):
            self.process_cell(cell, None)  # No LLM service by default
        return notebook_content


class TextProcessingTask(Task):
    """Base class for tasks that process text in cells."""
    
    @abstractmethod
    def process_text(self, text: str, llm_service=None) -> str:
        """Process the text content of a cell."""
        return text
    
    def process_cell(self, cell: Dict[str, Any], llm_service=None) -> None:
        """Process a single notebook cell by updating its source text."""
        # Only process specific cell types by default
        if cell['cell_type'] in self.get_supported_cell_types():
            cell['source'] = self.process_text(cell['source'], llm_service)
    
    def get_supported_cell_types(self) -> List[str]:
        """Return a list of cell types this task supports."""
        return ['markdown', 'code']
    
    def process_batch(self, cells: List[Dict[str, Any]], llm_service=None) -> None:
        """Process a batch of cells at once for better throughput."""
        # Only implemented in some subclasses that support batch processing
        for cell in cells:
            self.process_cell(cell, llm_service)
