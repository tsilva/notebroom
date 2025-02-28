"""Task registration and management for notebroom."""
import os
import yaml
from typing import Dict, List, Optional, Any, Callable, Type
from pathlib import Path

class TaskRegistry:
    """Registry for available tasks in the system."""
    
    _instance = None
    _tasks = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskRegistry, cls).__new__(cls)
        return cls._instance
    
    def register_task(self, name: str, task_class):
        """Register a task with the registry."""
        self._tasks[name] = task_class
    
    def get_task(self, name: str):
        """Get a task by name."""
        if name not in self._tasks:
            raise ValueError(f"Task '{name}' is not registered. Available tasks: {', '.join(self._tasks.keys())}")
        return self._tasks[name]
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available task names."""
        return list(self._tasks.keys())
    
    def load_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load task definitions from YAML file."""
        if not os.path.exists(yaml_path):
            return {}
            
        with open(yaml_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config.get('tasks', [])
            except yaml.YAMLError:
                return {}

# Global registry instance
registry = TaskRegistry()

# Import all task implementations to ensure they're registered
from .clean_markdown import CleanMarkdownTask
from .emojify import EmojifyTask
from .standardize_indentation import StandardizeIndentationTask
from .fix_colab_links import FixColabLinksTask

# Create backward compatibility mappings
# Maps task names to their implementation classes
TASK_MAP = {
    task_name: registry._tasks[task_name] 
    for task_name in registry._tasks
}

# List of available task names
AVAILABLE_TASKS = registry.get_available_tasks()

# Tasks that require LLM services
# For now, let's assume none of our tasks require LLM by default
# This should be updated based on the actual requirements
LLM_REQUIRED_TASKS = []

# Export for easier imports
__all__ = [
    'registry', 
    'TASK_MAP', 
    'AVAILABLE_TASKS', 
    'LLM_REQUIRED_TASKS',
    'CleanMarkdownTask', 
    'EmojifyTask', 
    'StandardizeIndentationTask', 
    'FixColabLinksTask'
]