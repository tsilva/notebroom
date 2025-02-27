"""Task implementations for Notebroom."""
import importlib
import inspect
import os
import sys
import pkgutil
from typing import Dict, List, Type

from .base import Task

# Dictionary that will be populated with discovered tasks
TASK_MAP = {}
AVAILABLE_TASKS = []
LLM_REQUIRED_TASKS = []

def register_tasks() -> None:
    """Discover and register all Task subclasses automatically."""
    global TASK_MAP, AVAILABLE_TASKS, LLM_REQUIRED_TASKS
    
    # Get the directory of the current package
    package_dir = os.path.dirname(__file__)
    
    # Import all modules in the tasks package
    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
        if not is_pkg and module_name != 'base':
            module = importlib.import_module(f"{__name__}.{module_name}")
            
            # Find all Task subclasses in the module
            for _, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Task) and 
                    obj != Task and
                    hasattr(obj, 'task_id') and 
                    obj.task_id):
                    
                    # Register the task
                    TASK_MAP[obj.task_id] = obj
                    
                    # Add to LLM required tasks if specified
                    if obj.requires_llm:
                        LLM_REQUIRED_TASKS.append(obj.task_id)
    
    # Create the list of available tasks
    AVAILABLE_TASKS = list(TASK_MAP.keys())

# Execute the discovery and registration process
register_tasks()