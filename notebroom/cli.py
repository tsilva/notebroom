"""Command line interface for notebroom."""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from .tasks import registry

def find_config():
    """Find the tasks.yaml config file in the current directory or parent directories."""
    current_dir = Path.cwd()
    
    # Look in current directory and parents
    for dir_path in [current_dir] + list(current_dir.parents):
        config_path = dir_path / 'tasks.yaml'
        if config_path.exists():
            return str(config_path)
    
    # Fall back to the example in the package
    package_dir = Path(__file__).parent.parent
    return str(package_dir / 'tasks.yaml.example')

def main(args: Optional[List[str]] = None):
    """Main entry point for the notebroom command line tool."""
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description='Notebroom: Notebook Room service')
    parser.add_argument('task', nargs='?', help='Task to run')
    parser.add_argument('--config', '-c', help='Path to config file (default: search for tasks.yaml)')
    parser.add_argument('--list', '-l', action='store_true', help='List available tasks')
    
    parsed_args = parser.parse_args(args)
    
    # Determine config file path
    config_path = parsed_args.config or find_config()
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # List available tasks if requested
    if parsed_args.list:
        print("Available tasks:")
        for task_name in registry.get_available_tasks():
            print(f"  - {task_name}")
        return 0
    
    # Run specified task
    if not parsed_args.task:
        print("Error: No task specified. Use --list to see available tasks.")
        return 1
    
    # Try to get the task
    try:
        task_class = registry.get_task(parsed_args.task)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return 1
        
    # Load task config from YAML
    task_configs = registry.load_from_yaml(config_path)
    
    # Find the specific task config
    task_config = {}
    for task_entry in task_configs:
        if task_entry.get('name') == parsed_args.task:
            task_config = task_entry
            break
    
    # Create and run the task
    try:
        task = task_class(task_config)
        task.run()
        print(f"Task '{parsed_args.task}' completed successfully.")
        return 0
    except Exception as e:
        print(f"Error running task '{parsed_args.task}': {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
