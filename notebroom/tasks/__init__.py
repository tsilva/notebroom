"""Task implementations for Notebroom."""

from .base import Task, TextProcessingTask
from .clean_markdown import CleanMarkdownTask
from .emojify import EmojifyTask
from .fix_colab_links import FixColabLinks
from .dump_notebook import DumpNotebookTask
from .standardize_indentation import StandardizeIndentationTask

# Dictionary mapping task names to task classes
TASK_MAP = {
    "clean_markdown": CleanMarkdownTask,
    "emojify": EmojifyTask,
    "fix_colab_links": FixColabLinks,
    "dump_markdown": DumpNotebookTask,
    "standardize_indentation": StandardizeIndentationTask
}

# List of available tasks
AVAILABLE_TASKS = list(TASK_MAP.keys())

# List of tasks that require an LLM service
LLM_REQUIRED_TASKS = ["clean_markdown", "emojify"]
