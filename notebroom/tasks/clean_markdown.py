"""Task for cleaning markdown cells in a Jupyter notebook."""

from .base import TextProcessingTask

PROMPT = """Your task is to make existing educational content more concise and clear.
Important rules:
- DO NOT add new information or change meaning.
- DO NOT modify section headers.
- FOCUS on making the given text more concise while preserving all information.
- ENSURE clarity and educational value.
- MAINTAIN technical accuracy.
Return ONLY the rewritten markdown cell. Do not include any introductory or concluding remarks.
""".strip()
    
class CleanMarkdownTask(TextProcessingTask):
    """Task for cleaning markdown cells in a Jupyter notebook."""
    
    def __init__(self, config):
        super().__init__(config, self.PROMPT)
