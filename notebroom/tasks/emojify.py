"""Task for adding emojis to markdown cells using LLMs."""

from .base import TextProcessingTask


class EmojifyTask(TextProcessingTask):
    """Task for adding emojis to markdown cells using LLMs."""
    
    PROMPT = """Your task is to add emojis to existing text to make it more engaging.
    Important rules:
    - DO NOT add new information or change meaning.
    - Preserve the original content exactly.
    - Add emojis where they naturally fit to enhance readability and engagement.
    - Use emojis sparingly and appropriately.
    Return ONLY the emojified markdown cell. Do not include any introductory or concluding remarks.
    """.strip()
    
    def __init__(self, config):
        super().__init__(config, self.PROMPT)
