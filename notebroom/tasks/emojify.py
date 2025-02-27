"""Task for adding emojis to markdown headings."""
from notebroom.tasks.base import TextProcessingTask

class EmojifyTask(TextProcessingTask):
    """Task for adding relevant emojis to markdown headings."""
    
    # Define task metadata
    task_id = "emojify"
    requires_llm = True
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = """
        You are an emoji specialist. Your task is to add relevant emojis to markdown headings.
        
        Rules:
        - Only add emojis to heading lines (those starting with # symbols)
        - Choose emojis that are relevant to the heading content
        - Add 1-2 emojis at the start of each heading
        - Do not modify any other content
        - Return the entire text with emojis added only to headings
        """
    
    def get_supported_cell_types(self):
        """Only process markdown cells."""
        return ['markdown']
    
    def process_text(self, text, llm_service=None):
        """Add emojis to headings using LLM."""
        if not text.strip() or not llm_service:
            # Skip empty cells or if no LLM service is available
            return text
        
        # Check if the text has any headings (starts with #)
        has_headings = any(line.strip().startswith('#') for line in text.split('\n'))
        if not has_headings:
            return text
            
        # Send to LLM for emojification
        return llm_service.process_text(self.system_prompt, text)
