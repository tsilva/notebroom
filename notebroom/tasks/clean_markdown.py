"""Task for cleaning and formatting markdown cells."""
from notebroom.tasks.base import TextProcessingTask

PROMPT = """
You are a Jupyter Notebook markdown formatter. Your task is to enhance the readability, structure, and correctness of markdown text while keeping as close as possible to the original content, meaning, and flow.

Guidelines:  
- Preserve the original message, intent, and style; make changes only to improve clarity or fix errors.  
- Correct markdown syntax errors (e.g., improper headers, lists, or code blocks).  
- Fix typos and grammar mistakes without altering the intended meaning.  
- Use a logical heading hierarchy (# for h1, ## for h2, ### for h3).  
- Ensure consistent spacing between sections (e.g., one blank line between elements).  
- Break text into paragraphs where it improves readability, while maintaining the original flow.  
- Properly format code blocks using triple backticks (```).  
- Fix broken or misaligned lists and indentation.  
- Retain all original links, images, and embedded content unchanged.  
- If the original text is already clear, grammatically correct, and well-formatted, return it unchanged.  
- Provide concise output focused solely on the improved markdown text.
""".strip()

class CleanMarkdownTask(TextProcessingTask):
    """Task for cleaning and formatting markdown cells in notebooks."""
    
    # Define task metadata
    task_id = "clean_markdown"
    requires_llm = True
    
    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = PROMPT
    
    def get_supported_cell_types(self):
        """Only clean markdown cells."""
        return ['markdown']
    
    def should_skip_cell(self, cell):
        """Check if the markdown cell should be skipped from processing.
        
        Args:
            cell: The notebook cell to check
            
        Returns:
            bool: True if the cell should be skipped
        """
        # Skip non-markdown cells (redundant with get_supported_cell_types, but for safety)
        if cell.get('cell_type') != 'markdown':
            return True
            
        # Skip empty cells
        text = cell.get('source', '').strip()
        if not text:
            return True
            
        # Skip Google Colab badge cells (typically at the top)
        colab_patterns = [
            "[![Open In Colab]",
            "colab.research.google.com",
            "badges/colab-badge.svg",
            "Open in Colab"
        ]
        
        for pattern in colab_patterns:
            if pattern in text:
                return True
                
        # Skip metadata-like cells (often formatted in a specific way)
        metadata_patterns = [
            "<!-- METADATA",
            "<!-- NOTEBOOK-METADATA",
            "<!-- CELL-METADATA"
        ]
        
        for pattern in metadata_patterns:
            if text.startswith(pattern):
                return True
                
        return False
    
    def process_text(self, text, llm_service=None):
        """Clean markdown text using LLM."""
        if not text.strip() or not llm_service:
            # Skip empty cells or if no LLM service is available
            return text
        
        # Send to LLM for cleaning
        return llm_service.process_text(self.system_prompt, text)
    
    def process_batch(self, cells, llm_service=None):
        """Process multiple cells in batch for better throughput."""
        if not llm_service:
            # Fall back to individual processing if no LLM service
            super().process_batch(cells, llm_service)
            return
        
        # Filter for markdown cells only, excluding those that should be skipped
        markdown_cells = [cell for cell in cells if cell['cell_type'] == 'markdown' and not self.should_skip_cell(cell)]
        if not markdown_cells:
            return
        
        # Prepare batch requests
        batch_tasks = []
        for cell in markdown_cells:
            batch_tasks.append({
                'system_prompt': self.system_prompt,
                'user_text': cell['source']
            })
        
        if not batch_tasks:
            return
            
        # Process all in batch
        results = llm_service.process_batch(batch_tasks)
        
        # Update cells with results
        result_index = 0
        for cell in markdown_cells:
            cell['source'] = results[result_index]
            result_index += 1
