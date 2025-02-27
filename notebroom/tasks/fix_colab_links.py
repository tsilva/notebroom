"""Task for fixing 'Open in Colab' links to point to the correct GitHub repository."""

import os
import re
import subprocess
import urllib.parse
from nbformat import NotebookNode
from typing import Optional, Dict

from colorama import Fore

from .base import Task
from notebroom.utils import log_msg


class FixColabLinks(Task):
    """Task for fixing 'Open in Colab' links to point to the correct GitHub repository."""
    
    def __init__(self, config):
        super().__init__(config)
        self.notebook_path = None
        self.repo_info = None
    
    def find_git_root(self, path: str) -> Optional[str]:
        """Find the root directory of the Git repository containing the given path."""
        current = os.path.abspath(path)
        while current != '/':
            if os.path.exists(os.path.join(current, '.git')):
                return current
            current = os.path.dirname(current)
        return None
    
    def get_repo_info(self, notebook_path: str) -> Optional[Dict[str, str]]:
        """Get the GitHub repository information for a notebook file."""
        if self.repo_info:
            return self.repo_info
            
        git_root = self.find_git_root(os.path.dirname(notebook_path))
        if not git_root:
            log_msg("Could not find Git repository", Fore.RED, '❌')
            return None
            
        try:
            # Get remote URL
            remote_url = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'],
                cwd=git_root, text=True
            ).strip()
            
            # Extract username and repo name
            if remote_url.startswith('git@'):
                match = re.match(r'git@github\.com:([^/]+)/([^.]+)\.?.*', remote_url)
                if match:
                    username, repo = match.groups()
            else:
                match = re.match(r'https?://github\.com/([^/]+)/([^.]+)\.?.*', remote_url)
                if match:
                    username, repo = match.groups()
                else:
                    log_msg(f"Could not parse GitHub URL: {remote_url}", Fore.RED, '❌')
                    return None
                    
            # Get relative path from repo root to notebook
            rel_path = os.path.relpath(notebook_path, git_root)
            
            self.repo_info = {
                'username': username,
                'repo': repo,
                'root': git_root,
                'rel_path': rel_path
            }
            return self.repo_info
            
        except subprocess.CalledProcessError:
            log_msg("Failed to get Git repository info", Fore.RED, '❌')
            return None
    
    def create_colab_url(self, notebook_path: str) -> Optional[str]:
        """Create a correct 'Open in Colab' URL for a notebook file."""
        repo_info = self.get_repo_info(notebook_path)
        if not repo_info:
            return None
            
        encoded_path = urllib.parse.quote(repo_info['rel_path'])
        url = (f"https://colab.research.google.com/github/{repo_info['username']}/"
               f"{repo_info['repo']}/blob/main/{encoded_path}")
        return url
    
    def fix_colab_links(self, cell_source: str, notebook_path: str) -> str:
        """Fix 'Open in Colab' links in a markdown cell."""
        if not self.notebook_path:
            self.notebook_path = notebook_path
            
        # Patterns for matching Colab links
        patterns = [
            # Markdown style links
            r'\[(?:Open|Run|View) (?:in|on) Colab\]\((https?://colab\.research\.google\.com/[^\)]+)\)',
            r'\[\!\[(?:Open|Run|View) (?:in|on) Colab\]\(https?://[^\)]+\)\]\((https?://colab\.research\.google\.com/[^\)]+)\)',
            # HTML style links with image
            r'<a href="(https?://colab\.research\.google\.com/[^"]+)"[^>]*>.*?</a>'
        ]
        
        colab_url = self.create_colab_url(notebook_path)
        if not colab_url:
            return cell_source
            
        modified_source = cell_source
        
        # Handle different link patterns
        for i, pattern in enumerate(patterns):
            matches = re.finditer(pattern, cell_source)
            for match in matches:
                full_match = match.group(0)
                
                # Extract the URL based on pattern type
                if i == 0:  # Simple markdown link
                    old_link = match.group(1)
                    replacement = full_match.replace(old_link, colab_url)
                elif i == 1:  # Markdown link with image
                    old_link = match.group(2)
                    replacement = full_match.replace(old_link, colab_url)
                elif i == 2:  # HTML link
                    old_link = match.group(1)
                    # For HTML links, we need to construct a proper HTML replacement
                    replacement = f'<a href="{colab_url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
                
                modified_source = modified_source.replace(full_match, replacement)
                
        return modified_source
    
    def process_cell(self, cell: NotebookNode, llm_service: Optional = None) -> NotebookNode:
        """Process a single cell."""
        if cell.cell_type != 'markdown':
            return cell
            
        # Improve detection of colab links with broader patterns
        if not any(pattern in cell.source.lower() for pattern in ['colab', 'href', '<a ']):
            return cell
            
        log_msg(f"\nChecking cell for Colab links:", Fore.CYAN, '🔍')
        try:
            new_text = self.fix_colab_links(cell.source, self.notebook_path)
            if new_text != cell.source:
                log_msg("Original:", Fore.RED, '📄')
                log_msg(cell.source, Fore.RED)
                log_msg("Fixed:", Fore.GREEN, '✨')
                log_msg(new_text, Fore.GREEN)
                log_msg("-" * 80)
                cell.source = new_text
            else:
                log_msg("No Colab links to fix in this cell", Fore.YELLOW)
        except Exception as e:
            log_msg(f"Error fixing Colab links: {e}", Fore.RED, '❌')
        return cell
    
    def process_notebook(self, infile: str, outfile: str, 
                         llm_service: Optional = None, nb: NotebookNode = None) -> None:
        """Process an entire notebook."""
        self.notebook_path = os.path.abspath(infile)
        super().process_notebook(infile, outfile, llm_service, nb)
