"""Prompt template management service"""
import os
from pathlib import Path
from jinja2 import Template
from typing import Dict, Any


class PromptService:
    """Service for loading and rendering prompt templates"""

    def __init__(self, templates_dir: str = "prompts"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, Template] = {}

    def get_prompt(self, template_id: str, **kwargs) -> str:
        """Get a formatted prompt template

        Args:
            template_id: Template identifier (e.g., 'property_search.intent_analysis')
            **kwargs: Variables to render in the template

        Returns:
            Rendered prompt string
        """
        # Load template if not cached
        if template_id not in self.templates:
            self._load_template(template_id)

        # Render template with kwargs
        return self.templates[template_id].render(**kwargs)

    def _load_template(self, template_id: str):
        """Load a template from file system"""
        # Convert template_id to file path (e.g., 'property_search.intent_analysis' -> 'property_search/intent_analysis.txt')
        template_path = self.templates_dir / f"{template_id.replace('.', '/')}.txt"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, "r") as f:
            template_content = f.read()

        self.templates[template_id] = Template(template_content)

    def add_template(self, template_id: str, template_content: str) -> None:
        """Add a new template programmatically

        Args:
            template_id: Template identifier
            template_content: Template content string
        """
        self.templates[template_id] = Template(template_content)

        # Optionally save to file system
        template_path = self.templates_dir / f"{template_id.replace('.', '/')}.txt"
        template_path.parent.mkdir(parents=True, exist_ok=True)

        with open(template_path, "w") as f:
            f.write(template_content)

    def reload_templates(self):
        """Reload all templates from disk (useful for development)"""
        self.templates.clear()


# Global prompt service instance
prompt_service = PromptService()
