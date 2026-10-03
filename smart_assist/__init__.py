"""Smart Assist knowledge graph application."""

from .code_graph import CodeGraphIngestor
from .pipeline import SmartAssistPipeline

__all__ = ["SmartAssistPipeline", "CodeGraphIngestor"]
