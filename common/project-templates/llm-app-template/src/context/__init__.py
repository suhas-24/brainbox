"""
Advanced Context Engineering Framework for AI Forge

Context engineering is the art and science of crafting the right context for LLMs
to produce accurate, relevant, and contextually appropriate responses.

Key Components:
- Context Assembly: Building comprehensive context from multiple sources
- Context Compression: Optimizing context for token limits
- Context Retrieval: Finding relevant information from memory/knowledge bases
- Context Adaptation: Adapting context based on user, task, and environment
- Context Validation: Ensuring context quality and relevance
"""

from .context_manager import ContextManager
from .context_assembler import ContextAssembler
from .context_compressor import ContextCompressor
from .context_retriever import ContextRetriever
from .context_validator import ContextValidator
from .adaptive_context import AdaptiveContextEngine

__all__ = [
    "ContextManager",
    "ContextAssembler",
    "ContextCompressor", 
    "ContextRetriever",
    "ContextValidator",
    "AdaptiveContextEngine"
]
