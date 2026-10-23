"""Independent ML knowledge module; importing it does not load the corpus."""
from .core import canonical
from .runtime import ReferenceArgs, ReferenceStore
from .memory import KnowledgeMemory
from .context import SelectionContext, select, prepare_request

__all__ = ["canonical", "ReferenceArgs", "ReferenceStore", "KnowledgeMemory", "SelectionContext", "select", "prepare_request"]
