"""
memory_backend.py — Backward-compat re-export of ctrl_memory.backend.

Previous versions of ctrl-memory had the backend at the project root.
This stub preserves imports for anyone referencing `memory_backend` directly.
New code should import from `ctrl_memory.backend`.
"""
from ctrl_memory.backend import (
    DEFAULT_STORAGE_DIR,
    EmbeddingEngine,
    FactNotFoundError,
    MemoryStore,
    SQLiteStore,
    create_store,
)

__all__ = [
    "DEFAULT_STORAGE_DIR",
    "EmbeddingEngine",
    "FactNotFoundError",
    "MemoryStore",
    "SQLiteStore",
    "create_store",
]
