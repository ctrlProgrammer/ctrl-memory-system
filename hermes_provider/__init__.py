"""
ctrl-memory — Hermes Memory Provider Plugin.

Wraps the SQLiteStore backend as a full Hermes MemoryProvider with:
  - Automatic prefetch: relevant facts injected before each turn.
  - Auto-capture: user statements that look like facts are stored.
  - System prompt block: agent knows it has memory.
  - Tool schemas: the agent can explicitly add/search/update/delete facts.

Shares the same database file as the MCP server when both use the
default path (~/.ctrl-memory/memory.db), so facts are consistent
across the Hermes plugin and any MCP-connected agent.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from ctrl_memory.backend import SQLiteStore, EmbeddingEngine, FactNotFoundError, _validate_fact_input

log = logging.getLogger(__name__)

# ── Tool Schemas ─────────────────────────────────────────────────────────

ADD_MEMORY_SCHEMA = {
    "name": "ctrl_memory_add",
    "description": "Store a fact about the user, project, or preferences for cross-session recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember"},
            "tags": {"type": "string", "description": "Optional comma-separated labels"},
        },
        "required": ["content"],
    },
}

SEARCH_MEMORY_SCHEMA = {
    "name": "ctrl_memory_search",
    "description": "Search stored facts by keyword or natural-language query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword or natural-language query"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        },
        "required": ["query"],
    },
}

UPDATE_MEMORY_SCHEMA = {
    "name": "ctrl_memory_update",
    "description": "Update an existing fact's content and/or tags.",
    "parameters": {
        "type": "object",
        "properties": {
            "fact_id": {"type": "integer", "description": "The fact's numeric ID"},
            "content": {"type": "string", "description": "New content text"},
            "tags": {"type": "string", "description": "New comma-separated tags"},
        },
        "required": ["fact_id"],
    },
}

DELETE_MEMORY_SCHEMA = {
    "name": "ctrl_memory_delete",
    "description": "Delete a specific fact by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "fact_id": {"type": "integer", "description": "The fact's numeric ID"},
        },
        "required": ["fact_id"],
    },
}

STATUS_SCHEMA = {
    "name": "ctrl_memory_status",
    "description": "Show how many facts are stored and whether semantic search is available.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ── MemoryProvider Implementation ─────────────────────────────────────────

class CtrlMemoryProvider(MemoryProvider):
    """
    Hermes memory provider backed by SQLite with optional semantic search.

    Features:
      - Automatic prefetch: relevant facts injected before each LLM call.
      - Auto-capture: user statements with 'I', 'my', 'we' are saved.
      - Tools: add, search (keyword + semantic), update, delete, status.
      - Shares the database with the ctrl-memory MCP server by default.
    """

    def __init__(self):
        self._store: Optional[SQLiteStore] = None
        self._user_id: str = "default"
        self._session_id: str = ""

    @property
    def name(self) -> str:
        return "ctrl-memory"

    # ── Lifecycle ─────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Always available — local SQLite, no network needed."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """
        Set up the SQLite store inside the Hermes home directory.

        Uses kwargs['hermes_home'] for profile isolation. Falls back to
        ~/.hermes if not provided.
        """
        hermes_home = Path(kwargs.get("hermes_home", "~/.hermes")).expanduser()
        db_path = hermes_home / "ctrl-memory" / "memory.db"

        # Optionally try to load the embedding engine.
        ee = None
        try:
            ee = EmbeddingEngine()
            if not ee.is_available:
                ee = None
        except Exception:
            ee = None

        self._store = SQLiteStore(db_path=str(db_path), embedding_engine=ee)
        self._session_id = session_id
        self._user_id = kwargs.get("user_id", "default")
        log.info(
            "CtrlMemory initialized for user '%s' at %s (semantic=%s)",
            self._user_id, db_path, ee is not None,
        )

    def system_prompt_block(self) -> str:
        """Tell the agent how many facts it has stored."""
        if not self._store:
            return ""
        count = self._store.count_facts(self._user_id)
        semantic = (
            "semantic search available"
            if (hasattr(self._store, '_ee') and self._store._ee and self._store._ee.is_available)
            else "keyword search"
        )
        return (
            f"# Ctrl-Memory\n"
            f"{count} facts stored ({semantic}).\n"
            f"Use ctrl_memory_add to save facts, ctrl_memory_search to recall them."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """
        Automatically recall relevant facts before each turn.

        Uses the current conversation as a query to find related facts.
        Falls back to returning recent facts if query is empty.
        """
        if not self._store:
            return ""

        query = (query or "").strip()
        if not query:
            return ""

        # Try semantic first, fall back to keyword.
        results = self._store.search_facts_semantic(self._user_id, query, limit=5)
        if not results:
            results = self._store.search_facts(self._user_id, query, limit=5)

        if not results:
            return ""

        lines = []
        for r in results:
            score = r.get("score")
            score_str = f" [score={score}]" if score else ""
            lines.append(f"- {r['content']}{score_str}")

        return "## Relevant Memories\n" + "\n".join(lines)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list = None,
    ) -> None:
        """
        Automatically capture user statements that look like facts worth
        remembering.

        Heuristic: sentences containing first-person markers (I, my, we, our).
        A more sophisticated approach would use LLM extraction, but this is
        zero-cost and catches the common case.

        Deduplication: if the same content text (or a close prefix match)
        was already auto-captured in the last 60 seconds, skip storage.
        """
        if not self._store or not user_content:
            return

        markers = [" i ", " my ", " we ", " our ", " i'm ", " i've ", " i use "]
        text = user_content.lower()
        if not (any(m in text for m in markers) or any(
            text.startswith(m.strip()) for m in markers
        )):
            return

        content = user_content.strip()[:500]

        # Deduplication: check if a similar fact was already stored recently.
        now = time.time()
        recent = self._store.search_facts(self._user_id, content, limit=3)
        for fact in recent:
            if fact.get("tags") == "auto-captured" and (
                fact["content"] == content
                or content.startswith(fact["content"])
                or fact["content"].startswith(content)
            ):
                if now - fact.get("created_at", 0) < 60:
                    return  # Already captured within the last minute

        self._store.add_fact(
            self._user_id,
            content=content,
            tags="auto-captured",
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas the agent can call."""
        return [ADD_MEMORY_SCHEMA, SEARCH_MEMORY_SCHEMA, UPDATE_MEMORY_SCHEMA,
                DELETE_MEMORY_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call to the store backend."""
        if not self._store:
            return json.dumps({"error": "Provider not initialized"})

        tool_name = tool_name.lower()

        if tool_name == "ctrl_memory_add":
            record = self._store.add_fact(
                self._user_id,
                content=args["content"],
                tags=args.get("tags", ""),
            )
            return json.dumps({"fact_id": record["id"], "status": "stored"})

        if tool_name == "ctrl_memory_search":
            # Try semantic first, fall back to keyword.
            query = args["query"]
            limit = args.get("limit", 5)
            results = self._store.search_facts_semantic(self._user_id, query, limit=limit)
            if not results:
                results = self._store.search_facts(self._user_id, query, limit=limit)
            return json.dumps({"results": results, "count": len(results)})

        if tool_name == "ctrl_memory_update":
            fact_id = int(args["fact_id"])
            content = args.get("content")
            tags = args.get("tags")
            try:
                record = self._store.update_fact(
                    self._user_id, fact_id,
                    content=content, tags=tags,
                )
                return json.dumps({"fact_id": record["id"], "status": "updated"})
            except FactNotFoundError as e:
                return json.dumps({"error": str(e)})

        if tool_name == "ctrl_memory_delete":
            ok = self._store.delete_fact(
                self._user_id,
                fact_id=int(args["fact_id"]),
            )
            return json.dumps({"deleted": ok})

        if tool_name == "ctrl_memory_status":
            count = self._store.count_facts(self._user_id)
            semantic = (
                hasattr(self._store, '_ee')
                and self._store._ee is not None
                and self._store._ee.is_available
            )
            return json.dumps({
                "fact_count": count,
                "semantic_search": semantic,
                "backend": "sqlite",
            })

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        """Close the database connection."""
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None

    def on_session_end(self, messages: list) -> None:
        """Log a summary of stored facts on session end."""
        if self._store:
            count = self._store.count_facts(self._user_id)
            log.info(
                "Session '%s' ended. %d facts stored for user '%s'.",
                self._session_id, count, self._user_id,
            )


# ── Plugin Entry Point ────────────────────────────────────────────────────

def register(ctx) -> None:
    """Register this provider with Hermes."""
    ctx.register_memory_provider(CtrlMemoryProvider())
