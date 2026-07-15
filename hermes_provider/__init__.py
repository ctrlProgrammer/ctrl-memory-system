"""
ctrl-memory — Hermes Memory Provider Plugin.

Wraps the SQLiteStore backend as a full Hermes MemoryProvider with:
  - Automatic prefetch: relevant facts injected before each turn.
  - Auto-capture: user statements that look like facts are stored.
  - System prompt block: agent knows it has memory.
  - Tool schemas: the agent can explicitly add/search/delete facts.

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

# Attempt to import our backend from the ctrl-memory project.
# Falls back to a bundled inline copy if import fails (plugin runs standalone).
try:
    from memory_backend import SQLiteStore, EmbeddingEngine, FactNotFoundError
except ImportError:
    # ── Inline minimal backend for standalone plugin deployment ──────
    import sqlite3
    import math

    class FactNotFoundError(Exception):
        pass

    class EmbeddingEngine:
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self._model = None
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
            except ImportError:
                pass

        @property
        def is_available(self):
            return self._model is not None

        def embed(self, text: str) -> List[float]:
            if not self._model:
                raise RuntimeError("sentence-transformers not installed")
            return self._model.encode(text, normalize_embeddings=True).tolist()

    class SQLiteStore:
        def __init__(self, db_path, embedding_engine=None):
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user_id ON facts(user_id)")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    fact_id INTEGER PRIMARY KEY,
                    vector BLOB NOT NULL,
                    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE
                )
            """)
            self._conn.commit()
            self._ee = embedding_engine

        def add_fact(self, user_id, content, tags=""):
            now = time.time()
            cur = self._conn.execute(
                "INSERT INTO facts (user_id, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, content, tags, now, now),
            )
            fact_id = cur.lastrowid
            self._maybe_embed(fact_id, content)
            self._conn.commit()
            return {"id": fact_id, "content": content, "tags": tags}

        def get_all_facts(self, user_id):
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)
            ).fetchall()
            return [dict(r) for r in rows]

        def count_facts(self, user_id):
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM facts WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row["cnt"]

        def search_facts(self, user_id, query, limit=5):
            q = query.lower().strip()
            if not q:
                return []
            like = f"%{q}%"
            rows = self._conn.execute(
                """SELECT * FROM facts
                   WHERE user_id = ? AND (LOWER(content) LIKE ? OR LOWER(tags) LIKE ?)
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, like, like, limit),
            ).fetchall()
            return [dict(r) for r in rows]

        def search_facts_semantic(self, user_id, query, limit=5):
            if not self._ee or not self._ee.is_available:
                return self.search_facts(user_id, query, limit)
            q = query.strip()
            if not q:
                return []
            query_vec = self._ee.embed(q)
            rows = self._conn.execute(
                """SELECT e.fact_id, e.vector, f.content, f.tags
                   FROM embeddings e JOIN facts f ON e.fact_id = f.id
                   WHERE f.user_id = ?""",
                (user_id,),
            ).fetchall()
            import struct
            scored = []
            for row in rows:
                blob = row["vector"]
                count = len(blob) // 4
                fact_vec = list(struct.unpack(f"<{count}f", blob))
                dot = sum(x * y for x, y in zip(query_vec, fact_vec))
                nq = math.sqrt(sum(x * x for x in query_vec))
                nf = math.sqrt(sum(x * x for x in fact_vec))
                score = dot / (nq * nf) if nq and nf else 0.0
                scored.append({
                    "id": row["fact_id"], "content": row["content"],
                    "tags": row["tags"], "score": round(score, 4),
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

        def delete_fact(self, user_id, fact_id):
            cur = self._conn.execute(
                "DELETE FROM facts WHERE user_id = ? AND id = ?", (user_id, fact_id)
            )
            self._conn.commit()
            return cur.rowcount > 0

        def clear_all(self, user_id):
            self._conn.execute(
                "DELETE FROM embeddings WHERE fact_id IN (SELECT id FROM facts WHERE user_id = ?)",
                (user_id,),
            )
            self._conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
            self._conn.commit()

        def _maybe_embed(self, fact_id, content):
            if not self._ee or not self._ee.is_available:
                return
            vec = self._ee.embed(content)
            import struct
            blob = struct.pack(f"<{len(vec)}f", *vec)
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (fact_id, vector) VALUES (?, ?)",
                (fact_id, blob),
            )

        def close(self):
            self._conn.close()
    # ── End of inline backend ────────────────────────────────────────


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
      - Tools: add, search (keyword + semantic), delete, status.
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
        hermes_home = Path(kwargs.get("hermes_home", "~/.hermes"))
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
        """
        if not self._store or not user_content:
            return

        markers = [" i ", " my ", " we ", " our ", " i'm ", " i've ", " i use "]
        text = user_content.lower()
        if any(m in text for m in markers) or any(
            text.startswith(m.strip()) for m in markers
        ):
            content = user_content.strip()[:500]
            self._store.add_fact(
                self._user_id,
                content=content,
                tags="auto-captured",
            )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas the agent can call."""
        return [ADD_MEMORY_SCHEMA, SEARCH_MEMORY_SCHEMA, DELETE_MEMORY_SCHEMA, STATUS_SCHEMA]

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
