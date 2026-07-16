"""
memory_backend.py — Storage backends for LLM agent memory.

Provides two pluggable backends with the same public API:
  - MemoryStore:  JSON files (one per user) — zero dependencies, human-readable.
  - SQLiteStore:  Single SQLite DB with LIKE + optional semantic search.

When using SQLiteStore, semantic search activates automatically if
sentence-transformers is installed. Zero config needed.

KISS principle:
  - Same method signatures across backends — swap with one line.
  - Substring search always works; semantic search auto-detects.
  - Embeddings are computed locally, no cloud API calls.
"""

import json
import logging
import math
import sqlite3
import struct
import time
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


# ── Input validation ──────────────────────────────────────────────────────

def _validate_fact_input(content: str, tags: str) -> None:
    """Validate content and tags; raise ValueError on violation."""
    if not content or not content.strip():
        raise ValueError("content must be a non-empty string")
    if len(content) > 100_000:
        raise ValueError(f"content too long ({len(content)} chars, max 100_000)")
    if len(tags) > 10_000:
        raise ValueError(f"tags too long ({len(tags)} chars, max 10_000)")


# ── Constants ────────────────────────────────────────────────────────────

#: Default directory where storage files live.
DEFAULT_STORAGE_DIR = Path.home() / ".ctrl-memory"


# ── Exceptions ───────────────────────────────────────────────────────────

class FactNotFoundError(Exception):
    """Raised when trying to get/update/delete a fact that doesn't exist."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Engine (optional — requires sentence-transformers)
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Lightweight wrapper around sentence-transformers for local vector search.

    Fully optional: if sentence-transformers isn't installed, is_available
    returns False and all methods are no-ops. No forced dependencies.

    Usage:
        ee = EmbeddingEngine()
        if ee.is_available:
            vec = ee.embed("Your text here")
            score = ee.cosine_similarity(vec_a, vec_b)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Load the embedding model on init.

        Args:
            model_name: A sentence-transformers model name.
                        Defaults to 'all-MiniLM-L6-v2' (fast, ~80MB, 384-d vectors).
        """
        self._model = None
        self._model_name = model_name
        self._dimension = 384  # fallback if model fails to load
        self._load_model()
        # If the model loaded, read the actual dimension from it.
        if self._model is not None:
            try:
                self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception:
                pass  # keep the fallback

    def _load_model(self) -> None:
        """Try loading the model; log on failure."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            self._model = None
        except Exception as exc:
            log.warning("Failed to load embedding model '%s': %s", self._model_name, exc)
            self._model = None

    @property
    def is_available(self) -> bool:
        """True if the embedding model loaded successfully."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Output dimension of the embedding vectors."""
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """
        Convert a single text string to a vector embedding.

        Args:
            text: Text to embed.

        Returns:
            List of floats (the embedding vector).

        Raises:
            RuntimeError: If the model isn't loaded.
        """
        if not self._model:
            raise RuntimeError(
                "Embedding model not available. Install sentence-transformers."
            )
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings in one batch (faster).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self._model:
            raise RuntimeError(
                "Embedding model not available. Install sentence-transformers."
            )
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Cosine similarity between two vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Similarity score between 0 and 1.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ── Embedding serialisation helpers ──────────────────────────────────────

def _pack_embedding(vec: List[float]) -> bytes:
    """Pack a list of floats into a compact binary blob (little-endian)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> List[float]:
    """Unpack a binary blob back into a list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _levenshtein(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Pure Python implementation — zero dependencies. O(n*m) time,
    O(min(n,m)) memory using two-row optimization.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(
                prev[j + 1] + 1,       # deletion
                curr[j] + 1,            # insertion
                prev[j] + cost,         # substitution
            ))
        prev = curr
    return prev[-1]


def _damerau_levenshtein(s1: str, s2: str) -> int:
    """
    Compute Damerau-Levenshtein distance (adds transposition).

    Like Levenshtein but also counts swapping two adjacent characters
    as a single edit. Needed for typos like 'typsecript' → 'typescript'.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    # Use full matrix for transposition checks.
    d = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        d[i][0] = i
    for j in range(len(s2) + 1):
        d[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,           # deletion
                d[i][j-1] + 1,           # insertion
                d[i-1][j-1] + cost,      # substitution
            )
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)  # transposition
    return d[len(s1)][len(s2)]


def _fuzzy_token_match(token: str, text: str, max_dist: int = 1) -> bool:
    """
    Check if a token (or a close variant within edit distance) appears in text.

    Uses three strategies:
      1. Exact substring match (fast path).
      2. Word-level Damerau-Levenshtein for typos + transpositions.
      3. Compound-word splitting: splits hyphenated/underscored words
         (e.g. 'error-handling') and checks each part.

    Args:
        token: Query token to search for (lowered).
        text:  Content text to search in (lowered).
        max_dist: Maximum edit distance allowed (default 1).

    Returns:
        True if the token (or a close variant) appears in text.
    """
    if token in text:
        return True

    # Check each word in text.
    for word in text.split():
        # Fast path: token is substring of word (compound words).
        if token in word:
            return True

        # Split compound words by common separators.
        for part in word.replace("-", " ").replace("_", " ").replace("/", " ").split():
            if part and len(part) >= len(token) - max_dist:
                if _damerau_levenshtein(token, part) <= max_dist:
                    return True

        # Length guard before full Damerau-Levenshtein.
        if abs(len(word) - len(token)) > max_dist * 2:
            continue
        if _damerau_levenshtein(token, word) <= max_dist:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_store(backend: str = "json", **kwargs):
    """
    Create a store instance by backend name.

    Args:
        backend: "json" (default) or "sqlite".
        **kwargs: Passed to the store constructor (e.g. storage_dir, db_path,
                  embedding_engine for SQLiteStore).

    Returns:
        MemoryStore or SQLiteStore instance.

    Usage:
        store = create_store("json")
        store = create_store("sqlite", db_path="/tmp/mem.db")
        store = create_store("sqlite", embedding_engine=EmbeddingEngine())
    """
    if backend == "sqlite":
        return SQLiteStore(**kwargs)
    return MemoryStore(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# JSON Backend
# ═══════════════════════════════════════════════════════════════════════════

class MemoryStore:
    """
    A JSON-file-based fact store. One JSON file per user.

    **Concurrency note:** This backend is NOT safe for concurrent access from
    multiple processes. It reads, mutates, and rewrites the entire file per
    operation with no file locking. For concurrent-safe storage, use
    SQLiteStore instead.

    Each fact is a dict with:
      - id (int):         Auto-incrementing identifier, unique per user.
      - content (str):    The fact text to remember.
      - tags (str):       Optional comma-separated labels for filtering.
      - created_at (float): Unix timestamp when the fact was created.
      - updated_at (float): Unix timestamp of last modification.

    Usage:
        store = MemoryStore()
        store.add_fact("alice", "User prefers dark mode", tags="preference")
        results = store.search_facts("alice", "dark mode")
    """

    def __init__(self, storage_dir: Optional[str | Path] = None) -> None:
        """
        Initialize the store.

        Args:
            storage_dir: Directory for JSON files. Defaults to ~/.ctrl-memory.
        """
        self._dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _validate_input(content: str, tags: str) -> None:
        _validate_fact_input(content, tags)

    def _path_for(self, user_id: str) -> Path:
        """
        Get the file path for a given user's facts.

        Args:
            user_id: Unique user identifier.

        Returns:
            Path to the user's JSON file.
        """
        return self._dir / f"{user_id}.json"

    def _load(self, user_id: str) -> list:
        """
        Load all facts for a user from disk.

        Args:
            user_id: Unique user identifier.

        Returns:
            List of fact dicts, or empty list if file doesn't exist.
        """
        path = self._path_for(user_id)
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def _save(self, user_id: str, facts: list) -> None:
        """
        Write a list of facts to disk for a user.

        Args:
            user_id: Unique user identifier.
            facts:   List of fact dicts to persist.
        """
        self._path_for(user_id).write_text(
            json.dumps(facts, indent=2, ensure_ascii=False)
        )

    def _next_id(self, facts: list) -> int:
        """
        Compute the next auto-increment ID from an existing fact list.

        Args:
            facts: Current list of fact dicts.

        Returns:
            The next available integer ID.
        """
        if not facts:
            return 1
        return max(f["id"] for f in facts) + 1

    # ── CRUD: Create ─────────────────────────────────────────────────────

    def add_fact(
        self,
        user_id: str,
        content: str,
        tags: str = "",
    ) -> dict:
        """
        Store a new fact for a user.

        Args:
            user_id: Unique user identifier.
            content: The fact text to remember.
            tags:    Optional comma-separated labels (e.g. "preference,project").

        Returns:
            The newly created fact record dict.
        """
        self._validate_input(content, tags)
        facts = self._load(user_id)
        now = time.time()
        record = {
            "id": self._next_id(facts),
            "content": content,
            "tags": tags,
            "created_at": now,
            "updated_at": now,
        }
        facts.append(record)
        self._save(user_id, facts)
        return record

    # ── CRUD: Read ───────────────────────────────────────────────────────

    def get_fact(self, user_id: str, fact_id: int) -> dict:
        """
        Retrieve a single fact by its ID.

        Args:
            user_id: Unique user identifier.
            fact_id: The fact's numeric ID.

        Returns:
            The fact dict.

        Raises:
            FactNotFoundError: If no fact with that ID exists.
        """
        facts = self._load(user_id)
        for f in facts:
            if f["id"] == fact_id:
                return f
        raise FactNotFoundError(
            f"Fact with id={fact_id} not found for user '{user_id}'."
        )

    def get_all_facts(self, user_id: str) -> List[dict]:
        """
        Retrieve every fact stored for a user.

        Args:
            user_id: Unique user identifier.

        Returns:
            List of all fact dicts (empty list if none).
        """
        return self._load(user_id)

    def count_facts(self, user_id: str) -> int:
        """
        Count how many facts a user has stored.

        Args:
            user_id: Unique user identifier.

        Returns:
            Number of facts.
        """
        return len(self._load(user_id))

    def search_facts(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        Search a user's facts by keyword (case-insensitive multi-token match).

        Splits the query into meaningful tokens (≥2 chars, skipping stop words)
        and matches facts whose content or tags contain ANY of them.
        This provides good recall for alias resolution while keeping noise low.

        Args:
            user_id: Unique user identifier.
            query:   Keyword to search for.
            limit:   Maximum number of results to return (default 10).

        Returns:
            List of matching fact dicts, newest first.
        """
        # Stop words that provide no search signal.
        STOP_WORDS = frozenset({
            "what's", "whats", "why's", "whys", "how's", "hows",
            "when's", "whens", "where's", "wheres",
            "what", "why", "how", "when", "where", "who",
            "the", "a", "an", "is", "are", "was", "were",
            "do", "does", "did", "done", "doing",
            "i", "my", "me", "mine", "we", "our", "us",
            "you", "your", "they", "them", "their",
            "it", "its", "this", "that", "these", "those",
            "in", "on", "at", "to", "for", "of", "with",
            "and", "or", "but", "not", "no", "nor",
            "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "has", "have", "had",
            "about", "into", "over", "after", "before",
            "up", "down", "out", "off", "just", "also",
            "very", "too", "so", "than", "as", "if",
        })

        tokens = [
            t for t in query.lower().strip().split()
            if len(t) >= 2 and t not in STOP_WORDS
        ]
        if not tokens:
            return []

        facts = self._load(user_id)

        def _match_count(f: dict) -> int:
            """Number of query tokens found in content or tags."""
            text = f"{f['content'].lower()} {f['tags'].lower()}"
            return sum(1 for t in tokens if t in text)

        scored = [(f, _match_count(f)) for f in facts]
        scored = [(f, n) for f, n in scored if n > 0]
        if not scored:
            return []

        # Sort by match count (desc), then recency (desc).
        scored.sort(key=lambda x: (-x[1], -x[0]["updated_at"], -x[0]["created_at"]))
        return [f for f, _ in scored[:limit]]

    # ── CRUD: Update ─────────────────────────────────────────────────────

    def update_fact(
        self,
        user_id: str,
        fact_id: int,
        content: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> dict:
        """
        Update an existing fact's content and/or tags.

        Args:
            user_id:  Unique user identifier.
            fact_id:  The fact's numeric ID.
            content:  New content text (None = leave unchanged).
            tags:     New tags string (None = leave unchanged).

        Returns:
            The updated fact dict.

        Raises:
            FactNotFoundError: If no fact with that ID exists.
        """
        facts = self._load(user_id)
        for f in facts:
            if f["id"] == fact_id:
                if content is not None:
                    f["content"] = content
                if tags is not None:
                    f["tags"] = tags
                f["updated_at"] = time.time()
                self._save(user_id, facts)
                return f
        raise FactNotFoundError(
            f"Fact with id={fact_id} not found for user '{user_id}'."
        )

    # ── CRUD: Delete ─────────────────────────────────────────────────────

    def delete_fact(self, user_id: str, fact_id: int) -> bool:
        """
        Delete a single fact by its ID.

        Args:
            user_id: Unique user identifier.
            fact_id: The fact's numeric ID.

        Returns:
            True if a fact was deleted, False if not found.
        """
        facts = self._load(user_id)
        before = len(facts)
        facts = [f for f in facts if f["id"] != fact_id]
        if len(facts) < before:
            self._save(user_id, facts)
            return True
        return False

    def clear_all(self, user_id: str) -> None:
        """
        Delete ALL facts for a user. Removes the JSON file entirely.

        Args:
            user_id: Unique user identifier.
        """
        path = self._path_for(user_id)
        if path.exists():
            path.unlink()


# ═══════════════════════════════════════════════════════════════════════════
# SQLite Backend
# ═══════════════════════════════════════════════════════════════════════════

class SQLiteStore:
    """
    A SQLite-backed fact store. Single database file for all users.

    Each fact is stored as a row with:
      - id (int):         Auto-incrementing primary key.
      - user_id (str):    Who the fact belongs to.
      - content (str):    The fact text to remember.
      - tags (str):       Optional comma-separated labels.
      - created_at (float): Unix timestamp when the fact was created.
      - updated_at (float): Unix timestamp of last modification.

    Optionally supports semantic (vector) search via an EmbeddingEngine.
    When an engine is provided, facts are automatically embedded on insert
    and update, and search_facts_semantic() becomes available.

    Advantages over MemoryStore (JSON):
      - Concurrent-safe (WAL mode).
      - Faster search via SQL LIKE.
      - Single file to back up.
      - Optional vector search for semantic similarity.

    Usage:
        store = SQLiteStore(db_path="/tmp/memory.db")
        store.add_fact("alice", "User prefers dark mode", tags="preference")
        results = store.search_facts("alice", "dark mode")
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        """
        Initialize the store, creating tables if they don't exist.

        Args:
            db_path:         Path to the SQLite database file.
                             Defaults to ~/.ctrl-memory/memory.db.
            embedding_engine: Optional EmbeddingEngine for semantic search.
                              If provided, facts are auto-embedded.
        """
        self._db_path = Path(db_path or DEFAULT_STORAGE_DIR / "memory.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ee = embedding_engine
        self._init_tables()

    def __del__(self) -> None:
        """Ensure the database connection is closed on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def _init_tables(self) -> None:
        """
        Create the facts and embeddings tables if they don't exist.
        Safe to call repeatedly — uses IF NOT EXISTS.
        """
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                tags       TEXT    DEFAULT '',
                created_at REAL    NOT NULL,
                updated_at REAL    NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_user_id
            ON facts(user_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                fact_id   INTEGER PRIMARY KEY,
                vector    BLOB NOT NULL,
                FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE
            )
        """)
        self._conn.commit()

    # ── CRUD: Create ─────────────────────────────────────────────────────

    def add_fact(
        self,
        user_id: str,
        content: str,
        tags: str = "",
    ) -> dict:
        """
        Store a new fact for a user.

        If an EmbeddingEngine is configured, the content is automatically
        vectorised and stored for semantic search.

        Args:
            user_id: Unique user identifier.
            content: The fact text to remember.
            tags:    Optional comma-separated labels.

        Returns:
            The newly created fact record dict.
        """
        _validate_fact_input(content, tags)
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO facts (user_id, content, tags, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, content, tags, now, now),
        )
        fact_id = cur.lastrowid
        self._maybe_embed(fact_id, content)
        self._conn.commit()
        return {
            "id": fact_id,
            "content": content,
            "tags": tags,
            "created_at": now,
            "updated_at": now,
        }

    # ── Embedding helpers ─────────────────────────────────────────────────

    def _maybe_embed(self, fact_id: int, content: str) -> None:
        """
        Compute and store embedding for a fact if the engine is available.

        This is a no-op when no EmbeddingEngine is configured.

        Args:
            fact_id: The fact's numeric ID.
            content: Text to embed.
        """
        if not self._ee or not self._ee.is_available:
            return
        vec = self._ee.embed(content)
        blob = _pack_embedding(vec)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (fact_id, vector) VALUES (?, ?)",
            (fact_id, blob),
        )

    def _delete_embedding(self, fact_id: int) -> None:
        """Remove the embedding for a fact (called on delete/update)."""
        self._conn.execute(
            "DELETE FROM embeddings WHERE fact_id = ?", (fact_id,)
        )

    # ── CRUD: Read ───────────────────────────────────────────────────────

    def get_fact(self, user_id: str, fact_id: int) -> dict:
        """
        Retrieve a single fact by its ID (scoped to user).

        Args:
            user_id: Unique user identifier.
            fact_id: The fact's numeric ID.

        Returns:
            The fact dict.

        Raises:
            FactNotFoundError: If no fact with that ID exists for the user.
        """
        row = self._conn.execute(
            "SELECT * FROM facts WHERE user_id = ? AND id = ?",
            (user_id, fact_id),
        ).fetchone()
        if row is None:
            raise FactNotFoundError(
                f"Fact with id={fact_id} not found for user '{user_id}'."
            )
        return dict(row)

    def get_all_facts(self, user_id: str) -> List[dict]:
        """
        Retrieve every fact stored for a user.

        Args:
            user_id: Unique user identifier.

        Returns:
            List of all fact dicts (empty list if none).
        """
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_facts(self, user_id: str) -> int:
        """
        Count how many facts a user has stored.

        Args:
            user_id: Unique user identifier.

        Returns:
            Number of facts.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM facts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["cnt"]

    def search_facts(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        Search a user's facts by keyword (case-insensitive multi-token match).

        Splits the query into meaningful tokens (≥2 chars, skipping stop words)
        and matches facts whose content or tags contain ANY of them.

        Args:
            user_id: Unique user identifier.
            query:   Keyword to search for.
            limit:   Maximum results (default 10).

        Returns:
            List of matching fact dicts, newest first.
        """
        # Stop words that provide no search signal.
        STOP_WORDS = frozenset({
            "what's", "whats", "why's", "whys", "how's", "hows",
            "when's", "whens", "where's", "wheres",
            "what", "why", "how", "when", "where", "who",
            "the", "a", "an", "is", "are", "was", "were",
            "do", "does", "did", "done", "doing",
            "i", "my", "me", "mine", "we", "our", "us",
            "you", "your", "they", "them", "their",
            "it", "its", "this", "that", "these", "those",
            "in", "on", "at", "to", "for", "of", "with",
            "and", "or", "but", "not", "no", "nor",
            "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "has", "have", "had",
            "about", "into", "over", "after", "before",
            "up", "down", "out", "off", "just", "also",
            "very", "too", "so", "than", "as", "if",
        })

        tokens = [
            t for t in query.lower().strip().split()
            if len(t) >= 2 and t not in STOP_WORDS
        ]
        if not tokens:
            return []

        # Build OR conditions — ANY meaningful token triggers a match.
        conditions = []
        params = [user_id]
        for token in tokens:
            like = f"%{token}%"
            conditions.append("(LOWER(content) LIKE ? OR LOWER(tags) LIKE ?)")
            params.extend([like, like])

        sql = f"""SELECT * FROM facts
                 WHERE user_id = ?
                   AND ({' OR '.join(conditions)})
                 ORDER BY updated_at DESC, created_at DESC
                 LIMIT ?"""
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _search_facts_fuzzy(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        Fallback search with Levenshtein-based typo tolerance.

        Called when keyword search returns no results. Splits the query
        into meaningful tokens and checks if any word in a fact's content
        or tags is within edit distance 1 of any query token.

        Args:
            user_id: Unique user identifier.
            query:   Query with possible typos.
            limit:   Maximum results (default 10).

        Returns:
            List of matching fact dicts, newest first.
        """
        STOP_WORDS = frozenset({
            "what's", "whats", "why's", "whys", "how's", "hows",
            "when's", "whens", "where's", "wheres",
            "what", "why", "how", "when", "where", "who",
            "the", "a", "an", "is", "are", "was", "were",
            "do", "does", "did", "done", "doing",
            "i", "my", "me", "mine", "we", "our", "us",
            "in", "on", "at", "to", "for", "of", "with",
            "and", "or", "but", "not", "no", "nor",
        })

        tokens = [
            t for t in query.lower().strip().split()
            if len(t) >= 3 and t not in STOP_WORDS
        ]
        if not tokens:
            return []

        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()

        results = []
        for row in rows:
            content = (row["content"] or "").lower()
            tags = (row["tags"] or "").lower()
            text = f"{content} {tags}"
            for token in tokens:
                if _fuzzy_token_match(token, text, max_dist=1):
                    results.append(dict(row))
                    break

        results.sort(key=lambda x: x["updated_at"], reverse=True)
        return results[:limit]

    def search_facts_semantic(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[dict]:
        """
        Hybrid search: keyword recall + semantic re-ranking.

        First retrieves candidates via keyword search (fast, indexed), then
        re-ranks them by cosine similarity against the query vector. This
        combines the recall of keyword search with the relevance ranking of
        semantic search, avoiding the noise of a pure vector scan.

        When no EmbeddingEngine is configured, falls back to keyword-only
        results (sorted by recency).

        Args:
            user_id:  Unique user identifier.
            query:    Natural-language query.
            limit:    Maximum results (default 10).
            min_score: Minimum cosine similarity score (default 0.0).
                       Raise to filter out weakly-related results.

        Returns:
            List of fact dicts with an added 'score' key (cosine similarity),
            sorted by highest score first.
        """
        q = query.strip()
        if not q:
            return []

        # Step 1: Get keyword candidates (fast, broad recall).
        candidates = self.search_facts(user_id, q, limit=limit * 3)

        if not candidates:
            # Step 1b: Fuzzy fallback — Levenshtein distance for typos.
            candidates = self._search_facts_fuzzy(user_id, q, limit=limit * 3)

        if not candidates:
            return []

        # Step 2: If embedding engine available, re-rank by cosine similarity.
        if self._ee and self._ee.is_available:
            query_vec = self._ee.embed(q)
            for c in candidates:
                # Load embedding for this candidate.
                row = self._conn.execute(
                    "SELECT vector FROM embeddings WHERE fact_id = ?",
                    (c["id"],),
                ).fetchone()
                if row:
                    fact_vec = _unpack_embedding(row["vector"])
                    score = EmbeddingEngine.cosine_similarity(query_vec, fact_vec)
                    c["score"] = round(score, 4)
                else:
                    c["score"] = 0.0

            # Filter by minimum score, then sort by score descending.
            candidates = [c for c in candidates if c["score"] >= min_score]
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            # No embedding engine — candidates are already keyword-matched,
            # just mark them with a default score.
            for c in candidates:
                c["score"] = 0.0

        return candidates[:limit]

    # ── CRUD: Update ─────────────────────────────────────────────────────

    def update_fact(
        self,
        user_id: str,
        fact_id: int,
        content: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> dict:
        """
        Update an existing fact's content and/or tags.

        If content changes and an EmbeddingEngine is configured, the
        embedding is automatically recomputed.

        Args:
            user_id:  Unique user identifier.
            fact_id:  The fact's numeric ID.
            content:  New content text (None = leave unchanged).
            tags:     New tags string (None = leave unchanged).

        Returns:
            The updated fact dict.

        Raises:
            FactNotFoundError: If no fact with that ID exists for the user.
        """
        # Check existence first.
        existing = self.get_fact(user_id, fact_id)

        updates = []
        params = []
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if tags is not None:
            updates.append("tags = ?")
            params.append(tags)

        if not updates:
            return existing

        updates.append("updated_at = ?")
        params.append(time.time())
        params.extend([user_id, fact_id])

        self._conn.execute(
            f"UPDATE facts SET {', '.join(updates)} WHERE user_id = ? AND id = ?",
            params,
        )
        # Re-embed if content changed.
        if content is not None:
            self._delete_embedding(fact_id)
            self._maybe_embed(fact_id, content)
        self._conn.commit()
        return self.get_fact(user_id, fact_id)

    # ── CRUD: Delete ─────────────────────────────────────────────────────

    def delete_fact(self, user_id: str, fact_id: int) -> bool:
        """
        Delete a single fact by its ID (also removes its embedding).

        Args:
            user_id: Unique user identifier.
            fact_id: The fact's numeric ID.

        Returns:
            True if a row was deleted, False otherwise.
        """
        cur = self._conn.execute(
            "DELETE FROM facts WHERE user_id = ? AND id = ?",
            (user_id, fact_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def clear_all(self, user_id: str) -> None:
        """
        Delete ALL facts and their embeddings for a user.

        Args:
            user_id: Unique user identifier.
        """
        self._conn.execute(
            "DELETE FROM embeddings WHERE fact_id IN "
            "(SELECT id FROM facts WHERE user_id = ?)",
            (user_id,),
        )
        self._conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection, checkpointing WAL."""
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    def clear_all_users(self) -> None:
        """Delete ALL facts and embeddings for ALL users.

        This is intended for benchmark/utility use. For routine per-user
        cleanup, use clear_all(user_id) instead.
        """
        self._conn.execute("DELETE FROM embeddings")
        self._conn.execute("DELETE FROM facts")
        self._conn.commit()
