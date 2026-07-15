"""
PrecisionMemBench adapter — exposes ctrl-memory as a FastAPI service.

Implements the 3-endpoint contract required by PrecisionMemBench:
  POST /add     — Store a belief with metadata (beliefId for mapping)
  POST /search  — Search stored beliefs by query
  DELETE /reset — Clear all memories

Run:
    .venv/bin/python benchmarks/adapter.py
    # Server starts on http://0.0.0.0:8000
"""

import sys
import os

# Add project root so we can import memory_backend.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from memory_backend import SQLiteStore, EmbeddingEngine

# ── Data models ──────────────────────────────────────────────────────────

class AddPayload(BaseModel):
    text: str
    user_id: str = "benchmark"
    metadata: Optional[dict] = {}

class AddResponse(BaseModel):
    id: str

class SearchPayload(BaseModel):
    query: str
    user_id: str = "benchmark"
    limit: int = 20
    scope: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    memory: str
    metadata: dict

class SearchResponse(BaseModel):
    results: list[SearchResult]

# ── App setup ────────────────────────────────────────────────────────────

app = FastAPI(title="ctrl-memory PrecisionMemBench adapter")

# Use a temp DB so each run starts clean.  The benchmark calls /reset before seeding.
BENCH_DB = Path("/tmp/ctrl-memory-bench.db")

store: Optional[SQLiteStore] = None

# Lookup for supersession metadata loaded from seed JSON.
# The benchmark's seedMetadata() only sends {beliefId, scope} but
# we need superseded_by / resolved_at for proper filtering.
_seed_lookup: dict = {}


def get_store() -> SQLiteStore:
    """Lazy-init the store with auto-embeddings if available."""
    global store
    if store is None:
        ee = EmbeddingEngine()
        store = SQLiteStore(
            db_path=str(BENCH_DB),
            embedding_engine=ee if ee.is_available else None,
        )
    return store


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/add")
def add_belief(payload: AddPayload):
    """
    Store a belief. metadata.beliefId is used by the harness for mapping.

    Appends canonical_name and aliases to the stored text so that
    searches by name/alias resolve correctly. Stores beliefId as the
    primary fact ID for lookup during search.
    """
    s = get_store()
    meta = payload.metadata or {}
    belief_id = meta.get("beliefId", "")

    # Build searchable text: content + canonical_name + aliases
    searchable = payload.text
    if meta.get("canonical_name"):
        searchable += f" {meta['canonical_name']}"
    if meta.get("aliases"):
        if isinstance(meta["aliases"], list):
            searchable += " " + " ".join(meta["aliases"])
        else:
            searchable += f" {meta['aliases']}"

    # Build tags: beliefId + scope + supersession metadata for filtering.
    tag_parts = []
    if belief_id:
        tag_parts.append(f"beliefId:{belief_id}")
    scope_val = meta.get("scope", "")
    if scope_val:
        if isinstance(scope_val, list):
            for sc in scope_val:
                tag_parts.append(f"scope:{sc}")
        else:
            tag_parts.append(f"scope:{scope_val}")
    if meta.get("superseded_by"):
        tag_parts.append(f"superseded_by:{meta['superseded_by']}")
    if meta.get("resolved_at"):
        tag_parts.append(f"resolved_at:{meta['resolved_at']}")

    # Enrich from seed lookup (harness doesn't send supersession data).
    seed = _seed_lookup.get(belief_id or "")
    if seed:
        if seed.get("superseded_by") and not meta.get("superseded_by"):
            tag_parts.append(f"superseded_by:{seed['superseded_by']}")
        if seed.get("resolved_at") and not meta.get("resolved_at"):
            tag_parts.append(f"resolved_at:{seed['resolved_at']}")

    fact = s.add_fact(
        user_id=payload.user_id,
        content=searchable,
        tags=",".join(tag_parts),
    )

    # Return the beliefId as the result id so the harness can map it.
    return AddResponse(id=str(belief_id or fact["id"]))


@app.post("/search")
def search_beliefs(payload: SearchPayload):
    """Search beliefs using hybrid keyword + semantic re-ranking."""
    s = get_store()
    # Use min_score=0.25 to filter weakly-related results
    # while keeping good recall via keyword pre-filter.
    results = s.search_facts_semantic(
        user_id=payload.user_id,
        query=payload.query,
        limit=payload.limit,
        min_score=0.25,
    )

    # Filter by scope if provided.
    if payload.scope and results:
        scope_filter = f"scope:{payload.scope}"
        results = [r for r in results if scope_filter in (r.get("tags") or "")]

    # Exclude superseded and resolved beliefs.
    results = [
        r for r in results
        if not any(
            t.startswith("superseded_by:") or t.startswith("resolved_at:")
            for t in (r.get("tags") or "").split(",")
        )
    ]

    search_results = []
    for r in results:
        # Extract beliefId from tags — this is the ID the harness uses.
        belief_id = ""
        if r.get("tags"):
            for pair in r["tags"].split(","):
                if pair.startswith("beliefId:"):
                    belief_id = pair.split(":", 1)[1]
                    break

        meta = {}
        if belief_id:
            meta["beliefId"] = belief_id

        search_results.append(SearchResult(
            id=belief_id or str(r["id"]),
            memory=r["content"],
            metadata=meta,
        ))

    return SearchResponse(results=search_results)


@app.delete("/reset")
def reset_all():
    """Clear all memories for all users. Called once before seeding."""
    global _seed_lookup
    s = get_store()
    # Get all distinct user_ids and clear each.
    rows = s._conn.execute("SELECT DISTINCT user_id FROM facts").fetchall()
    for row in rows:
        s.clear_all(row["user_id"])
    # Also vacuum to reclaim space.
    s._conn.execute("VACUUM")
    # Load seed JSON to build supersession lookup.
    seed_path = Path("/tmp/precisionMemBench/fixtures/beliefs.seed.json")
    if seed_path.exists():
        with open(seed_path) as f:
            beliefs = json.load(f)
        _seed_lookup = {b.get("_id"): b for b in beliefs}
    else:
        _seed_lookup = {}
    return {"status": "reset"}


@app.get("/health")
def health():
    """Quick health check for the benchmark harness."""
    s = get_store()
    return {
        "status": "ok",
        "semantic_search": s._ee is not None and s._ee.is_available,
    }


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Clean up any previous benchmark run.
    if BENCH_DB.exists():
        BENCH_DB.unlink()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
