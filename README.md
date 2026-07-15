# ctrl-memory

**Lightweight, local-first MCP memory server for LLM agents.**

Store, search, and retrieve agent memories with zero external dependencies. Plug it into Hermes, Claude Code, Cursor, or any MCP-compatible client.

```bash
pip install ctrl-memory
ctrl-memory-mcp
```

---

## ✨ Features

- **Zero-dependency core** — pure Python, no databases, no external services
- **Two backends** — JSON files (MVP, zero deps) or SQLite (production, WAL mode)
- **Semantic search** — optional `sentence-transformers` for cosine similarity ranking
- **Hybrid retrieval** — keyword recall + vector re-ranking for best precision/recall
- **Fuzzy matching** — Damerau-Levenshtein typo tolerance built in
- **Scope filtering** — tag-based domain isolation
- **Supersession awareness** — automatically filters outdated/obsolete facts
- **Hermes plugin** — auto-prefetch context, auto-capture conversation turns
- **MCP stdio server** — plug into any MCP-compatible client
- **User isolation** — each user's memory is fully separated
- **Cross-session persistence** — memories survive between conversations

---

## 🚀 Quick start

### One-liner install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/ctrlProgrammer/ctrl-memory-system/main/install.sh | bash
```

This creates an isolated virtual environment in `~/.local/share/ctrl-memory/`, installs ctrl-memory with semantic search, and makes the `ctrl-memory-mcp` command available globally. No `pip install --user` or `sudo` needed.

After install, open a new terminal (or `exec $SHELL`) and run:

```bash
ctrl-memory-mcp
```

### Manual install with pip

```bash
# Core (zero deps)
pip install ctrl-memory

# With semantic search
pip install "ctrl-memory[embeddings]"
```

> ⚠️ If your system restricts global pip installs (PEP 668), use the one-liner above or install inside a virtual environment:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> pip install "ctrl-memory[embeddings]"
> ```

### pipx (alternative)

```bash
pipx install "ctrl-memory[embeddings]"
```

### Run the MCP server

```bash
ctrl-memory-mcp
```

By default it uses JSON files in `~/.ctrl-memory/`. For SQLite:

```bash
ctrl-memory-mcp --backend sqlite
```

### Test it

```bash
# Add a fact
echo '{"jsonrpc":"2.0","id":1,"method":"add_memory","params":{"user_id":"alice","content":"Alice prefers Fastify over Express for Node.js APIs"}}' | ctrl-memory-mcp

# Search
echo '{"jsonrpc":"2.0","id":2,"method":"search_memory","params":{"user_id":"alice","query":"what framework does Alice use?"}}' | ctrl-memory-mcp
```

---

## 📦 Backends

### JSON backend (default)

- Zero dependencies
- One file per user: `~/.ctrl-memory/<user_id>.json`
- Auto-increment IDs, append-only writes
- Best for: MVP, personal use, <1000 facts

### SQLite backend

- Single `.db` file with WAL mode
- Indexed queries, ACID transactions
- Embedding storage for semantic search
- Best for: production, multi-user, >1000 facts

```bash
ctrl-memory-mcp --backend sqlite
```

---

## 🧠 Semantic search

When `sentence-transformers` is installed, ctrl-memory automatically enables:

- **Auto-embedding** — facts are vectorized at write time (384-dim all-MiniLM-L6-v2)
- **Hybrid search** — keyword candidates → cosine similarity re-ranking → sorted by relevance
- **Score filtering** — configurable `min_score` threshold to filter weak matches

```bash
pip install "ctrl-memory[embeddings]"
```

No flags needed — detection is automatic.

---

## 🔌 Hermes Agent plugin

ctrl-memory ships with a native Hermes Agent provider.

### Install

```bash
# Copy the plugin
cp -r hermes_provider ~/.hermes/hermes-agent/plugins/memory/ctrl-memory/
```

### Configure

Add to `~/.hermes/config.yaml`:

```yaml
memory:
  provider: ctrl-memory
  config:
    backend: sqlite       # or json (default)
    db_path: ~/.hermes/memory.db
```

The plugin provides:
- **Automatic prefetch** — relevant context injected before every LLM turn
- **Turn capture** — facts extracted from conversations and stored
- **4 tools** — `add_memory`, `search_memory`, `delete_memory`, `memory_status`

---

## 🔧 MCP tools

| Tool | Description |
|---|---|
| `add_memory` | Store a new fact with optional metadata tags |
| `search_memory` | Hybrid keyword + semantic search |
| `get_fact` | Retrieve a specific fact by ID |
| `list_facts` | List all facts for a user with pagination |
| `update_fact` | Edit an existing fact (re-embeds if semantic enabled) |
| `delete_fact` | Remove a fact (cleans up embedding) |
| `count_facts` | Get total fact count for a user |
| `search_memory_semantic` | Pure cosine similarity search (uses embeddings) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│          MCP Client                  │
│  (Hermes, Claude Code, Cursor...)   │
└──────────────┬──────────────────────┘
               │ stdio JSON-RPC
┌──────────────▼──────────────────────┐
│        mcp_server.py                │
│     MCP stdio transport layer       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│       memory_backend.py             │
│  ┌─────────┐  ┌──────────┐         │
│  │ JSON    │  │ SQLite   │         │
│  │ Store   │  │ Store    │         │
│  └─────────┘  └────┬─────┘         │
│                    │                │
│  ┌─────────────────▼──────────┐     │
│  │    EmbeddingEngine          │     │
│  │  (optional, auto-detect)   │     │
│  └────────────────────────────┘     │
└─────────────────────────────────────┘
```

### Search flow

```
Query
  │
  ▼
1. Keyword search (token-OR with stop-word filter)
  │
  ├── No results? → Fuzzy fallback (Damerau-Levenshtein)
  │
  ▼
2. Cosine similarity re-ranking (if embeddings available)
  │
  ▼
3. Scope filtering (if tags match)
  │
  ▼
4. Supersession filtering (remove obsolete facts)
  │
  ▼
5. Sort by score, return top N
```

---

## 📊 Benchmark results

Tested against **[PrecisionMemBench](https://github.com/tenurehq/precisionMemBench)** — 77 retrieval scenarios across alias resolution, fuzzy matching, scope isolation, noise resistance, supersession chains, and budget constraints.

### Score: **54 / 77** ✅ (70% passing)

| Phase | Passing | Δ |
|---|---|---|
| Base keyword search | 9 / 77 | — |
| Hybrid keyword + cosine | **42 / 77** | +33 |
| + Damerau-Levenshtein fuzzy | **43 / 77** | +1 |
| + Scope filtering | **49 / 77** | +6 |
| + Supersession filtering | **54 / 77** | +5 |

### What passes (54 tests)

| Category | Tests |
|---|---|
| **Alias resolution** | k8s → Kubernetes, GHA → GitHub Actions, ReactJS → React, POV → point of view, DLQ → dead letter queue, base class → composition-inheritance, exceptions → error handling, 2-word shingles |
| **Exact match** | repository-layer, canonical name, long query (400+ chars) |
| **Scope filtering** | Redis-in-writing returns character not datastore, code-scope doesn't leak writing, cross-scope blocked, user-edited respects scope, scope bleed protection |
| **Supersession** | SQLAlchemy superseded by MongoDB hidden, TSLint→ESLint→Biome chain resolves to terminal, resolved_at beliefs excluded, pinned+resolved excluded from questions |
| **Fuzzy matching** | All-caps case-insensitive (REACTJS), typo prefix guard, scope-aware fuzzy filtering |
| **Messy queries** | Filler-heavy extraction, compound surfaces 2 beliefs, negation still surfaces topic, all-caps case insensitive |
| **Budget/limits** | Ceiling eviction, zero graceful, one pinned wins, recency tiebreak |
| **Edge cases** | Cold start, empty query, whitespace query, short query passthrough, short query score clears, empty alias content path |
| **User isolation** | Other-user beliefs never leak |
| **Universal scope** | Persona prelude, explicit query with no relevant, zero-reinforcement fresh belief surfaces |

### What fails (23 tests) — root causes

| Cause | Tests | Why |
|---|---|---|
| **Token bleeding** | 8 | Generic query tokens match too many facts (e.g. "kube" finds multiple) |
| **Relation expansion** | 4 | "auth depends on redis" — not yet implemented |
| **Cap stress** | 3 | 6+ entities in single query needs NLP extraction |
| **Ranking weights** | 2 | canonical_name should outrank content match |
| **Multi-scope Redis** | 2 | Redis in both code+writing scopes needs per-scope dedup |
| **Fuzzy edge cases** | 2 | k9s vs k8s prefix guard, single-edit with token bleed |
| **Other** | 2 | why_it_matters not indexed, type isolation routing |

### Comparison with other providers

| Provider | Passing | Precision | Dependencies |
|---|---|---|---|
| **tenure** (reference) | **77 / 77** | 1.00 | MongoDB Atlas Search (BM25, shingles, fuzzy) |
| **ctrl-memory** | **54 / 77** | **~0.70** | **Zero external deps** |
| okf | ~30 / 77 | ~0.47 | PostgreSQL |
| supermemory | ~21 / 77 | ~0.22 | Supabase + API |
| yourmemory | ~21 / 77 | ~0.17 | MongoDB |
| mem0 | ~9 / 77 | ~0.06 | Qdrant + API |

**Second place** among all tested providers. ctrl-memory achieves this with **zero external dependencies** — no databases, no APIs, no cloud services.

---

## 🧪 Development

```bash
# Clone
git clone https://github.com/ctrl-alt-dev/ctrl-memory
cd ctrl-memory

# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[embeddings]"

# Test
python3 -m unittest discover tests -v

# Run benchmark
cd /tmp
git clone https://github.com/tenurehq/precisionMemBench
cd precisionMemBench
MEMORY_PROVIDER=ctrl-memory CTRL_MEMORY_URL=http://localhost:8000 \
  RESEED=true npx ava src/retrieval.external.eval.test.ts --timeout 10m
```

### Test suite

| File | Tests | What it covers |
|---|---|---|
| `test_memory_backend.py` | 23 | JSON store CRUD, search, user isolation |
| `test_sqlite_store.py` | 25 | SQLite store CRUD, search, embeddings |
| `test_mcp_server.py` | 24 | MCP protocol, JSON-RPC, tool dispatch |
| `test_embeddings.py` | 24 | Embedding engine, cosine similarity |
| `test_hermes_provider.py` | 24 | Plugin lifecycle, tools, prefetch |

**116 tests total** (105 run, 11 skip without sentence-transformers).

---

## 📄 License

MIT
