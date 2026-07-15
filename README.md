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

### Install

```bash
# Core (zero deps)
pip install ctrl-memory

# With semantic search
pip install "ctrl-memory[embeddings]"
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

Tested against **[PrecisionMemBench](https://github.com/tenurehq/precisionMemBench)** — 77 retrieval scenarios across alias resolution, fuzzy matching, scope isolation, noise resistance, and more.

| Phase | Passing | Improvement |
|---|---|---|
| Base keyword search | 9 / 77 | — |
| Hybrid keyword + cosine | **42 / 77** | +33 |
| + Damerau-Levenshtein | **43 / 77** | +1 |
| + Scope filtering | **49 / 77** | +6 |
| + Supersession filtering | **54 / 77** | +5 |

**Second-best result** among all tested providers, with zero external dependencies.

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
