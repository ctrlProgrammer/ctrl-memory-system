# Code Audit Report: ctrl-memory v0.3.0

**Audit date:** 2026-07-15
**Scope:** Full codebase (all source, tests, benchmarks, installer, config)
**Project root:** `/home/ctrl/Projects/Personal/ctrl-memory-system`
**Files reviewed:** 15 (6 source, 5 test, 1 benchmark, 1 installer, 1 config, 1 stub)

---

## Critical

### C-1: Near-duplicate `mcp_server.py` files create maintenance liability

**Files:** `ctrl_memory/server.py` ↔ `mcp_server.py` (root)
**Severity:** CRITICAL — Bug fixes must be applied to both, divergence inevitable

Two files with nearly identical `MCPMemoryServer` class exist:
- `ctrl_memory/server.py` imports from `ctrl_memory.backend`
- `mcp_server.py` imports from `memory_backend` (which is just `from ctrl_memory.backend import *`)

The tool definitions, dispatch tables, handlers, and `main()` are duplicated verbatim (~550 lines each). Any bug fix, schema change, or feature addition must be replicated in both. The root `mcp_server.py` is what `tests/test_mcp_server.py` actually invokes via `_find_server_script()`, so it's the de facto entry point; `ctrl_memory/server.py` is registered as the console_scripts entry in pyproject.toml.

**Recommendation:** Eliminate one. Either make `mcp_server.py` a thin re-export from `ctrl_memory.server`, or make the console_scripts entry point reference `mcp_server.py:main` directly.

---

### C-2: `EmbeddingEngine._load_model` silently swallows ALL exceptions

**File:** `ctrl_memory/backend.py`, lines 70–78
**Severity:** CRITICAL — Users get no diagnostic when embeddings silently fail

```python
def _load_model(self) -> None:
    try:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
    except ImportError:
        self._model = None
    except Exception:          # ← EVERYTHING swallowed: OOM, model download failure, corrupt cache, etc.
        self._model = None
```

Every non-ImportError exception is caught and silently discarded. If the model download fails (network issue, disk full), CUDA runs out of memory, or the model cache is corrupt, the engine reports `is_available=False` with zero logging. The user sees "semantic search unavailable" and has no way to diagnose why.

**Same pattern in** `hermes_provider/__init__.py` lines 38–41.

**Recommendation:** Log the exception message at `warning` level before swallowing, or at minimum differentiate `ImportError` from unexpected failures.

---

### C-3: Hermes plugin inline backend diverges from main SQLite backend

**File:** `hermes_provider/__init__.py`, lines 53–175
**Severity:** CRITICAL — Plugin behaves differently from MCP server on the same database

The `hermes_provider/__init__.py` contains an inline copy of `SQLiteStore` (the fallback when `from ctrl_memory.backend import SQLiteStore` fails). This inline version differs from the canonical `SQLiteStore` in `ctrl_memory/backend.py`:

| Feature | Main `SQLiteStore` | Inline plugin copy |
|---|---|---|
| `update_fact()` | ✅ Full implementation | ❌ **Not implemented** — plugin can't update facts |
| `search_facts()` | Multi-token OR with stop words | Single-token LIKE (whole query as one token) |
| `search_facts_semantic()` | Hybrid: keyword pre-filter + semantic re-rank | Full vector scan of all facts (no pre-filter) |
| `delete_fact()` | Embeddings CASCADE removed | No explicit embedding cleanup |
| `get_fact()` | ✅ Implemented | ❌ **Not implemented** |
| `_pack_embedding()` | Imports from module-level helper | Inline `import struct` each time |

The plugin's `search_facts` does a single `%query%` LIKE, while the main backend splits the query into tokens filtered by stop words and does OR matching. Same database, different search behavior.

**Recommendation:** Remove the inline fallback. Make the `ctrl_memory` package a hard dependency of the plugin, or ensure the inline copy is a generated/synced file with a test that fails if they diverge.

---

### C-4: `EmbeddingEngine.__init__` hardcodes vector dimension independent of model

**File:** `ctrl_memory/backend.py`, line 67
**Severity:** CRITICAL — Wrong dimension if a different model is used

```python
self._dimension = 384  # all-MiniLM-L6-v2 output size
```

This is only correct for the default model `all-MiniLM-L6-v2`. If a user passes `model_name="all-mpnet-base-v2"` (768 dimensions), the `_dimension` attribute is wrong. This affects:
- Any code that reads `ee.dimension` 
- The `_pack_embedding`/`_unpack_embedding` helpers (they infer dimension from blob length, so they're safe, but the public API lies)

**Recommendation:** Read the dimension from the loaded model (e.g., `self._model.get_sentence_embedding_dimension()`) after a successful load, and only fall back to 384 if not determinable.

---

## High

### H-1: Test `test_semantic_search_fallback_without_engine` expects wrong behavior

**File:** `tests/test_embeddings.py`, lines 273–279
**Severity:** HIGH — Test asserts 0 results but the code returns keyword results

```python
def test_semantic_search_fallback_without_engine(self):
    store_no_ee = SQLiteStore(db_path=self.tmp_file)
    store_no_ee.add_fact(self.user, "Test fact")
    results = store_no_ee.search_facts_semantic(self.user, "test")
    self.assertEqual(len(results), 0)  # ← BUG: should be > 0
```

The main `SQLiteStore.search_facts_semantic()` (backend.py, lines 968–972) falls back to keyword search results with `score=0.0` when no embedding engine is configured. So `search_facts_semantic("test")` with a fact containing "test" should return 1 result. The test was written against an older version of the code that returned empty when no engine was available. The code was changed to add hybrid fallback, but the test wasn't updated.

**This test is currently **failing** if the test suite runs without sentence-transformers.**

---

### H-2: Plugin test mocks `agent.memory_provider` at module level, affecting import order

**File:** `tests/test_hermes_provider.py`, lines 58–61
**Severity:** HIGH — Fragile import-time side effect, affects other tests if loaded first

```python
import sys
if "agent.memory_provider" not in sys.modules:
    sys.modules["agent.memory_provider"] = MagicMock()
    sys.modules["agent.memory_provider"].MemoryProvider = MockMemoryProvider

from hermes_provider import CtrlMemoryProvider, ...
```

This monkey-patches `sys.modules` at import time to prevent `hermes_provider/__init__.py` from failing on `from agent.memory_provider import MemoryProvider`. This has two problems:
1. **Global side effect**: If this test module is imported (even just for test discovery) before any test that might need the real `agent.memory_provider`, it poisons the module cache.
2. **No isolation**: Running this single test via `python -m unittest tests.test_hermes_provider` works, but running it alongside other tests that also need `agent.memory_provider` can break silently.

**Recommendation:** Use `unittest.mock.patch.dict('sys.modules', ...)` scoped to `setUpClass`/`setUp`, or restructure the plugin to accept the `MemoryProvider` base class as a dependency.

---

### H-3: `SQLiteStore` has no finalizer — WAL may not flush on gc

**File:** `ctrl_memory/backend.py` (entire `SQLiteStore` class)
**Severity:** HIGH — Potential data loss on unclean shutdown

The `SQLiteStore` class has a `close()` method but:
- `__del__` is not defined to call it
- The `MCPMemoryServer.run()` event loop has no signal handler (SIGTERM/SIGINT)
- If the process is killed or the object garbage-collected without explicit `close()`, the WAL may not be checkpointed

The `MCPMemoryServer.run()` loop (server.py, lines 475+) exits only on EOF from stdin, but doesn't call `self.store.close()` or otherwise clean up.

**Recommendation:** Add `__del__` or `__enter__`/`__exit__` to `SQLiteStore`, and add a `try/finally` in `run()` or register an `atexit` handler.

---

### H-4: Install script `dirname $0` check uses `-d` on a file, always false

**File:** `install.sh`, line 102
**Severity:** HIGH — Local install path detection is broken

```bash
if [ -d "$(dirname "$0")/memory_backend.py" ] || [ -f "$(dirname "$0")/pyproject.toml" ]; then
```

The first condition checks if `memory_backend.py` is a **directory** (`-d`), which is always false for a regular file. The second condition `[ -f "pyproject.toml" ]` is correct. So local install detection works only because of the second check, but if `pyproject.toml` were ever absent (e.g., running from a tarball with only the wheel), it would silently fall through to remote GitHub install.

---

### H-5: Version mismatch: plugin.yaml says 0.2.0, rest says 0.3.0

**Files:** `hermes_provider/plugin.yaml` line 2 vs `ctrl_memory/__init__.py` line 2, `pyproject.toml` line 3
**Severity:** HIGH — Confusion for Hermes plugin versioning/discovery

- `pyproject.toml`: `version = "0.3.0"`
- `ctrl_memory/__init__.py`: `__version__ = "0.3.0"`
- `hermes_provider/plugin.yaml`: `version: 0.2.0`

The plugin manifest lags behind by a full minor version.

---

## Medium

### M-1: JSON `MemoryStore` has no concurrency protection

**File:** `ctrl_memory/backend.py`, class `MemoryStore` (lines 284–576)
**Severity:** MEDIUM — Data corruption under concurrent access

The JSON backend reads, mutates, and writes the entire file per operation (`_load` → modify → `_save`). There is no file locking, no atomic rename, and no retry logic. Two processes writing to the same user file will silently corrupt each other's data. This is called out in the SQLiteStore docstring as a differentiator ("Concurrent-safe (WAL mode)"), but the JSON backend remains unprotected.

**Recommendation:** Document that MemoryStore is single-process only, or add `fcntl.flock` / `portalocker`-style locking.

---

### M-2: `search_facts` in both backends uses ANY-token matching (OR logic), not ALL-tokens

**File:** `ctrl_memory/backend.py`, lines 499–500 (MemoryStore) and 831–843 (SQLiteStore)
**Severity:** MEDIUM — Query "PostgreSQL deployment" matches facts containing only "deployment" or only "PostgreSQL" but not both

```python
# Match if ANY meaningful token appears.
for token in tokens:
    if token in content_lower or token in tags_lower:
        results.append(f)
        break
```

A search for "PostgreSQL deployment strategy" could return a fact about "Deployment to us-west-2" (matching only "deployment") and a separate fact about "Database is PostgreSQL 16" (matching only "PostgreSQL"), both of low relevance. This broad recall is intentional (per docstring), but combined with the `results.sort` after, the user sees unrelated facts ranked by recency, not by how many tokens matched.

**No token-count ranking is applied.** A fact matching 3/3 tokens is scored identically to one matching 1/3.

---

### M-3: `benchmarks/adapter.py` accesses `store._conn` and `store._ee` directly (private attributes)

**File:** `benchmarks/adapter.py`, lines 197–199, 219
**Severity:** MEDIUM — Breaks encapsulation, fragile under refactoring

```python
rows = s._conn.execute("SELECT DISTINCT user_id FROM facts").fetchall()  # line 197
semantic_search = s._ee is not None and s._ee.is_available  # line 219
```

The benchmark adapter reaches into private implementation details (`_conn`, `_ee`) instead of using public API methods (`clear_all`, `count_facts`, etc.). If the backend is refactored, the benchmark silently breaks at runtime.

---

### M-4: Hermes provider `initialize()` uses `kwargs.get("hermes_home", "~/.hermes")` — tilde not expanded

**File:** `hermes_provider/__init__.py`, line 266
**Severity:** MEDIUM — Fallback creates literal `~/.hermes/ctrl-memory/memory.db`

```python
hermes_home = Path(kwargs.get("hermes_home", "~/.hermes"))
```

If `hermes_home` is not passed in kwargs, the fallback `"~/.hermes"` is used verbatim — `Path("~/.hermes")` does NOT expand `~`. It should be `Path("~/.hermes").expanduser()`. If `hermes_home` IS passed by Hermes (which it should be), this works fine, but the fallback path would create a literal tilde directory.

---

### M-5: No input validation or size limits on `content` or `tags`

**File:** `ctrl_memory/backend.py`, `add_fact()` methods
**Severity:** MEDIUM — No protection against pathological inputs

- `content` has no maximum length
- `tags` has no format validation
- No sanitization against null bytes, control characters, or excessively long strings
- The `SQLiteStore` embedding call on a 100MB string would OOM the process
- The `MemoryStore` JSON serialization of a large content field could fail silently

---

### M-6: `sync_turn` auto-captures everything matching first-person markers, no deduplication

**File:** `hermes_provider/__init__.py`, lines 332–361
**Severity:** MEDIUM — Every turn with "I" or "my" creates a new fact; no similarity check

```python
markers = [" i ", " my ", " we ", " our ", " i'm ", " i've ", " i use "]
```

- "I think the weather is nice" is stored as a fact with tag "auto-captured"
- "I think the weather is nice today" in the next turn is stored as a **separate** fact
- No deduplication, no similarity check, no cooldown
- Over many sessions, "auto-captured" facts can accumulate rapidly with near-duplicate entries

Auto-capture also matches queries: "What do I have stored about X?" → stores "What do I have stored about X?" as a fact.

---

### M-7: No `.gitignore` for Python artifacts

**File:** `.gitignore` — **Does not exist**
**Severity:** MEDIUM — Build artifacts can be accidentally committed

The project has no `.gitignore`. The `__pycache__/`, `*.egg-info/`, `.venv/`, and other build artifacts may be tracked. (The repo currently shows clean status, suggesting they may have been manually excluded or the user has a global gitignore.)

---

### M-8: `sync_turn` tags as `auto-captured` but missing in `TOOL_DISPATCH` for `delete_memory` — plugin can't update auto-captured facts

**File:** `hermes_provider/__init__.py`, lines 207–218
**Severity:** MEDIUM — Plugin provides `ctrl_memory_delete` but has no `ctrl_memory_update`, so auto-captured facts can only be deleted, not corrected

The MCP server has `update_memory` as a tool; the Hermes plugin does not expose one. A user whose auto-captured fact is wrong has to delete and re-add.

---

## Low

### L-1: Test `test_default_storage_dir_is_created` creates/removes `~/.ctrl-memory`

**File:** `tests/test_memory_backend.py`, lines 234–244
**Severity:** LOW — Side effect on user's home directory

```python
store = MemoryStore()  # Creates ~/.ctrl-memory if it doesn't exist
shutil.rmtree(store._dir, ignore_errors=True)  # Deletes ~/.ctrl-memory
```

This test operates outside its temp directory. If the test crashes between these lines, the user is left with an empty `~/.ctrl-memory/`. Other tests use `tempfile.mkdtemp()` but this one does not.

---

### L-2: `memory_backend.py` uses star import (`from ctrl_memory.backend import *`)

**File:** `memory_backend.py`, line 2
**Severity:** LOW — Pollutes namespace, exports internal helpers

Exports `_levenshtein`, `_damerau_levenshtein`, `_fuzzy_token_match`, `_pack_embedding`, `_unpack_embedding` — all of which are prefixed with underscore as internal helpers. These become part of the `memory_backend` public API by accident.

---

### L-3: `cosine_similarity` recomputes norms every call (no caching)

**File:** `ctrl_memory/backend.py`, lines 137–142
**Severity:** LOW — Minor performance issue in tight loops

```python
norm_a = math.sqrt(sum(x * x for x in a))
norm_b = math.sqrt(sum(y * y for y in b))
```

Since `normalize_embeddings=True` is passed to the model, all vectors from `embed()` and `embed_many()` are unit-normalized. In that case, `norm_a == norm_b == 1.0` and the division is a no-op. However:
- Vectors loaded from the database (via `_unpack_embedding`) may not have been normalized at storage time
- `search_facts_semantic` recomputes norms for every candidate even though the stored vectors are all unit-length

**Recommendation:** For the common case where vectors are pre-normalized, skip the sqrt. Or store a flag alongside the embedding indicating whether it's normalized.

---

### L-4: `install.sh` uses `dirname $0` which is unreliable with piped install

**File:** `install.sh`, lines 102, 144, 157
**Severity:** LOW — PATH may be incorrect when installed via curl pipe | bash

When run as `curl -fsSL ... | bash`, `$0` is a temporary file descriptor like `/dev/stdin` or a bash alias. The `dirname "$0"` / `$(dirname "$0")` usage for detecting local filesystem paths only works when the script is run from a local file:

```bash
if [ -d "$(dirname "$0")/memory_backend.py" ]  # Always false when piped
PLUGIN_SRC="$(dirname "$0")/hermes_provider"    # Wrong path when piped
cp -r "$PLUGIN_SRC/"* "$HERMES_PLUGIN_DIR/"     # Copies nothing when piped
```

The remote install path (the `else` branch with `curl ... api.github.com/...`) is the fallback, so piped installs work — but the local path detection is misleading.

---

### L-5: `test_update_fact_no_changes` calls `update_fact` with no keyword args — relies on all-None handling

**File:** `tests/test_sqlite_store.py`, lines 106–110
**Severity:** LOW — Tests an edge case that's more of a no-op than a meaningful operation

```python
def test_update_fact_no_changes(self):
    fact = self.store.add_fact(self.user, "No change")
    updated = self.store.update_fact(self.user, fact["id"])  # content=None, tags=None
    self.assertEqual(updated["content"], "No change")
```

In `SQLiteStore.update_fact`, when `content is None` and `tags is None`, the `updates` list is empty and `return existing` (line 1016). No SQL is executed. The test passes, but it doesn't test any actual persistence logic.

---

### L-6: Test classes in `test_embeddings.py` share file via `self.tmp_file` across sibling test methods

**File:** `tests/test_embeddings.py`
**Severity:** LOW — Test isolation is fragile

`TestSQLiteStoreAutoEmbedding` and `TestSQLiteStoreSemanticSearch` both use `self.tmp_file` and create their own stores from `setUp`/`tearDown` — so isolation is maintained. However, `test_semantic_search_fallback_without_engine` inside `TestSQLiteStoreSemanticSearch` creates a second store `store_no_ee` pointing at `self.tmp_file` but does NOT close it in `tearDown`. The `setUp`'s store is closed by `tearDown`, but `store_no_ee` leaks a connection.

---

### L-7: `EmbeddingEngine.embed_many` not used anywhere

**File:** `ctrl_memory/backend.py`, lines 109–123
**Severity:** LOW — Dead code

`embed_many()` is defined but never called. `SQLiteStore._maybe_embed` (line 714) calls `self._ee.embed(content)` (single) for each fact individually. Batch embedding would be much faster for initial bulk loads but isn't wired up.

---

## Summary

| Severity | Count | Key issues |
|---|---|---|
| **CRITICAL** | 4 | Duplicate server files, silent exception swallowing, plugin backend divergence, hardcoded dimension |
| **HIGH** | 5 | Wrong test assertion, fragile import mocking, missing finalizer/signal handler, broken install detection, version mismatch |
| **MEDIUM** | 8 | No concurrency protection, OR-match recall without ranking, private attr access, tilde not expanded, no input validation, auto-capture dedup, no gitignore, missing update tool |
| **LOW** | 7 | Home-dir side effect, star import, redundant norm computation, piped-install fragility, minor test issues, dead code |

**Next step recommendation:** Tackle the CRITICAL items first — especially the duplicate server files (C-1) and the silent exception swallowing (C-2), which are both easy to fix and have the highest impact on correctness and debuggability.
