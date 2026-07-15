"""
mcp_server.py — MCP stdio server wrapping the MemoryStore backend.

Provides a Model Context Protocol (MCP) interface over stdin/stdout
so any MCP-compatible agent (Hermes, Claude Code, Cursor, Windsurf)
can store and recall facts across sessions.

Protocol: JSON-RPC 2.0 over stdio (one JSON object per line each way).
"""

import json
import sys
import traceback
from typing import Any, Callable, Dict

from ctrl_memory.backend import (
    MemoryStore,
    SQLiteStore,
    FactNotFoundError,
    create_store,
    EmbeddingEngine,
)


# ── Constants ────────────────────────────────────────────────────────────

#: MCP protocol version this server implements.
PROTOCOL_VERSION = "0.1.0"

#: Server identity sent during initialization handshake.
SERVER_INFO = {
    "name": "ctrl-memory",
    "version": "0.1.0",
}


# ── Tool Definitions ────────────────────────────────────────────────────
# These define the interface that MCP agents see in their tool list.
# Each matches an OpenAI-style function-calling schema.

TOOL_DEFINITIONS = [
    {
        "name": "add_memory",
        "description": (
            "Store a fact the user wants remembered across sessions. "
            "Use this whenever the user shares personal info, preferences, "
            "project details, decisions, or anything they'll ask about later."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
                "content": {
                    "type": "string",
                    "description": "The fact text to remember.",
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated tags (e.g. 'preference,project').",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Search stored facts by keyword. "
            "Case-insensitive substring match across both content and tags. "
            "Results are sorted by recency, most recent first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
                "query": {
                    "type": "string",
                    "description": "Keyword to search for in facts.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_memory",
        "description": "Retrieve a single stored fact by its numeric ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
                "fact_id": {
                    "type": "integer",
                    "description": "The fact's numeric ID.",
                },
            },
            "required": ["fact_id"],
        },
    },
    {
        "name": "list_memories",
        "description": "List ALL stored facts for a user. Use sparingly — may return many items.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
            },
            "required": [],
        },
    },
    {
        "name": "update_memory",
        "description": "Update an existing fact's content and/or tags.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
                "fact_id": {
                    "type": "integer",
                    "description": "The fact's numeric ID.",
                },
                "content": {
                    "type": "string",
                    "description": "New content text.",
                },
                "tags": {
                    "type": "string",
                    "description": "New comma-separated tags.",
                },
            },
            "required": ["fact_id"],
        },
    },
    {
        "name": "delete_memory",
        "description": "Delete a specific fact by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
                "fact_id": {
                    "type": "integer",
                    "description": "The fact's numeric ID to delete.",
                },
            },
            "required": ["fact_id"],
        },
    },
    {
        "name": "count_memories",
        "description": "Count how many facts are stored for a user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_memory_semantic",
        "description": (
            "Search facts by meaning (semantic similarity), not keywords. "
            "Requires SQLite backend with --embed enabled. "
            "Use this for natural-language queries like 'what's my deployment setup?' "
            "Returns results with a similarity score (0-1). "
            "Falls back to keyword search if embeddings aren't available."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier (default: 'default').",
                },
                "query": {
                    "type": "string",
                    "description": "Natural-language query (e.g. 'deployment preferences').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10).",
                },
            },
            "required": ["query"],
        },
    },
]

#: Map tool name → (method_on_MemoryStore, requires_user_id, requires_params).
#: Each entry describes how to dispatch the tool call.
TOOL_DISPATCH: Dict[str, Dict[str, Any]] = {
    "add_memory": {
        "method": "add_fact",
        "required_params": ["content"],
        "optional_params": {"tags": ""},
        "user_id_default": "default",
    },
    "search_memory": {
        "method": "search_facts",
        "required_params": ["query"],
        "optional_params": {"limit": 10},
        "user_id_default": "default",
    },
    "get_memory": {
        "method": "get_fact",
        "required_params": ["fact_id"],
        "optional_params": {},
        "user_id_default": "default",
    },
    "list_memories": {
        "method": "get_all_facts",
        "required_params": [],
        "optional_params": {},
        "user_id_default": "default",
    },
    "update_memory": {
        "method": "update_fact",
        "required_params": ["fact_id"],
        "optional_params": {"content": None, "tags": None},
        "user_id_default": "default",
    },
    "delete_memory": {
        "method": "delete_fact",
        "required_params": ["fact_id"],
        "optional_params": {},
        "user_id_default": "default",
    },
    "count_memories": {
        "method": "count_facts",
        "required_params": [],
        "optional_params": {},
        "user_id_default": "default",
    },
    "search_memory_semantic": {
        "method": "search_facts_semantic",
        "required_params": ["query"],
        "optional_params": {"limit": 10},
        "user_id_default": "default",
    },
}


# ── Helper: Build JSON-RPC Response ─────────────────────────────────────

def _success_response(req_id: Any, result: Any) -> dict:
    """Build a standard JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error_response(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    """Build a standard JSON-RPC error response."""
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def _text_content(text: str) -> list:
    """Wrap plain text in the MCP content array format."""
    return [{"type": "text", "text": text}]


def _json_text(obj: Any) -> list:
    """Wrap a JSON-serializable object as MCP text content."""
    return _text_content(json.dumps(obj, indent=2, ensure_ascii=False))


# ── MCP Server ──────────────────────────────────────────────────────────

class MCPMemoryServer:
    """
    MCP server that exposes MemoryStore operations as tools over stdio.

    Reads JSON-RPC 2.0 requests from stdin and writes responses to stdout.
    Stderr is reserved for logging and diagnostics.

    Usage:
        server = MCPMemoryServer()                      # JSON backend
        server = MCPMemoryServer(backend="sqlite")      # SQLite + auto-embeddings if available
        server.run()
    """

    def __init__(
        self,
        backend: str = "json",
        storage_dir: str | None = None,
        db_path: str | None = None,
    ) -> None:
        """
        Initialize the server with the chosen storage backend.

        When using SQLite, the server automatically attempts to load an
        EmbeddingEngine for semantic search. If sentence-transformers is
        not installed, it falls back gracefully to keyword search.

        Args:
            backend:    "json" (default) or "sqlite".
            storage_dir: JSON backend: directory for user fact files.
            db_path:     SQLite backend: path to the database file.
        """
        if backend == "sqlite":
            # Auto-detect embeddings — works if sentence-transformers is installed.
            ee = EmbeddingEngine()
            if not ee.is_available:
                print(
                    "⚠️  Semantic search unavailable. Install:  pip install ctrl-memory[embeddings]",
                    file=sys.stderr,
                )
            self.store = SQLiteStore(db_path=db_path, embedding_engine=ee if ee.is_available else None)
        else:
            self.store = MemoryStore(storage_dir=storage_dir)

    # ── Request Router ───────────────────────────────────────────────────

    def handle_request(self, request: dict) -> dict:
        """
        Route a single JSON-RPC request to the appropriate handler.

        Args:
            request: Parsed JSON-RPC 2.0 request dict.

        Returns:
            JSON-RPC 2.0 response dict (success or error).
        """
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        # ── Lifecycle ────────────────────────────────────────────────
        if method == "initialize":
            return self._handle_initialize(req_id, params)

        # ── Tool discovery ───────────────────────────────────────────
        if method == "tools/list":
            return self._handle_tools_list(req_id)

        # ── Tool execution ───────────────────────────────────────────
        if method == "tools/call":
            return self._handle_tool_call(req_id, params)

        # ── Unknown method ───────────────────────────────────────────
        return _error_response(
            req_id,
            code=-32601,  # Method not found
            message=f"Unknown method: {method}",
        )

    # ── Handlers ─────────────────────────────────────────────────────────

    def _handle_initialize(self, req_id: Any, params: dict) -> dict:
        """
        Handle the MCP initialization handshake.

        The client sends capabilities and expects server capabilities + info.
        """
        return _success_response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })

    def _handle_tools_list(self, req_id: Any) -> dict:
        """Return the list of available tools and their schemas."""
        return _success_response(req_id, {"tools": TOOL_DEFINITIONS})

    def _handle_tool_call(self, req_id: Any, params: dict) -> dict:
        """
        Execute a tool by name with provided arguments.

        Looks up the tool in TOOL_DISPATCH, extracts user_id and params,
        calls the corresponding MemoryStore method, and returns the result.
        """
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Validate tool exists.
        if tool_name not in TOOL_DISPATCH:
            return _error_response(
                req_id,
                code=-32601,
                message=f"Unknown tool: {tool_name}",
            )

        dispatch = TOOL_DISPATCH[tool_name]
        method_name = dispatch["method"]
        required_params = dispatch["required_params"]
        optional_params = dispatch.get("optional_params", {})

        # Extract user_id (optional, with default).
        user_id = arguments.get("user_id", dispatch["user_id_default"])

        # Validate required params.
        missing = [p for p in required_params if p not in arguments]
        if missing:
            return _error_response(
                req_id,
                code=-32602,  # Invalid params
                message=f"Missing required parameters: {', '.join(missing)}",
            )

        # Build the kwargs for the store method.
        kwargs = {}
        for param in required_params:
            kwargs[param] = arguments[param]
        for param, default in optional_params.items():
            kwargs[param] = arguments.get(param, default)

        # Call the store method and format the result.
        try:
            method: Callable = getattr(self.store, method_name)
            result = method(user_id, **kwargs)
            return self._format_result(req_id, tool_name, result)
        except FactNotFoundError as e:
            return _error_response(req_id, code=-32000, message=str(e))
        except Exception as e:
            return _error_response(
                req_id,
                code=-32603,  # Internal error
                message=f"Error executing {tool_name}: {e}",
                data={"traceback": traceback.format_exc()},
            )

    # ── Result Formatting ────────────────────────────────────────────────

    def _format_result(self, req_id: Any, tool_name: str, result: Any) -> dict:
        """
        Format a MemoryStore result into an MCP-compatible response.

        Different tools return different types (dict, list, bool, int),
        so we wrap them into a consistent JSON text content format.
        """
        return _success_response(req_id, {
            "content": _json_text({"tool": tool_name, "result": result}),
        })

    # ── Stdio Event Loop ─────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main event loop: read JSON-RPC from stdin, write responses to stdout.

        This is the stdio transport pattern used by all MCP servers:
          1. Read one line from stdin.
          2. Parse it as JSON.
          3. Process the request.
          4. Write one line of JSON to stdout.
          5. Flush stdout so the client gets it immediately.
        Stderr captures internal errors without corrupting the protocol stream.
        """
        while True:
            line = sys.stdin.readline()
            if not line:
                # EOF — stdin closed, shut down cleanly.
                break

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                # Invalid JSON from the client — return a parse error.
                error_resp = _error_response(
                    None,
                    code=-32700,  # Parse error
                    message="Invalid JSON",
                    data=str(e),
                )
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                # Unexpected error in the handler itself.
                error_resp = _error_response(
                    None,
                    code=-32603,  # Internal error
                    message="Server error",
                    data={"error": str(e), "traceback": traceback.format_exc()},
                )
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()


# ── Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: parse CLI args and run the MCP server.

    Semantic search is auto-detected: if sentence-transformers is installed,
    it loads automatically with --backend sqlite. No flag needed.

    Usage:
        ctrl-memory-mcp                               # JSON backend
        ctrl-memory-mcp --backend sqlite               # SQLite + keyword search
                                                       # (semantic if sentence-transformers installed)
    """
    import argparse
    parser = argparse.ArgumentParser(description="Ctrl-Memory MCP server")
    parser.add_argument(
        "--backend", choices=["json", "sqlite"], default="json",
        help="Storage backend: 'json' (default) or 'sqlite'",
    )
    parser.add_argument(
        "--db-path",
        help="SQLite database path (default: ~/.ctrl-memory/memory.db)",
    )
    parser.add_argument(
        "--storage-dir",
        help="JSON storage directory (default: ~/.ctrl-memory)",
    )
    args = parser.parse_args()
    server = MCPMemoryServer(
        backend=args.backend,
        storage_dir=args.storage_dir,
        db_path=args.db_path,
    )
    server.run()


if __name__ == "__main__":
    main()
