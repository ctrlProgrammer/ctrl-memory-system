"""
mcp_server.py — Thin CLI wrapper for ctrl-memory MCP server.

This module re-exports the canonical MCPMemoryServer from ctrl_memory.server.
It exists for backward compatibility: the integration tests spawn this file
as a subprocess, and the install.sh symlink may reference it.

Usage:
    python3 mcp_server.py [--backend json|sqlite] [--db-path PATH] [--storage-dir DIR]
"""
from ctrl_memory.server import main

if __name__ == "__main__":
    main()
