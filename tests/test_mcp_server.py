"""
Integration tests for the MCP server — covers both JSON and SQLite backends.

Spawns mcp_server.py as a subprocess over stdio and communicates via
JSON-RPC 2.0 messages, exactly like a real MCP client would.

Run with:
    python -m unittest tests.test_mcp_server -v
"""

import json
import os
import subprocess
import tempfile
import time
import unittest


def _find_server_script() -> str:
    """Find mcp_server.py relative to this test file."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "mcp_server.py",
    )


class _BaseMCPTest:
    """
    Mixin for MCP integration tests.

    Subclasses override BACKEND to test different storage backends.
    Must also inherit unittest.TestCase.
    """

    BACKEND: str = "json"  # Override in subclasses.

    @classmethod
    def setUpClass(cls):
        """Start the MCP server with the configured backend."""
        cls.tmp_dir = tempfile.mkdtemp()
        cls.tmp_db = os.path.join(cls.tmp_dir, "test.db")
        server_script = _find_server_script()

        cmd = ["python3", server_script, "--backend", cls.BACKEND]
        if cls.BACKEND == "sqlite":
            cmd += ["--db-path", cls.tmp_db]
        else:
            cmd += ["--storage-dir", cls.tmp_dir]

        cls.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(server_script),
        )
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        """Shut down the server and clean up temp files."""
        if cls.proc.poll() is None:
            cls.proc.terminate()
            cls.proc.wait(timeout=5)
        import shutil
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _send(self, request: dict) -> dict:
        """Send a JSON-RPC request and return the response."""
        if self.proc.poll() is not None:
            stderr_output = self.proc.stderr.read()
            self.fail(
                f"MCP server process died. "
                f"Return code: {self.proc.returncode}. "
                f"Stderr: {stderr_output}"
            )

        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        response_line = self.proc.stdout.readline()
        if not response_line:
            stderr_output = self.proc.stderr.read()
            self.fail(
                f"MCP server returned empty response. "
                f"Stderr: {stderr_output}"
            )

        return json.loads(response_line)

    # ── Tests ────────────────────────────────────────────────────────────

    def test_initialize(self):
        """The server responds to the MCP initialize handshake."""
        response = self._send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
        })
        self.assertIn("result", response)
        self.assertEqual(response["result"]["protocolVersion"], "0.1.0")
        self.assertEqual(response["result"]["serverInfo"]["name"], "ctrl-memory")

    def test_tools_list(self):
        """The server returns a list of tool definitions."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        response = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        self.assertIn("result", response)
        tools = response["result"]["tools"]
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

        tool_names = [t["name"] for t in tools]
        for expected in ["add_memory", "search_memory", "get_memory",
                         "list_memories", "update_memory", "delete_memory",
                         "count_memories"]:
            with self.subTest(tool=expected):
                self.assertIn(expected, tool_names)

        for tool in tools:
            with self.subTest(tool=tool["name"]):
                self.assertIn("inputSchema", tool)
                self.assertEqual(tool["inputSchema"]["type"], "object")

    def test_add_and_search_memory(self):
        """Add a fact, then search for it via the MCP interface."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

        add_resp = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "add_memory",
                "arguments": {
                    "user_id": "mcp_test",
                    "content": "Integration test fact: prefers VS Code",
                    "tags": "test,editor",
                },
            },
        })
        self.assertIn("result", add_resp)
        add_data = json.loads(add_resp["result"]["content"][0]["text"])
        self.assertEqual(add_data["tool"], "add_memory")
        self.assertIn("id", add_data["result"])

        search_resp = self._send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "search_memory",
                "arguments": {"user_id": "mcp_test", "query": "VS Code"},
            },
        })
        self.assertIn("result", search_resp)
        search_data = json.loads(search_resp["result"]["content"][0]["text"])
        self.assertEqual(search_data["tool"], "search_memory")
        self.assertGreaterEqual(len(search_data["result"]), 1)
        self.assertIn("VS Code", search_data["result"][0]["content"])

    def test_list_memories(self):
        """list_memories returns all facts for a user."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self._send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": "add_memory",
                                "arguments": {"user_id": "list_user", "content": "Fact one"}}})
        self._send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "add_memory",
                                "arguments": {"user_id": "list_user", "content": "Fact two"}}})

        list_resp = self._send({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "list_memories",
                       "arguments": {"user_id": "list_user"}},
        })
        list_data = json.loads(list_resp["result"]["content"][0]["text"])
        self.assertEqual(len(list_data["result"]), 2)

    def test_count_memories(self):
        """count_memories returns the correct count."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self._send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": "add_memory",
                                "arguments": {"user_id": "cnt_user", "content": "Only fact"}}})
        count_resp = self._send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "count_memories",
                       "arguments": {"user_id": "cnt_user"}},
        })
        count_data = json.loads(count_resp["result"]["content"][0]["text"])
        self.assertEqual(count_data["result"], 1)

    def test_delete_memory(self):
        """Add a fact, delete it, then confirm it's gone."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

        add_resp = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "add_memory",
                       "arguments": {"user_id": "del_user", "content": "To be deleted"}},
        })
        fact_id = json.loads(add_resp["result"]["content"][0]["text"])["result"]["id"]

        del_resp = self._send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "delete_memory",
                       "arguments": {"user_id": "del_user", "fact_id": fact_id}},
        })
        self.assertTrue(json.loads(del_resp["result"]["content"][0]["text"])["result"])

        count_resp = self._send({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "count_memories",
                       "arguments": {"user_id": "del_user"}},
        })
        self.assertEqual(
            json.loads(count_resp["result"]["content"][0]["text"])["result"], 0
        )

    def test_update_memory(self):
        """Update a fact's content, then verify the change."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

        add_resp = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "add_memory",
                       "arguments": {"user_id": "upd_user", "content": "Old content", "tags": "old"}},
        })
        fact_id = json.loads(add_resp["result"]["content"][0]["text"])["result"]["id"]

        upd_resp = self._send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "update_memory",
                       "arguments": {"user_id": "upd_user", "fact_id": fact_id,
                                     "content": "New content", "tags": "updated"}},
        })
        upd_data = json.loads(upd_resp["result"]["content"][0]["text"])
        self.assertEqual(upd_data["result"]["content"], "New content")
        self.assertEqual(upd_data["result"]["tags"], "updated")

    def test_get_memory(self):
        """get_memory retrieves a single fact by ID."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

        add_resp = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "add_memory",
                       "arguments": {"user_id": "get_user", "content": "Target fact"}},
        })
        fact_id = json.loads(add_resp["result"]["content"][0]["text"])["result"]["id"]

        get_resp = self._send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "get_memory",
                       "arguments": {"user_id": "get_user", "fact_id": fact_id}},
        })
        get_data = json.loads(get_resp["result"]["content"][0]["text"])
        self.assertEqual(get_data["result"]["content"], "Target fact")

    def test_error_unknown_method(self):
        """Unknown methods return a JSON-RPC method-not-found error."""
        response = self._send({
            "jsonrpc": "2.0", "id": 1, "method": "nonexistent_method",
        })
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601)

    def test_error_unknown_tool(self):
        """Unknown tool names return a JSON-RPC method-not-found error."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        response = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601)

    def test_error_missing_required_params(self):
        """Calling a tool without required params returns invalid-params error."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        response = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "add_memory",
                       "arguments": {"user_id": "test"}},
        })
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32602)
        self.assertIn("content", response["error"]["message"])

    def test_default_user_id(self):
        """Tools work without specifying user_id (defaults to 'default')."""
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        response = self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "count_memories", "arguments": {}},
        })
        self.assertIn("result", response)


# ── Backend-specific test classes ────────────────────────────────────────

class TestMCPMemoryServerJSON(_BaseMCPTest, unittest.TestCase):
    """MCP integration tests with the JSON backend."""
    BACKEND = "json"


class TestMCPMemoryServerSQLite(_BaseMCPTest, unittest.TestCase):
    """MCP integration tests with the SQLite backend."""
    BACKEND = "sqlite"


if __name__ == "__main__":
    unittest.main()
