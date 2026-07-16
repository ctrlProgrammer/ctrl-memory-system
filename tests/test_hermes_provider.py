"""
Tests for the Ctrl-Memory Hermes Memory Provider Plugin.

Since the plugin depends on `agent.memory_provider` (only available inside
Hermes), we mock the MemoryProvider ABC and test the plugin logic in
isolation.

Run with:
    python -m unittest tests.test_hermes_provider -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Mock the Hermes MemoryProvider ABC ───────────────────────────────────

class MockMemoryProvider:
    """Minimal stand-in for Hermes' agent.memory_provider.MemoryProvider."""

    def __init__(self):
        self._store = None

    @property
    def name(self):
        return "ctrl-memory"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def system_prompt_block(self):
        return ""

    def prefetch(self, query, *, session_id=""):
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"error": "not implemented"})

    def shutdown(self):
        pass


# Patch agent.memory_provider so the plugin can import CtrlMemoryProvider.
# We scope the patch to setUpModule/tearDownModule so it doesn't leak into
# other test files that may be run in the same process.
_orig_modules: dict = {}


def setUpModule():
    global _orig_modules
    if "agent.memory_provider" in __import__("sys").modules:
        _orig_modules["agent.memory_provider"] = __import__("sys").modules["agent.memory_provider"]
    mock_mod = MagicMock()
    mock_mod.MemoryProvider = MockMemoryProvider
    __import__("sys").modules["agent.memory_provider"] = mock_mod


def tearDownModule():
    if "agent.memory_provider" in _orig_modules:
        __import__("sys").modules["agent.memory_provider"] = _orig_modules["agent.memory_provider"]
    else:
        __import__("sys").modules.pop("agent.memory_provider", None)


# Now we can import the plugin.
from hermes_provider import CtrlMemoryProvider, ADD_MEMORY_SCHEMA, SEARCH_MEMORY_SCHEMA


class TestCtrlMemoryProviderInit(unittest.TestCase):
    """Tests for provider initialisation and lifecycle."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.provider = CtrlMemoryProvider()
        # Patch the hermes_home path to use our temp dir.
        self.provider.initialize(
            session_id="test-session",
            hermes_home=self.tmp_dir,
            user_id="test_user",
        )

    def tearDown(self):
        self.provider.shutdown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_provider_name(self):
        """Provider name is 'ctrl-memory'."""
        self.assertEqual(self.provider.name, "ctrl-memory")

    def test_is_available(self):
        """Provider is always available."""
        self.assertTrue(self.provider.is_available())

    def test_initialise_creates_store(self):
        """After initialize, the store should exist and be usable."""
        self.assertIsNotNone(self.provider._store)
        # The DB file should be created.
        db_path = Path(self.tmp_dir) / "ctrl-memory" / "memory.db"
        self.assertTrue(db_path.exists())

    def test_system_prompt_block_returns_string(self):
        """system_prompt_block returns a non-empty string after init."""
        block = self.provider.system_prompt_block()
        self.assertIsInstance(block, str)
        self.assertIn("0 facts stored", block)

    def test_system_prompt_block_updates_count(self):
        """After adding facts, the system prompt reflects the count."""
        self.provider._store.add_fact("test_user", "A fact")
        block = self.provider.system_prompt_block()
        self.assertIn("1 facts stored", block)


class TestCtrlMemoryToolSchemas(unittest.TestCase):
    """Tests for tool schema definitions."""

    def setUp(self):
        self.provider = CtrlMemoryProvider()

    def test_get_tool_schemas_returns_list(self):
        """get_tool_schemas returns a non-empty list."""
        schemas = self.provider.get_tool_schemas()
        self.assertIsInstance(schemas, list)
        self.assertGreater(len(schemas), 0)

    def test_tool_schemas_have_valid_structure(self):
        """Each tool schema has name, description, and parameters."""
        schemas = self.provider.get_tool_schemas()
        for schema in schemas:
            with self.subTest(tool=schema.get("name")):
                self.assertIn("name", schema)
                self.assertIn("description", schema)
                self.assertIn("parameters", schema)
                self.assertEqual(schema["parameters"]["type"], "object")

    def test_expected_tools_present(self):
        """The expected tool names are all present."""
        schemas = self.provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        for expected in ["ctrl_memory_add", "ctrl_memory_search",
                         "ctrl_memory_update", "ctrl_memory_delete",
                         "ctrl_memory_status"]:
            with self.subTest(tool=expected):
                self.assertIn(expected, names)


class TestCtrlMemoryToolCalls(unittest.TestCase):
    """Tests for tool call dispatch."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.provider = CtrlMemoryProvider()
        self.provider.initialize(
            session_id="test-session",
            hermes_home=self.tmp_dir,
            user_id="tool_user",
        )

    def tearDown(self):
        self.provider.shutdown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_tool_stores_fact(self):
        """ctrl_memory_add stores a fact and returns its ID."""
        result = self.provider.handle_tool_call(
            "ctrl_memory_add",
            {"content": "A test fact", "tags": "test"},
        )
        data = json.loads(result)
        self.assertIn("fact_id", data)
        self.assertEqual(data["status"], "stored")

    def test_search_tool_returns_results(self):
        """ctrl_memory_search finds stored facts."""
        self.provider._store.add_fact("tool_user", "Searchable content")
        result = self.provider.handle_tool_call(
            "ctrl_memory_search",
            {"query": "Searchable", "limit": 5},
        )
        data = json.loads(result)
        self.assertGreater(data["count"], 0)
        self.assertIn("Searchable", data["results"][0]["content"])

    def test_update_tool_updates_fact(self):
        """ctrl_memory_update modifies an existing fact."""
        fact = self.provider._store.add_fact("tool_user", "Original text", tags="old")
        result = self.provider.handle_tool_call(
            "ctrl_memory_update",
            {"fact_id": fact["id"], "content": "Updated text", "tags": "new"},
        )
        data = json.loads(result)
        self.assertEqual(data["status"], "updated")
        self.assertEqual(data["fact_id"], fact["id"])
        # Verify the update persisted.
        updated = self.provider._store.get_fact("tool_user", fact["id"])
        self.assertEqual(updated["content"], "Updated text")
        self.assertEqual(updated["tags"], "new")

    def test_delete_tool_removes_fact(self):
        """ctrl_memory_delete removes a fact by ID."""
        fact = self.provider._store.add_fact("tool_user", "To delete")
        result = self.provider.handle_tool_call(
            "ctrl_memory_delete",
            {"fact_id": fact["id"]},
        )
        data = json.loads(result)
        self.assertTrue(data["deleted"])

    def test_status_tool_returns_counts(self):
        """ctrl_memory_status returns fact count and backend info."""
        self.provider._store.add_fact("tool_user", "A fact")
        result = self.provider.handle_tool_call(
            "ctrl_memory_status",
            {},
        )
        data = json.loads(result)
        self.assertEqual(data["fact_count"], 1)
        self.assertEqual(data["backend"], "sqlite")
        self.assertIn("semantic_search", data)

    def test_unknown_tool_returns_error(self):
        """An unknown tool name returns an error JSON."""
        result = self.provider.handle_tool_call(
            "nonexistent_tool",
            {},
        )
        data = json.loads(result)
        self.assertIn("error", data)


class TestCtrlMemoryPrefetch(unittest.TestCase):
    """Tests for the automatic prefetch lifecycle hook."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.provider = CtrlMemoryProvider()
        self.provider.initialize(
            session_id="test-session",
            hermes_home=self.tmp_dir,
            user_id="prefetch_user",
        )
        # Seed some facts.
        self.provider._store.add_fact("prefetch_user", "Database is PostgreSQL")
        self.provider._store.add_fact("prefetch_user", "User prefers VS Code")
        self.provider._store.add_fact("prefetch_user", "Server is in us-west-2")

    def tearDown(self):
        self.provider.shutdown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_prefetch_returns_relevant_facts(self):
        """prefetch returns context string with matching facts."""
        context = self.provider.prefetch("database", session_id="test-session")
        self.assertIsInstance(context, str)
        self.assertIn("PostgreSQL", context)

    def test_prefetch_empty_query_returns_empty(self):
        """prefetch with empty query returns empty string."""
        context = self.provider.prefetch("", session_id="test-session")
        self.assertEqual(context, "")

    def test_prefetch_no_match_returns_empty(self):
        """prefetch with no matching facts returns empty string."""
        context = self.provider.prefetch("nonexistent_topic_xyz", session_id="test-session")
        self.assertEqual(context, "")

    def test_prefetch_includes_relevant_memories_header(self):
        """Returned context starts with the header marker."""
        context = self.provider.prefetch("VS Code", session_id="test-session")
        self.assertTrue(
            context.startswith("## Relevant Memories"),
            msg=f"Expected header, got: {context[:50]}",
        )


class TestCtrlMemorySyncTurn(unittest.TestCase):
    """Tests for the auto-capture lifecycle hook."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.provider = CtrlMemoryProvider()
        self.provider.initialize(
            session_id="test-session",
            hermes_home=self.tmp_dir,
            user_id="sync_user",
        )

    def tearDown(self):
        self.provider.shutdown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_sync_turn_captures_first_person_statements(self):
        """Messages with 'I' or 'my' are auto-captured."""
        self.provider.sync_turn(
            user_content="I prefer dark mode in my editor",
            assistant_content="Got it!",
            session_id="test-session",
        )
        facts = self.provider._store.get_all_facts("sync_user")
        self.assertGreater(len(facts), 0)
        self.assertIn("dark mode", facts[0]["content"])

    def test_sync_turn_ignores_non_first_person(self):
        """Messages without first-person markers are NOT auto-captured."""
        self.provider.sync_turn(
            user_content="What is the capital of France?",
            assistant_content="Paris",
            session_id="test-session",
        )
        facts = self.provider._store.get_all_facts("sync_user")
        self.assertEqual(len(facts), 0)

    def test_sync_turn_tags_auto_captured_facts(self):
        """Auto-captured facts get the 'auto-captured' tag."""
        self.provider.sync_turn(
            user_content="My project uses FastAPI",
            assistant_content="Noted!",
            session_id="test-session",
        )
        facts = self.provider._store.get_all_facts("sync_user")
        self.assertIn("auto-captured", facts[0]["tags"])

    def test_sync_turn_empty_user_content_does_nothing(self):
        """Empty user content doesn't cause errors or store anything."""
        self.provider.sync_turn(
            user_content="",
            assistant_content="Hello",
            session_id="test-session",
        )
        self.assertEqual(self.provider._store.count_facts("sync_user"), 0)

    def test_sync_turn_deduplicates_near_duplicates(self):
        """Rapid repeated auto-captures of the same content are deduplicated."""
        self.provider.sync_turn(
            user_content="I like Python",
            assistant_content="Cool!",
            session_id="test-session",
        )
        self.provider.sync_turn(
            user_content="I like Python",
            assistant_content="Cool!",
            session_id="test-session",
        )
        facts = self.provider._store.get_all_facts("sync_user")
        self.assertEqual(len(facts), 1, "Should not store duplicate auto-captures")


if __name__ == "__main__":
    unittest.main()
