"""
Tests for the MemoryStore backend — Phase 1 of the MCP memory system.

These tests don't need the MCP server at all. They exercise the core storage
logic: CRUD, search, cross-session persistence, and noise isolation.

Run with:
    python -m pytest tests/test_memory_backend.py -v
    # or
    python -m unittest tests.test_memory_backend
"""

import os
import tempfile
import unittest

from memory_backend import MemoryStore, FactNotFoundError


class TestMemoryStore(unittest.TestCase):
    """Unit tests for the JSON-file storage backend."""

    def setUp(self):
        """
        Create a fresh MemoryStore in a temp directory for each test.
        This keeps tests isolated from real user data and from each other.
        """
        self.tmp_dir = tempfile.mkdtemp()
        self.store = MemoryStore(storage_dir=self.tmp_dir)
        self.user = "test_user"

    def tearDown(self):
        """Clean up the temp directory after each test."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── Basic CRUD ───────────────────────────────────────────────────

    def test_add_and_get_fact(self):
        """Adding a fact returns it and get_fact retrieves it by ID."""
        fact = self.store.add_fact(self.user, "User prefers dark mode", tags="preference")
        self.assertIn("id", fact)
        self.assertEqual(fact["content"], "User prefers dark mode")
        self.assertEqual(fact["tags"], "preference")

        # Retrieve by ID.
        retrieved = self.store.get_fact(self.user, fact["id"])
        self.assertEqual(retrieved["content"], "User prefers dark mode")

    def test_add_fact_auto_increments_id(self):
        """Each new fact gets a unique, incrementing ID."""
        f1 = self.store.add_fact(self.user, "Fact one")
        f2 = self.store.add_fact(self.user, "Fact two")
        f3 = self.store.add_fact(self.user, "Fact three")
        self.assertEqual(f1["id"], 1)
        self.assertEqual(f2["id"], 2)
        self.assertEqual(f3["id"], 3)

    def test_get_fact_not_found_raises_error(self):
        """get_fact raises FactNotFoundError for a nonexistent ID."""
        with self.assertRaises(FactNotFoundError):
            self.store.get_fact(self.user, 999)

    def test_get_all_facts_returns_all(self):
        """get_all_facts returns every stored fact for the user."""
        self.store.add_fact(self.user, "Fact A")
        self.store.add_fact(self.user, "Fact B")
        self.store.add_fact(self.user, "Fact C")
        all_facts = self.store.get_all_facts(self.user)
        self.assertEqual(len(all_facts), 3)

    def test_count_facts(self):
        """count_facts returns the correct number of stored facts."""
        self.assertEqual(self.store.count_facts(self.user), 0)
        self.store.add_fact(self.user, "One")
        self.assertEqual(self.store.count_facts(self.user), 1)
        self.store.add_fact(self.user, "Two")
        self.assertEqual(self.store.count_facts(self.user), 2)

    # ── Update ───────────────────────────────────────────────────────

    def test_update_fact_content(self):
        """update_fact changes the content and updates the timestamp."""
        fact = self.store.add_fact(self.user, "Original content", tags="old")
        updated = self.store.update_fact(
            self.user, fact["id"], content="Updated content", tags="new"
        )
        self.assertEqual(updated["content"], "Updated content")
        self.assertEqual(updated["tags"], "new")
        # Timestamp should be newer.
        self.assertGreater(updated["updated_at"], fact["created_at"])

    def test_update_fact_partial(self):
        """update_fact with only content leaves tags unchanged."""
        fact = self.store.add_fact(self.user, "Original", tags="keep-me")
        updated = self.store.update_fact(self.user, fact["id"], content="Changed")
        self.assertEqual(updated["content"], "Changed")
        self.assertEqual(updated["tags"], "keep-me")

    def test_update_nonexistent_fact_raises_error(self):
        """update_fact on a nonexistent ID raises FactNotFoundError."""
        with self.assertRaises(FactNotFoundError):
            self.store.update_fact(self.user, 999, content="nope")

    # ── Delete ───────────────────────────────────────────────────────

    def test_delete_fact_removes_it(self):
        """delete_fact removes exactly the specified fact."""
        f1 = self.store.add_fact(self.user, "Fact one")
        self.store.add_fact(self.user, "Fact two")
        deleted = self.store.delete_fact(self.user, f1["id"])
        self.assertTrue(deleted)
        remaining = self.store.get_all_facts(self.user)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["content"], "Fact two")

    def test_delete_nonexistent_fact_returns_false(self):
        """delete_fact returns False when the fact doesn't exist."""
        result = self.store.delete_fact(self.user, 999)
        self.assertFalse(result)

    def test_clear_all_removes_everything(self):
        """clear_all removes the JSON file entirely — no facts remain."""
        self.store.add_fact(self.user, "Something")
        self.store.add_fact(self.user, "Something else")
        self.store.clear_all(self.user)
        self.assertEqual(self.store.count_facts(self.user), 0)

    # ── Search ───────────────────────────────────────────────────────

    def test_search_by_content_substring(self):
        """search_facts finds facts whose content contains the query."""
        self.store.add_fact(self.user, "User prefers dark mode", tags="preference")
        results = self.store.search_facts(self.user, "dark")
        self.assertEqual(len(results), 1)
        self.assertIn("dark mode", results[0]["content"])

    def test_search_by_tags(self):
        """search_facts also matches against the tags field."""
        self.store.add_fact(self.user, "Deployment in us-west-2", tags="project,infra")
        results = self.store.search_facts(self.user, "infra")
        self.assertEqual(len(results), 1)

    def test_search_is_case_insensitive(self):
        """search_facts matches regardless of casing."""
        self.store.add_fact(self.user, "Project uses FastAPI")
        results = self.store.search_facts(self.user, "fastapi")
        self.assertEqual(len(results), 1)

    def test_search_no_match_returns_empty(self):
        """search_facts returns empty list when nothing matches."""
        self.store.add_fact(self.user, "Something irrelevant")
        results = self.store.search_facts(self.user, "nonexistent")
        self.assertEqual(len(results), 0)

    def test_search_empty_query_returns_empty(self):
        """search_facts with empty string returns no results."""
        self.store.add_fact(self.user, "Anything")
        results = self.store.search_facts(self.user, "")
        self.assertEqual(len(results), 0)

    def test_search_whitespace_query_returns_empty(self):
        """search_facts with whitespace-only query returns no results."""
        results = self.store.search_facts(self.user, "   ")
        self.assertEqual(len(results), 0)

    def test_search_respects_limit(self):
        """search_facts returns no more than `limit` results."""
        for i in range(10):
            self.store.add_fact(self.user, f"Common fact number {i}")
        results = self.store.search_facts(self.user, "Common", limit=3)
        self.assertEqual(len(results), 3)

    def test_search_returns_newest_first(self):
        """search_facts sorts results by updated_at descending."""
        self.store.add_fact(self.user, "Old fact", tags="common")
        import time
        time.sleep(0.01)
        self.store.add_fact(self.user, "New fact", tags="common")
        results = self.store.search_facts(self.user, "common")
        self.assertEqual(results[0]["content"], "New fact")

    # ── Cross-Session Persistence ────────────────────────────────────

    def test_cross_session_persistence(self):
        """
        Facts survive across store instances (simulating separate sessions).
        Both instances point at the same storage directory.
        """
        store_a = MemoryStore(storage_dir=self.tmp_dir)
        store_a.add_fact(self.user, "Stored in session 1")

        store_b = MemoryStore(storage_dir=self.tmp_dir)
        facts = store_b.get_all_facts(self.user)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["content"], "Stored in session 1")

    # ── Noise Isolation ──────────────────────────────────────────────

    def test_noise_isolation(self):
        """
        Irrelevant facts don't appear in search results for a specific query.
        This tests precision: returning only what's asked for.
        """
        # Seed relevant facts.
        self.store.add_fact(self.user, "Database is PostgreSQL 16")
        self.store.add_fact(self.user, "Server runs on us-west-2")

        # Seed noise (off-topic facts).
        self.store.add_fact(self.user, "Weather is nice today")
        self.store.add_fact(self.user, "Cats are adorable")

        # Query for database.
        results = self.store.search_facts(self.user, "database")
        self.assertTrue(any("PostgreSQL" in f["content"] for f in results))
        self.assertFalse(any("Cats" in f["content"] for f in results))
        self.assertFalse(any("Weather" in f["content"] for f in results))

    def test_user_isolation(self):
        """Facts for different users are completely isolated."""
        self.store.add_fact("alice", "Alice's secret project", tags="secret")
        self.store.add_fact("bob", "Bob's todo list", tags="task")

        alice_facts = self.store.get_all_facts("alice")
        bob_facts = self.store.get_all_facts("bob")

        self.assertEqual(len(alice_facts), 1)
        self.assertEqual(len(bob_facts), 1)
        self.assertIn("Alice", alice_facts[0]["content"])
        self.assertIn("Bob", bob_facts[0]["content"])

    # ── Default Storage Dir ──────────────────────────────────────────

    def test_default_storage_dir_is_created(self):
        """
        MemoryStore with default args creates ~/.ctrl-memory/ if needed.
        We test this indirectly by checking that a store with no args works.
        """
        # Just ensure the class can be instantiated without errors.
        store = MemoryStore()
        self.assertIsNotNone(store)
        # Clean up the directory we just created.
        import shutil
        shutil.rmtree(store._dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
