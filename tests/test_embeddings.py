"""
Tests for the EmbeddingEngine and semantic search — Phase 3 of the memory system.

Covers:
  - EmbeddingEngine class (fallback when sentence-transformers is missing)
  - _pack_embedding / _unpack_embedding serialisation round-trip
  - cosine_similarity computation
  - SQLiteStore auto-embedding on add_fact
  - SQLiteStore search_facts_semantic
  - Re-embedding on update_fact
  - Embedding cleanup on delete_fact and clear_all

Run with:
    python -m unittest tests.test_embeddings -v
"""

import os
import struct
import tempfile
import unittest

from ctrl_memory.backend import (
    EmbeddingEngine,
    SQLiteStore,
    _pack_embedding,
    _unpack_embedding,
)


class TestEmbeddingSerialisation(unittest.TestCase):
    """Tests for the float-vector <-> binary blob helpers."""

    def test_pack_unpack_round_trip(self):
        """Packing then unpacking returns the same vector."""
        original = [0.1, 0.2, 0.3, -0.4, 0.0, 0.99]
        blob = _pack_embedding(original)
        restored = _unpack_embedding(blob)
        self.assertEqual(len(restored), len(original))
        for a, b in zip(original, restored):
            self.assertAlmostEqual(a, b, places=6)

    def test_empty_vector(self):
        """An empty vector packs to an empty bytes object."""
        blob = _pack_embedding([])
        self.assertEqual(blob, b"")
        self.assertEqual(_unpack_embedding(blob), [])

    def test_single_element(self):
        """A single-element vector round-trips correctly."""
        blob = _pack_embedding([3.14159])
        restored = _unpack_embedding(blob)
        self.assertAlmostEqual(restored[0], 3.14159, places=5)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for the static cosine_similarity method."""

    def test_identical_vectors(self):
        """Cosine similarity of a vector with itself is 1.0."""
        vec = [1.0, 0.0, 0.0]
        score = EmbeddingEngine.cosine_similarity(vec, vec)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        score = EmbeddingEngine.cosine_similarity(a, b)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors is -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        score = EmbeddingEngine.cosine_similarity(a, b)
        self.assertAlmostEqual(score, -1.0, places=6)

    def test_partial_similarity(self):
        """Vectors with some shared direction produce a score between 0 and 1."""
        a = [1.0, 0.0]
        b = [0.5, 0.5]
        score = EmbeddingEngine.cosine_similarity(a, b)
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)

    def test_zero_vector_returns_zero(self):
        """Cosine similarity with a zero vector returns 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 0.0]
        score = EmbeddingEngine.cosine_similarity(a, b)
        self.assertEqual(score, 0.0)


class TestEmbeddingEngine(unittest.TestCase):
    """Tests for the EmbeddingEngine."""

    def test_engine_creates_without_crash(self):
        """EmbeddingEngine can be instantiated even without sentence-transformers."""
        ee = EmbeddingEngine()
        self.assertIsNotNone(ee)

    def test_is_available_may_be_false(self):
        """If sentence-transformers isn't installed, is_available is False."""
        ee = EmbeddingEngine()
        # The test environment may or may not have it — just check it's a bool.
        self.assertIsInstance(ee.is_available, bool)

    def test_embed_raises_if_not_available(self):
        """Calling embed() when model isn't loaded raises RuntimeError."""
        ee = EmbeddingEngine()
        if not ee.is_available:
            with self.assertRaises(RuntimeError):
                ee.embed("test")
        else:
            vec = ee.embed("test")
            self.assertEqual(len(vec), ee.dimension)


class TestSQLiteStoreAutoEmbedding(unittest.TestCase):
    """
    Tests for automatic embedding in SQLiteStore.

    These tests run whether or not sentence-transformers is installed,
    but the semantic search tests only work when it IS installed.
    Use setUp to detect availability.
    """

    def setUp(self):
        self.tmp_file = tempfile.mktemp(suffix=".db")
        self.ee = EmbeddingEngine()
        # Create store WITH embedding engine.
        self.store = SQLiteStore(
            db_path=self.tmp_file,
            embedding_engine=self.ee if self.ee.is_available else None,
        )
        self.user = "sem_test"

    def tearDown(self):
        self.store._conn.close()
        for ext in ("", "-wal", "-shm"):
            p = self.tmp_file + ext
            if os.path.exists(p):
                os.unlink(p)

    def test_add_fact_does_not_crash_with_engine(self):
        """Adding a fact with an EmbeddingEngine configured doesn't error."""
        fact = self.store.add_fact(self.user, "Test fact", tags="test")
        self.assertIn("id", fact)

    def test_embedding_table_gets_populated_when_available(self):
        """When engine is available, embeddings table gets a row on add."""
        if not self.ee.is_available:
            self.skipTest("sentence-transformers not installed")

        self.store.add_fact(self.user, "Something to remember")
        row = self.store._conn.execute(
            "SELECT COUNT(*) AS cnt FROM embeddings"
        ).fetchone()
        self.assertGreater(row["cnt"], 0)

    def test_delete_fact_cleans_up_embedding(self):
        """Deleting a fact also removes its embedding row."""
        if not self.ee.is_available:
            self.skipTest("sentence-transformers not installed")

        fact = self.store.add_fact(self.user, "Delete me")
        self.store.delete_fact(self.user, fact["id"])
        row = self.store._conn.execute(
            "SELECT COUNT(*) AS cnt FROM embeddings"
        ).fetchone()
        self.assertEqual(row["cnt"], 0)

    def test_clear_all_cleans_up_embeddings(self):
        """Clearing all facts also removes all embedding rows."""
        if not self.ee.is_available:
            self.skipTest("sentence-transformers not installed")

        self.store.add_fact(self.user, "Fact A")
        self.store.add_fact(self.user, "Fact B")
        self.store.clear_all(self.user)
        row = self.store._conn.execute(
            "SELECT COUNT(*) AS cnt FROM embeddings"
        ).fetchone()
        self.assertEqual(row["cnt"], 0)

    def test_update_fact_content_reembeds(self):
        """Updating content recomputes (replaces) the embedding."""
        if not self.ee.is_available:
            self.skipTest("sentence-transformers not installed")

        fact = self.store.add_fact(self.user, "Original content")
        # Capture the original embedding blob.
        orig_blob = self.store._conn.execute(
            "SELECT vector FROM embeddings WHERE fact_id = ?", (fact["id"],)
        ).fetchone()["vector"]

        self.store.update_fact(self.user, fact["id"], content="Completely different content")
        new_blob = self.store._conn.execute(
            "SELECT vector FROM embeddings WHERE fact_id = ?", (fact["id"],)
        ).fetchone()["vector"]

        self.assertNotEqual(orig_blob, new_blob)


class TestSQLiteStoreSemanticSearch(unittest.TestCase):
    """
    Semantic search tests — require sentence-transformers to be installed.
    """

    def setUp(self):
        self.tmp_file = tempfile.mktemp(suffix=".db")
        self.ee = EmbeddingEngine()
        if not self.ee.is_available:
            self.skipTest("sentence-transformers not installed")

        self.store = SQLiteStore(db_path=self.tmp_file, embedding_engine=self.ee)
        self.user = "sem_test"

        # Seed facts with distinct topics.
        self.store.add_fact(self.user, "User prefers dark mode in all editors")
        self.store.add_fact(self.user, "Database is PostgreSQL 16 running on us-west-2")
        self.store.add_fact(self.user, "Project uses FastAPI with Pydantic v2")
        self.store.add_fact(self.user, "Deployment rollback failed due to stale migration")
        self.store.add_fact(self.user, "Weather is nice today")  # drift / noise

    def tearDown(self):
        self.store._conn.close()
        for ext in ("", "-wal", "-shm"):
            p = self.tmp_file + ext
            if os.path.exists(p):
                os.unlink(p)

    def test_semantic_search_returns_results(self):
        """Semantic search returns results ordered by similarity."""
        results = self.store.search_facts_semantic(self.user, "database setup")
        self.assertGreater(len(results), 0)
        # The PostgreSQL fact should be top or near-top.
        self.assertTrue(
            any("PostgreSQL" in r["content"] for r in results),
            msg=f"Expected PostgreSQL fact in: {[r['content'] for r in results]}",
        )

    def test_semantic_search_respects_limit(self):
        """Semantic search returns no more than `limit` results."""
        results = self.store.search_facts_semantic(self.user, "technology", limit=2)
        self.assertLessEqual(len(results), 2)

    def test_semantic_search_has_scores(self):
        """Each result has a 'score' key between 0 and 1."""
        results = self.store.search_facts_semantic(self.user, "deployment")
        for r in results:
            self.assertIn("score", r)
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_semantic_search_noise_isolation(self):
        """
        Semantic search should rank relevant facts above off-topic noise.
        The 'weather' fact should not appear in results for tech queries.
        """
        results = self.store.search_facts_semantic(self.user, "server infrastructure")
        weather_facts = [r for r in results if "Weather" in r["content"]]
        self.assertEqual(
            len(weather_facts), 0,
            msg=f"Weather noise leaked into results: {results}",
        )

    def test_semantic_search_empty_query(self):
        """Empty query returns empty list."""
        results = self.store.search_facts_semantic(self.user, "")
        self.assertEqual(len(results), 0)

    def test_semantic_search_fallback_without_engine(self):
        """Without an engine, search_facts_semantic falls back to keyword search."""
        store_no_ee = SQLiteStore(db_path=self.tmp_file)
        try:
            store_no_ee.add_fact(self.user, "Test fact")
            results = store_no_ee.search_facts_semantic(self.user, "test")
            # Falls back to keyword search, should return the fact with score=0.0
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0]["score"], 0.0)
        finally:
            store_no_ee.close()

    def test_semantic_search_meaning_vs_keyword(self):
        """
        Semantic search finds facts by meaning, not just keyword match.
        Querying 'rollback problem' should find the migration fact even
        though it doesn't contain the word 'problem'.
        """
        results = self.store.search_facts_semantic(self.user, "rollback problem")
        migration_facts = [r for r in results if "migration" in r["content"]]
        self.assertGreater(
            len(migration_facts), 0,
            msg=f"Expected migration fact in: {[r['content'] for r in results]}",
        )


if __name__ == "__main__":
    unittest.main()
