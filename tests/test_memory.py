"""Tests for src/agent_llm/memory.py"""

import json
import math
import time

import pytest

from agent_llm.memory import (
    MemoryEntry,
    MemoryStream,
    cosine_similarity,
    _recency_score,
    _DECAY_FACTOR,
)


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------


class TestMemoryEntry:
    def test_defaults(self):
        e = MemoryEntry(content="hello")
        assert e.kind == "observation"
        assert e.importance == 5
        assert e.embedding is None
        assert len(e.id) == 12

    def test_to_dict_omits_none_embedding(self):
        e = MemoryEntry(content="test")
        d = e.to_dict()
        assert "embedding" not in d
        assert d["content"] == "test"

    def test_to_dict_includes_embedding(self):
        e = MemoryEntry(content="x", embedding=[0.1, 0.2])
        d = e.to_dict()
        assert d["embedding"] == [0.1, 0.2]

    def test_roundtrip(self):
        e = MemoryEntry(content="rt", kind="reflection", importance=8)
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        assert e2.content == e.content
        assert e2.kind == e.kind
        assert e2.importance == e.importance
        assert e2.id == e.id

    def test_from_dict_defaults(self):
        e = MemoryEntry.from_dict({"content": "minimal"})
        assert e.kind == "observation"
        assert e.importance == 5


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty(self):
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert cosine_similarity([1, 2], [1]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------


class TestRecencyScore:
    def test_now_is_one(self):
        now = time.time()
        assert _recency_score(now, now) == pytest.approx(1.0)

    def test_one_hour_ago(self):
        now = time.time()
        expected = _DECAY_FACTOR ** 1.0
        assert _recency_score(now - 3600, now) == pytest.approx(expected)

    def test_future_clamped(self):
        now = time.time()
        assert _recency_score(now + 1000, now) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MemoryStream — basic operations
# ---------------------------------------------------------------------------


class TestMemoryStreamBasic:
    def test_add_and_len(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        assert len(ms) == 0
        ms.add("first")
        assert len(ms) == 1
        ms.add("second", kind="reflection", importance=9)
        assert len(ms) == 2

    def test_get_recent(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        for i in range(5):
            ms.add(f"entry-{i}")
        recent = ms.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].content == "entry-4"

    def test_cumulative_importance(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        ms.add("a", importance=3)
        ms.add("b", importance=7)
        assert ms.cumulative_importance == 10
        ms.reset_cumulative_importance()
        assert ms.cumulative_importance == 0

    def test_entries_returns_copy(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        ms.add("x")
        entries = ms.entries
        entries.clear()
        assert len(ms) == 1


# ---------------------------------------------------------------------------
# MemoryStream — persistence
# ---------------------------------------------------------------------------


class TestMemoryStreamPersistence:
    def test_persist_to_jsonl(self, tmp_path):
        path = tmp_path / "mem.jsonl"
        ms = MemoryStream(path)
        ms.add("one", importance=3)
        ms.add("two", importance=7)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        d = json.loads(lines[0])
        assert d["content"] == "one"

    def test_reload_from_disk(self, tmp_path):
        path = tmp_path / "mem.jsonl"
        ms1 = MemoryStream(path)
        ms1.add("alpha")
        ms1.add("beta", kind="plan", importance=8)

        ms2 = MemoryStream(path)
        assert len(ms2) == 2
        assert ms2.entries[0].content == "alpha"
        assert ms2.entries[1].kind == "plan"
        assert ms2.entries[1].importance == 8

    def test_handles_missing_file(self, tmp_path):
        ms = MemoryStream(tmp_path / "nonexistent.jsonl")
        assert len(ms) == 0

    def test_handles_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("NOT JSON\n")
        ms = MemoryStream(path)
        assert len(ms) == 0

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "mem.jsonl"
        ms = MemoryStream(path)
        ms.add("test")
        assert path.exists()


# ---------------------------------------------------------------------------
# MemoryStream — retrieval
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Returns a fixed embedding based on content length for deterministic tests."""

    def embed(self, text: str) -> list[float]:
        n = len(text) % 5
        return [float(n), 1.0 - float(n) / 5.0, float(n) / 3.0]


class TestMemoryStreamRetrieval:
    def test_retrieve_without_embedder(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl", embedder=None)
        ms.add("low", importance=1)
        ms.add("high", importance=10)
        results = ms.retrieve("anything", k=2)
        assert len(results) == 2
        # Higher importance should rank first (relevance=1.0 for all)
        assert results[0].content == "high"

    def test_retrieve_with_embedder(self, tmp_path):
        embedder = _FakeEmbedder()
        ms = MemoryStream(tmp_path / "mem.jsonl", embedder=embedder)
        ms.add("short", importance=5)
        ms.add("medium len", importance=5)
        ms.add("a very long content string", importance=5)
        results = ms.retrieve("short", k=3)
        assert len(results) == 3

    def test_retrieve_k_limits(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        for i in range(20):
            ms.add(f"entry-{i}")
        results = ms.retrieve("test", k=5)
        assert len(results) == 5

    def test_retrieve_empty_stream(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        assert ms.retrieve("anything") == []

    def test_embedding_stored_on_disk(self, tmp_path):
        path = tmp_path / "mem.jsonl"
        embedder = _FakeEmbedder()
        ms = MemoryStream(path, embedder=embedder)
        ms.add("test content")

        line = path.read_text().strip()
        d = json.loads(line)
        assert "embedding" in d
        assert isinstance(d["embedding"], list)
