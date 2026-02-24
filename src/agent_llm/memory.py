"""
Memory stream for the generative agent architecture.

Implements an append-only log of timestamped observations, reflections, and plans.
Each entry carries an importance score and optional embedding for retrieval.
Retrieval scores memories by recency * relevance * importance (Park et al.).
Persistence: JSONL file per agent.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Anything that can turn text into a float vector."""

    def embed(self, text: str) -> list[float]: ...


class ImportanceScorer(Protocol):
    """Anything that can rate a memory's importance on 1-10."""

    def score_importance(self, text: str) -> int: ...


# ── Cosine similarity (no numpy needed) ─────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Memory entry ─────────────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    content: str
    kind: str = "observation"  # observation | reflection | plan
    importance: int = 5
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["embedding"] is None:
            del d["embedding"]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        return cls(
            content=d["content"],
            kind=d.get("kind", "observation"),
            importance=d.get("importance", 5),
            created_at=d.get("created_at", time.time()),
            id=d.get("id", uuid.uuid4().hex[:12]),
            embedding=d.get("embedding"),
        )


# ── Recency decay ───────────────────────────────────────────────────────────

_DECAY_FACTOR = 0.995  # per-hour exponential decay


def _recency_score(created_at: float, now: float) -> float:
    hours_ago = max((now - created_at) / 3600.0, 0.0)
    return _DECAY_FACTOR ** hours_ago


# ── Memory stream ────────────────────────────────────────────────────────────


class MemoryStream:
    """
    Append-only memory stream with retrieval.

    Parameters
    ----------
    persist_path:
        Path to a .jsonl file for persistence. Created if absent.
    embedder:
        Optional embedding provider. When None, relevance scoring is skipped.
    """

    def __init__(
        self,
        persist_path: str | Path,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.persist_path = Path(persist_path)
        self.embedder = embedder
        self._entries: list[MemoryEntry] = []
        self._cumulative_importance: int = 0
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with self.persist_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._entries.append(MemoryEntry.from_dict(json.loads(line)))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load memory stream from %s: %s", self.persist_path, exc)

    def _append_to_disk(self, entry: MemoryEntry) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.persist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Failed to persist memory entry: %s", exc)

    # ── public API ───────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        kind: str = "observation",
        importance: int = 5,
    ) -> MemoryEntry:
        """Add a new memory entry. Embedding is computed if an embedder is available."""
        embedding = None
        if self.embedder is not None:
            try:
                embedding = self.embedder.embed(content)
            except Exception as exc:
                logger.warning("Embedding failed, storing without: %s", exc)

        entry = MemoryEntry(
            content=content,
            kind=kind,
            importance=importance,
            embedding=embedding,
        )
        self._entries.append(entry)
        self._cumulative_importance += importance
        self._append_to_disk(entry)
        return entry

    def get_recent(self, n: int = 20) -> list[MemoryEntry]:
        """Return the last *n* entries (most recent last)."""
        return self._entries[-n:]

    def retrieve(self, query: str, k: int = 10) -> list[MemoryEntry]:
        """
        Retrieve top-k memories scored by recency * relevance * importance.

        If no embedder is available, relevance is set to 1.0 for all entries.
        """
        if not self._entries:
            return []

        query_emb: list[float] | None = None
        if self.embedder is not None:
            try:
                query_emb = self.embedder.embed(query)
            except Exception:
                query_emb = None

        now = time.time()
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries:
            recency = _recency_score(entry.created_at, now)
            importance = entry.importance / 10.0  # normalise to 0-1

            if query_emb is not None and entry.embedding is not None:
                relevance = max(cosine_similarity(query_emb, entry.embedding), 0.0)
            else:
                relevance = 1.0

            score = recency * relevance * importance
            scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    @property
    def cumulative_importance(self) -> int:
        return self._cumulative_importance

    def reset_cumulative_importance(self) -> None:
        self._cumulative_importance = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)
