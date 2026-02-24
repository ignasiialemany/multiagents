"""Tests for src/agent_llm/reflection.py"""

import pytest

from agent_llm.memory import MemoryStream
from agent_llm.reflection import maybe_reflect, make_plan


# ---------------------------------------------------------------------------
# Mock LLM for testing
# ---------------------------------------------------------------------------


class MockLLM:
    """Returns canned responses for complete() and score_importance()."""

    def __init__(self, complete_response="Mock insight one.\nMock insight two."):
        self._complete_response = complete_response

    def complete(self, messages, tools=None):
        return (self._complete_response, [])

    def score_importance(self, text):
        return 7

    def embed(self, text):
        return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# maybe_reflect
# ---------------------------------------------------------------------------


class TestMaybeReflect:
    def test_no_reflection_below_threshold(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        ms.add("small note", importance=3)
        llm = MockLLM()
        result = maybe_reflect(ms, llm, threshold=50)
        assert result == []

    def test_reflects_when_threshold_exceeded(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        for i in range(10):
            ms.add(f"important event {i}", importance=8)
        # cumulative = 80, above threshold=50
        llm = MockLLM()
        result = maybe_reflect(ms, llm, threshold=50)
        assert len(result) == 2
        assert "Mock insight one" in result[0]
        assert "Mock insight two" in result[1]

    def test_reflections_added_to_memory(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        for i in range(10):
            ms.add(f"event {i}", importance=8)
        llm = MockLLM()
        initial_len = len(ms)
        maybe_reflect(ms, llm, threshold=50)
        assert len(ms) == initial_len + 2
        reflections = [e for e in ms.entries if e.kind == "reflection"]
        assert len(reflections) == 2

    def test_cumulative_importance_resets_after_reflect(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        for i in range(10):
            ms.add(f"event {i}", importance=6)
        llm = MockLLM()
        maybe_reflect(ms, llm, threshold=50)
        assert ms.cumulative_importance == 0

    def test_forced_reflection_with_zero_threshold(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        ms.add("single event", importance=1)
        llm = MockLLM()
        result = maybe_reflect(ms, llm, threshold=0)
        assert len(result) >= 1

    def test_handles_llm_failure(self, tmp_path):
        class FailingLLM:
            def complete(self, messages, tools=None):
                raise RuntimeError("LLM down")
            def score_importance(self, text):
                return 5

        ms = MemoryStream(tmp_path / "mem.jsonl")
        for i in range(10):
            ms.add(f"event {i}", importance=8)
        result = maybe_reflect(ms, FailingLLM(), threshold=50)
        assert result == []


# ---------------------------------------------------------------------------
# make_plan
# ---------------------------------------------------------------------------


class TestMakePlan:
    def test_returns_plan_text(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        ms.add("user wants a web app", importance=7)
        llm = MockLLM(complete_response="1. Design\n2. Implement\n3. Test")
        plan = make_plan(ms, llm, "build a web app")
        assert "Design" in plan
        assert "Implement" in plan

    def test_plan_stored_in_memory(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        ms.add("context", importance=5)
        llm = MockLLM(complete_response="1. Step one\n2. Step two")
        initial_len = len(ms)
        make_plan(ms, llm, "my goal")
        assert len(ms) == initial_len + 1
        plans = [e for e in ms.entries if e.kind == "plan"]
        assert len(plans) == 1

    def test_plan_with_empty_memory(self, tmp_path):
        ms = MemoryStream(tmp_path / "mem.jsonl")
        llm = MockLLM(complete_response="1. Start from scratch")
        plan = make_plan(ms, llm, "a goal")
        assert "Start from scratch" in plan

    def test_handles_llm_failure(self, tmp_path):
        class FailingLLM:
            def complete(self, messages, tools=None):
                raise RuntimeError("LLM down")
            def score_importance(self, text):
                return 5

        ms = MemoryStream(tmp_path / "mem.jsonl")
        plan = make_plan(ms, FailingLLM(), "goal")
        assert "planning failed" in plan


# ---------------------------------------------------------------------------
# Integration: reflect + plan together
# ---------------------------------------------------------------------------


class TestReflectionPlanIntegration:
    def test_reflect_then_plan(self, tmp_path):
        """Reflections created by maybe_reflect are visible to make_plan."""
        ms = MemoryStream(tmp_path / "mem.jsonl")
        for i in range(10):
            ms.add(f"event {i}", importance=8)

        llm = MockLLM(complete_response="Key insight: events are important.")
        maybe_reflect(ms, llm, threshold=50)

        llm2 = MockLLM(complete_response="1. Use the key insight\n2. Build on it")
        plan = make_plan(ms, llm2, "next steps")
        assert "Build on it" in plan
        # Memory should have original events + reflection + plan
        kinds = [e.kind for e in ms.entries]
        assert "reflection" in kinds
        assert "plan" in kinds
