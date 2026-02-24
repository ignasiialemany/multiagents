"""Tests for src/agent_llm/agent.py"""

import pytest

from agent_llm.agent import Agent, MAX_ITERATIONS


# ---------------------------------------------------------------------------
# Helpers / mock infrastructure
# ---------------------------------------------------------------------------


class MockLLM:
    """Minimal mock that cycles through a list of prepared responses."""

    def __init__(self, responses):
        """
        Args:
            responses: list of (content, tool_calls) tuples returned in order.
                       The last element is repeated if the list is exhausted.
        """
        self._responses = responses
        self._call_index = 0

    def complete(self, messages, tools=None):
        """Return the next prepared response, repeating the last one if needed."""
        idx = min(self._call_index, len(self._responses) - 1)
        self._call_index += 1
        return self._responses[idx]


def _tool_call(name, input_dict, call_id="tc-1"):
    """Build a tool_call dict in the format Agent.run expects."""
    return {"id": call_id, "name": name, "input": input_dict}


def _make_tool(name, description, properties, execute_fn):
    """Construct a tool dict compatible with Agent."""
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties},
        "execute": execute_fn,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_tools_direct_answer():
    """Agent with no tools returns the LLM's direct answer immediately."""
    llm = MockLLM([("The answer is 4.", [])])
    agent = Agent(llm, {})

    final_text, _ = agent.run([{"role": "user", "content": "2+2?"}])

    assert "4" in final_text


def test_unknown_tool_returns_error_in_messages():
    """
    When the LLM requests an unknown tool the agent appends an error tool
    message and continues; on the next LLM call it receives a final answer.
    """
    llm = MockLLM(
        [
            # First call: ask for an unknown tool
            ("", [_tool_call("summarize_document", {"doc": "x"})]),
            # Second call: produce a final answer
            ("done.", []),
        ]
    )
    agent = Agent(llm, {})  # no tools registered

    final_text, messages = agent.run([{"role": "user", "content": "summarize it"}])

    assert final_text == "done."

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert tool_messages, "Expected at least one tool-role message"
    combined = " ".join(m.get("content", "") for m in tool_messages).lower()
    assert "unknown tool" in combined or "error" in combined


def test_tool_call_executed():
    """A registered tool is called with the correct arguments."""
    call_log = []

    def echo(text=""):
        call_log.append(text)
        return text

    echo_tool = _make_tool("echo", "Echo back text", {"text": {"type": "string"}}, echo)
    tools = {"echo": echo_tool}

    llm = MockLLM(
        [
            ("", [_tool_call("echo", {"text": "hi"})]),
            ("ok", []),
        ]
    )
    agent = Agent(llm, tools)

    final_text, _ = agent.run([{"role": "user", "content": "echo hi"}])

    assert len(call_log) == 1, "execute should be called exactly once"
    assert call_log[0] == "hi"
    assert final_text == "ok"


def test_dedup_skips_repeated_tool_call():
    """
    If the LLM requests the same tool call twice with identical arguments, the
    agent deduplicates and only executes the tool once.
    """
    call_count = [0]

    def read(path=""):
        call_count[0] += 1
        return f"contents of {path}"

    read_tool = _make_tool("read", "Read a file", {"path": {"type": "string"}}, read)
    tools = {"read": read_tool}

    llm = MockLLM(
        [
            # Call 1: tool request
            ("", [_tool_call("read", {"path": "foo"}, call_id="tc-1")]),
            # Call 2: same tool request again
            ("", [_tool_call("read", {"path": "foo"}, call_id="tc-2")]),
            # Call 3: final answer
            ("done", []),
        ]
    )
    agent = Agent(llm, tools)

    final_text, _ = agent.run([{"role": "user", "content": "read foo twice"}])

    assert final_text == "done"
    assert call_count[0] == 1, "Dedup should prevent the second execution"


def test_max_iterations_returns_sentinel():
    """
    When the LLM never produces a final answer the agent returns the
    'maximum iterations' sentinel after MAX_ITERATIONS loops.
    """
    # A tool that always succeeds so we don't hit the unknown-tool path
    noop_tool = _make_tool("noop", "Do nothing", {}, lambda: "ok")
    tools = {"noop": noop_tool}

    # Always return a new tool_call so the loop never terminates naturally;
    # use a unique call_id per response to defeat dedup.
    call_number = [0]

    class InfiniteToolLLM:
        def complete(self, messages, tools=None):
            call_number[0] += 1
            return ("", [_tool_call("noop", {}, call_id=f"tc-{call_number[0]}")])

    agent = Agent(InfiniteToolLLM(), tools)

    final_text, _ = agent.run([{"role": "user", "content": "loop forever"}])

    assert "maximum iterations" in final_text.lower()


def test_none_input_handled_as_empty_dict():
    """
    A tool_call whose input is None is normalised to {} so tools with no
    required arguments are invoked without error.
    """
    llm = MockLLM(
        [
            ("", [_tool_call("ping", None)]),
            ("done", []),
        ]
    )
    ping_tool = _make_tool("ping", "Ping", {}, lambda: "pong")
    tools = {"ping": ping_tool}

    agent = Agent(llm, tools)
    final_text, _ = agent.run([{"role": "user", "content": "ping"}])

    assert final_text == "done"


def test_tool_execution_error_is_caught():
    """
    When a tool raises an exception the agent catches it, appends an error
    tool message, and continues to the next LLM call normally.
    """

    def explode():
        raise RuntimeError("boom")

    boom_tool = _make_tool("boom", "Always explodes", {}, explode)
    tools = {"boom": boom_tool}

    llm = MockLLM(
        [
            ("", [_tool_call("boom", {})]),
            ("recovered", []),
        ]
    )
    agent = Agent(llm, tools)

    final_text, messages = agent.run([{"role": "user", "content": "do it"}])

    assert final_text == "recovered"

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert tool_messages, "Expected an error tool message"
    error_content = tool_messages[0].get("content", "")
    assert "boom" in error_content or "Error" in error_content
