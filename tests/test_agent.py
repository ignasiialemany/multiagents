# Optional: formal tests for agent loop and unknown-tool handling.
# Run with: pytest tests/ (requires pip install -e ".[dev]")


def test_unknown_tool_handling() -> None:
    """When the model returns a tool name not in the registry, agent handles it and continues."""
    from agent_llm import Agent

    call_count = [0]

    class MockLLM:
        def complete(self, messages, tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return "", [{"id": "tc1", "name": "summarize_document", "input": {"path": "x"}}]
            return "I see that tool is not available.", []

    agent = Agent(MockLLM(), {})
    final_text, _ = agent.run([{"role": "user", "content": "Call summarize_document with path x."}])
    assert "not available" in final_text or "unknown" in final_text.lower()


def test_no_tools_direct_answer() -> None:
    """When the model returns no tool calls, agent returns that content immediately."""
    from agent_llm import Agent

    class MockLLM:
        def complete(self, messages, tools=None):
            return "The answer is 4.", []

    agent = Agent(MockLLM(), {})
    final_text, _ = agent.run([{"role": "user", "content": "What is 2+2?"}])
    assert "4" in final_text
