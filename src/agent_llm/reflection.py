"""
Reflection and planning for the generative agent architecture.

- maybe_reflect: triggered when cumulative importance exceeds a threshold;
  synthesises higher-level insights from recent high-importance memories.
- make_plan: given a goal, retrieves relevant memories and produces a
  step-by-step plan.

Both insert their outputs back into the memory stream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_llm.llm import OpenRouterLLMClient
    from agent_llm.memory import MemoryStream

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 50


def maybe_reflect(
    memory: MemoryStream,
    llm: OpenRouterLLMClient,
    threshold: int = _DEFAULT_THRESHOLD,
) -> list[str]:
    """
    Check whether cumulative importance since the last reflection exceeds
    *threshold*.  If so, synthesise 2-3 higher-level reflections and store
    them in the memory stream.

    Returns the list of reflection strings produced (empty if threshold not met).
    """
    if memory.cumulative_importance < threshold:
        return []

    recent = memory.get_recent(30)
    high = [e for e in recent if e.importance >= 6]
    if not high:
        high = recent[-10:]

    statements = "\n".join(
        f"- [{e.kind}] (importance {e.importance}) {e.content}" for e in high
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a reflective reasoning engine. Given a set of recent "
                "memories, produce 2-3 higher-level insights or generalisations. "
                "Each insight should be a single sentence. "
                "Return ONLY the insights, one per line, no numbering."
            ),
        },
        {
            "role": "user",
            "content": f"Recent memories:\n{statements}",
        },
    ]

    try:
        content, _ = llm.complete(messages, tools=None)
    except Exception as exc:
        logger.warning("Reflection LLM call failed: %s", exc)
        return []

    reflections: list[str] = []
    for line in content.strip().splitlines():
        line = line.strip().lstrip("-•").strip()
        if line:
            importance = llm.score_importance(line)
            memory.add(line, kind="reflection", importance=importance)
            reflections.append(line)

    memory.reset_cumulative_importance()
    logger.info("Produced %d reflections", len(reflections))
    return reflections


def make_plan(
    memory: MemoryStream,
    llm: OpenRouterLLMClient,
    goal: str,
) -> str:
    """
    Retrieve relevant memories for *goal*, ask the LLM to produce a
    step-by-step plan, and store it in the memory stream as kind="plan".

    Returns the plan text.
    """
    relevant = memory.retrieve(goal, k=15)

    context_lines = "\n".join(
        f"- [{e.kind}] {e.content}" for e in relevant
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a planning engine. Given relevant context and a goal, "
                "produce a concise step-by-step plan (3-7 steps). "
                "Return ONLY the numbered plan steps."
            ),
        },
        {
            "role": "user",
            "content": f"Goal: {goal}\n\nRelevant context:\n{context_lines}",
        },
    ]

    try:
        content, _ = llm.complete(messages, tools=None)
    except Exception as exc:
        logger.warning("Planning LLM call failed: %s", exc)
        return f"(planning failed: {exc})"

    plan_text = content.strip()
    importance = llm.score_importance(plan_text)
    memory.add(plan_text, kind="plan", importance=importance)
    logger.info("Created plan for goal: %s", goal[:60])
    return plan_text
