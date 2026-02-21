#!/usr/bin/env python3
"""
Run the agent with default tools plus any tools from tools/registry.json.
Usage: python scripts/run_with_registry.py [optional single prompt]
If no prompt is given, runs the same Phase 1-style prompts (happy-path + adversarial).
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))
load_dotenv(_repo_root / ".env.example")
load_dotenv(_repo_root / ".env")

from agent_llm import Agent, OpenRouterLLMClient, create_default_tools, load_registry_tools

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    notes_root = _repo_root / "notes"
    if not notes_root.is_dir():
        logger.error("notes/ directory not found at %s", notes_root)
        sys.exit(1)

    default_tools = create_default_tools(notes_root)
    registry_path = _repo_root / "tools" / "registry.json"
    custom_tools = load_registry_tools(registry_path)
    tools = {**default_tools, **custom_tools}
    if custom_tools:
        logger.info("Loaded %d custom tool(s) from registry: %s", len(custom_tools), list(custom_tools.keys()))

    llm_client = OpenRouterLLMClient(api_key=api_key)
    agent = Agent(llm_client, tools)

    if len(sys.argv) > 1:
        user_content = " ".join(sys.argv[1:]).strip()
        messages = [{"role": "user", "content": user_content}]
        final_text, _ = agent.run(messages)
        print("FINAL ANSWER:")
        print(final_text)
        return

    prompts = [
        ("Happy-path 1", "List the files in ./notes and tell me what looks interesting."),
        ("Happy-path 2", "Open notes/project.md and summarize the TODOs."),
        ("Adversarial (no tools)", "What is 2 + 2?"),
    ]
    for label, user_content in prompts:
        print("\n" + "=" * 60)
        print(f"PROMPT [{label}]: {user_content}")
        print("=" * 60)
        messages = [{"role": "user", "content": user_content}]
        try:
            final_text, _ = agent.run(messages)
            print("FINAL ANSWER:")
            print(final_text)
        except Exception as e:
            logger.exception("Run failed")
            print("RUN FAILED:", e)
    print("\nDone.")


if __name__ == "__main__":
    main()
