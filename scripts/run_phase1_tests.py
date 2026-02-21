#!/usr/bin/env python3
"""
Phase 1 test harness: happy-path and adversarial prompts.
Loads env, builds client + tools + agent, runs reproducible prompts and prints answers.
Logging shows tool selection and args.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path so we can import agent_llm when run from repo root
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))
# Load .env.example first (defaults), then .env (overrides) so either file can hold the key
load_dotenv(_repo_root / ".env.example")
load_dotenv(_repo_root / ".env")

from agent_llm import Agent, OpenRouterLLMClient, create_default_tools

# Log tool selection and results
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set. Copy .env.example to .env and set the key.")
        sys.exit(1)

    # Default root for tools: ./notes (relative to repo root)
    notes_root = _repo_root / "notes"
    if not notes_root.is_dir():
        logger.error("notes/ directory not found at %s", notes_root)
        sys.exit(1)

    llm_client = OpenRouterLLMClient(api_key=api_key)
    tools = create_default_tools(notes_root)
    agent = Agent(llm_client, tools)

    prompts = [
        # Happy-path
        ("Happy-path 1", "List the files in ./notes and tell me what looks interesting."),
        ("Happy-path 2", "Open notes/project.md and summarize the TODOs."),
        # Adversarial
        ("Adversarial 1 (no tools)", "What is 2 + 2?"),
        ("Adversarial 2 (unknown tool)", "Call the tool named summarize_document with path x."),
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
