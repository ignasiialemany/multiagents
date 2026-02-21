#!/usr/bin/env python3
"""
Tool designer: run the agent with the designer prompt and no tools to get a tool proposal.
Parses the response for a JSON proposal, logs it as "proposed", and optionally writes a proposal file.
Usage: python scripts/design_tool.py "I keep asking to grep the repo for pattern X in .py files only"
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))
load_dotenv(_repo_root / ".env.example")
load_dotenv(_repo_root / ".env")

from agent_llm import Agent, OpenRouterLLMClient
from agent_llm.designer import get_tool_designer_system_prompt, parse_tool_proposal

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TOOLS_DIR = _repo_root / "tools"
PROPOSALS_LOG = TOOLS_DIR / "proposals_log.jsonl"
PROPOSALS_SUBDIR = TOOLS_DIR / "proposals"


def _ensure_tools_dir() -> None:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSALS_SUBDIR.mkdir(parents=True, exist_ok=True)
    if not PROPOSALS_LOG.exists():
        PROPOSALS_LOG.touch()


def _append_proposal_log(entry: dict) -> None:
    _ensure_tools_dir()
    with open(PROPOSALS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/design_tool.py \"<description of the recurring task or desired tool>\"")
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    user_description = " ".join(sys.argv[1:]).strip()
    if not user_description:
        print("Provide a non-empty description.")
        sys.exit(1)

    system_prompt = get_tool_designer_system_prompt()
    user_message = f"Design a tool for this recurring need:\n\n{user_description}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    llm_client = OpenRouterLLMClient(api_key=api_key)
    agent = Agent(llm_client, tools={})
    logger.info("Running agent (no tools) to get a tool proposal...")
    final_text, _ = agent.run(messages)
    logger.info("Parsing proposal from response...")

    proposal = parse_tool_proposal(final_text)
    if not proposal:
        print("Could not parse a valid tool proposal from the response.")
        print("Final response (excerpt):")
        print(final_text[:1500] if len(final_text) > 1500 else final_text)
        sys.exit(1)

    name = proposal.get("name", "unknown")
    ts = datetime.now(timezone.utc).isoformat()
    log_entry = {"ts": ts, "type": "proposed", "name": name, "spec": proposal}
    _append_proposal_log(log_entry)

    proposal_path = PROPOSALS_SUBDIR / f"{name}_proposal.json"
    _ensure_tools_dir()
    proposal_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote proposal to %s", proposal_path)

    print("Proposed tool (for human review):")
    print(json.dumps(proposal, indent=2, ensure_ascii=False))
    print(f"\nLogged as 'proposed' in {PROPOSALS_LOG}")
    print(f"Spec written to {proposal_path}")


if __name__ == "__main__":
    main()
