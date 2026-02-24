"""
src/agent_llm/cli.py

Entry point for the ``agent-llm`` console script.

Usage
-----
    agent-llm init            # scaffold a new project in the current directory
    agent-llm "your task"     # delegate to agent_llm._runner.main()
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Scaffold content
# ---------------------------------------------------------------------------

_AGENTS_JSON: list[dict] = [
    {
        "agent_id": "architect",
        "description": "Plans and delegates tasks to other agents.",
        "inbox_id": "architect",
    },
    {
        "agent_id": "coder",
        "description": "Writes code and implements features.",
        "inbox_id": "coder",
    },
]

_TOOLS_JSON: list = []

_ENV_EXAMPLE: str = """\
# Copy this to .env and fill in your key
OPENROUTER_API_KEY=your_openrouter_key_here
# AGENT_LLM_REDIS_URL=redis://localhost:6379
# AGENT_LLM_MODEL=minimax/minimax-m2.5
"""

_NEXT_STEPS: str = """\

Next steps:
  1. Copy .agent-llm/.env.example to .env and add your OPENROUTER_API_KEY
  2. Start Redis:  docker run -d -p 6379:6379 redis
  3. Run:          agent-llm "your task here"
"""


# ---------------------------------------------------------------------------
# Init helper
# ---------------------------------------------------------------------------


def _write_if_missing(path: Path, content: str) -> None:
    """Write *content* to *path* if it does not already exist."""
    if path.exists():
        print(f"  {path.relative_to(path.parent.parent)} already exists, skipping")
        return
    path.write_text(content, encoding="utf-8")
    print(f"  Created {path.relative_to(path.parent.parent)}")


def _init(work_dir: Path) -> None:
    """Create the .agent-llm scaffold inside *work_dir*."""
    config_dir = work_dir / ".agent-llm"
    notes_dir = work_dir / "notes"

    config_dir.mkdir(parents=True, exist_ok=True)
    created_notes = not notes_dir.exists()
    notes_dir.mkdir(exist_ok=True)
    if created_notes:
        print(f"  Created notes/")

    # agents.json
    _write_if_missing(
        config_dir / "agents.json",
        json.dumps(_AGENTS_JSON, indent=2) + "\n",
    )

    # tools.json
    _write_if_missing(
        config_dir / "tools.json",
        json.dumps(_TOOLS_JSON, indent=2) + "\n",
    )

    # .env.example
    _write_if_missing(config_dir / ".env.example", _ENV_EXAMPLE)

    print(_NEXT_STEPS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console-script entry point for the ``agent-llm`` command."""
    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        _init(Path.cwd())
        return

    from agent_llm._runner import main as _runner_main  # noqa: PLC0415

    _runner_main()
