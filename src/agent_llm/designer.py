"""
Tool designer mode: prompt and parsing for agent-proposed tools.
"""

import json
import re
from pathlib import Path

REQUIRED_PROPOSAL_KEYS = ("name", "description", "input_schema", "why_it_helps")

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "tool_designer_system.txt"


def get_tool_designer_system_prompt() -> str:
    """Load the designer system prompt from the prompts directory."""
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def parse_tool_proposal(final_text: str) -> dict | None:
    """
    Extract and validate a tool proposal JSON from the agent's final text.
    Looks for a ```json ... ``` block first, then for a JSON object containing
    "name" and "input_schema". Returns the parsed dict if valid, else None.
    """
    if not final_text or not final_text.strip():
        return None

    # Try ```json ... ``` block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", final_text)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if _is_valid_proposal(data):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: from each '{' try to parse a JSON object (handles nested braces)
    for i, c in enumerate(final_text):
        if c == "{":
            try:
                data = json.loads(final_text[i:])
                if _is_valid_proposal(data):
                    return data
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _is_valid_proposal(data: dict) -> bool:
    """Check that data has all required keys and input_schema is a dict."""
    if not isinstance(data, dict):
        return False
    for key in REQUIRED_PROPOSAL_KEYS:
        if key not in data:
            return False
    if not isinstance(data.get("input_schema"), dict):
        return False
    return True
