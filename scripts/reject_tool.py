#!/usr/bin/env python3
"""
Log a tool proposal as rejected (human-in-the-loop).
Usage: python scripts/reject_tool.py <name> <reason>
Example: python scripts/reject_tool.py summarize_doc "Duplicate of read_file + manual summary"
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
TOOLS_DIR = _repo_root / "tools"
PROPOSALS_LOG = TOOLS_DIR / "proposals_log.jsonl"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: reject_tool.py <tool_name> <reason>")
        sys.exit(1)
    name = sys.argv[1]
    reason = " ".join(sys.argv[2:]).strip() or "No reason given"
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "rejected",
        "name": name,
        "reason": reason,
    }
    with open(PROPOSALS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Logged rejection of '{name}' in {PROPOSALS_LOG}")


if __name__ == "__main__":
    main()
