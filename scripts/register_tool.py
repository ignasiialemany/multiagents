#!/usr/bin/env python3
"""
Add an implemented tool to tools/registry.json and log it as "implemented".
Usage:
  python scripts/register_tool.py tools/proposals/grep_repo_proposal.json
  python scripts/register_tool.py --name grep_repo --module agent_llm.tools_custom --function grep_repo --spec tools/proposals/grep_repo_proposal.json
If only --spec is given, name/module/function are derived from the spec (name from spec, module and function default to agent_llm.tools_custom and spec name).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

TOOLS_DIR = _repo_root / "tools"
REGISTRY_PATH = TOOLS_DIR / "registry.json"
PROPOSALS_LOG = TOOLS_DIR / "proposals_log.jsonl"


def main() -> None:
    args = sys.argv[1:]
    spec_path = None
    name = None
    module = "agent_llm.tools_custom"
    function = None

    i = 0
    while i < len(args):
        if args[i] == "--spec" and i + 1 < len(args):
            spec_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--name" and i + 1 < len(args):
            name = args[i + 1]
            i += 2
        elif args[i] == "--module" and i + 1 < len(args):
            module = args[i + 1]
            i += 2
        elif args[i] == "--function" and i + 1 < len(args):
            function = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            spec_path = Path(args[i])
            i += 1
        else:
            i += 1

    if spec_path is None:
        print("Usage: register_tool.py [--name NAME] [--module MODULE] [--function FUNC] <spec.json>")
        print("  or:  register_tool.py --spec tools/proposals/<name>_proposal.json")
        sys.exit(1)

    if not spec_path.is_absolute():
        spec_path = _repo_root / spec_path
    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}")
        sys.exit(1)

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to read spec: {e}")
        sys.exit(1)

    tool_name = name or spec.get("name")
    if not tool_name:
        print("Spec must contain 'name' or use --name")
        sys.exit(1)
    function_name = function or tool_name
    description = spec.get("description", "")
    parameters = spec.get("input_schema")
    if not isinstance(parameters, dict):
        print("Spec must contain 'input_schema' (JSON Schema object)")
        sys.exit(1)
    why_it_helps = spec.get("why_it_helps", "")

    registry_entry = {
        "name": tool_name,
        "description": description,
        "parameters": parameters,
        "why_it_helps": why_it_helps,
        "module": module,
        "function": function_name,
    }

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registry = []
    else:
        registry = []
    if not isinstance(registry, list):
        registry = []
    for ent in registry:
        if isinstance(ent, dict) and ent.get("name") == tool_name:
            print(f"Tool '{tool_name}' already in registry; updating entry.")
            registry = [e for e in registry if not (isinstance(e, dict) and e.get("name") == tool_name)]
            break
    registry.append(registry_entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Added to registry: {tool_name} (module={module}, function={function_name})")

    log_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "implemented",
        "name": tool_name,
        "spec": registry_entry,
    }
    with open(PROPOSALS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(f"Logged as 'implemented' in {PROPOSALS_LOG}")


if __name__ == "__main__":
    main()
