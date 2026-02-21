# Tool registry and proposals (Phase 2)

- **registry.json**: List of implemented tools. Each entry: `{ "name", "description", "parameters", "why_it_helps" (optional), "module", "function" }`. The loader imports `module` and gets `function` as the execute callable.
- **proposals_log.jsonl**: Append-only log. Each line: `{ "ts": ISO8601, "type": "proposed" | "implemented" | "rejected", "name": "...", "reason": "..." (for rejected), "spec": {...} (optional) }`.
- **proposals/**: `design_tool.py` writes `<name>_proposal.json` here for review.

**Reject flow**: `python scripts/reject_tool.py <name> "reason"` appends a "rejected" entry to the log.

**Heuristic trigger** (optional, not implemented): On repeated tool failures, you could re-run the agent in designer mode with a prompt like "You just hit limitations. Propose a new tool that would have helped."
