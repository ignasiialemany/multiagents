# Tool registry and proposals (Phase 2)

- **registry.json**: List of implemented tools. Each entry: `{ "name", "description", "parameters", "why_it_helps" (optional), "module", "function" }`. The loader imports `module` and gets `function` as the execute callable.
- **proposals_log.jsonl**: Append-only log. Each line: `{ "ts": ISO8601, "type": "proposed" | "implemented" | "rejected", "name": "...", "reason": "..." (for rejected), "spec": {...} (optional) }`.
- **proposals/**: `<name>_proposal.json` files placed here for review before implementation.

**Reject flow**: Append a `{ "ts": ISO8601, "type": "rejected", "name": "...", "reason": "..." }` entry to `proposals_log.jsonl` manually or via your own tooling.

**Heuristic trigger** (optional, not implemented): On repeated tool failures, you could re-run the agent in designer mode with a prompt like "You just hit limitations. Propose a new tool that would have helped."
