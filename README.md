# agent-llm

Phase 1: Single agent that can reliably call a small set of custom tools.  
Phase 2: Agent can propose new tools; human reviews, implements, and registers them.

## Setup

1. Create a virtual environment and install:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e .
   ```

2. Copy `.env.example` to `.env` and set your OpenRouter API key:

   ```bash
   cp .env.example .env
   # Edit .env and set OPENROUTER_API_KEY=sk-or-...
   ```

   Get an API key from [openrouter.ai/keys](https://openrouter.ai/keys). Do not commit `.env` or put keys in code. The app uses OpenRouter (default model: `anthropic/claude-sonnet-4`; you can pick any model that supports tools on OpenRouter).

## Usage

Run the Phase 1 test harness (happy-path and adversarial prompts) from the repo root:

```bash
python scripts/run_phase1_tests.py
```

Logs will show tool selection and arguments for each run. Optional unit tests (no API key needed): `pip install -e ".[dev]"` then `pytest tests/`.

**Phase 2 – Tool proposals (human-in-the-loop):**

- **Design a tool**: `python scripts/design_tool.py "I keep asking to grep the repo for pattern X in .py files only"` — agent returns a JSON proposal; logged as "proposed" and written to `tools/proposals/<name>_proposal.json`.
- **Register after implementing**: Implement the function in `src/agent_llm/tools_custom.py`, then `python scripts/register_tool.py tools/proposals/<name>_proposal.json` — adds to `tools/registry.json` and logs "implemented".
- **Reject a proposal**: `python scripts/reject_tool.py <name> "reason"` — logs "rejected" in `tools/proposals_log.jsonl`.
- **Run with registry**: `python scripts/run_with_registry.py [prompt]` — loads default tools plus `tools/registry.json`; with a prompt runs once, without runs Phase 1–style prompts.

See `tools/README.md` for registry and log format.

## Project layout

- `src/agent_llm/` – agent package
  - `llm.py` – OpenRouter client wrapper (OpenAI-compatible API)
  - `tools.py` – tool definitions + `load_registry_tools()`
  - `tools_custom.py` – custom tools (Phase 2)
  - `agent.py` – agent wrapper and tool-call loop
  - `designer.py` – tool proposal parsing and designer prompt
  - `prompts/tool_designer_system.txt` – designer system prompt
- `notes/` – default root for file tools (safe sandbox)
- `tools/` – `registry.json`, `proposals_log.jsonl`, `proposals/` (Phase 2)
- `scripts/run_phase1_tests.py` – Phase 1 test runner
- `scripts/run_with_registry.py` – run agent with default + registry tools
- `scripts/design_tool.py` – get a tool proposal from the agent
- `scripts/register_tool.py` – add implemented tool to registry
- `scripts/reject_tool.py` – log a rejected proposal
- `tests/test_agent.py` – optional unit tests (mock LLM)
