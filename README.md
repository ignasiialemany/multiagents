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

Run the optional unit tests (no API key needed): `pip install -e ".[dev]"` then `pytest tests/`.

Run the multi-agent runner from the repo root:

```bash
python scripts/run_multi_agent.py
```

Logs will show each agent's tool calls and messages per round.

**Phase 2 – Tool proposals (human-in-the-loop):**

Tool proposals are managed directly. Implement the function in `src/agent_llm/tools_custom.py`, then add the entry to `tools/registry.json` — the tool will be available to all agents on the next run.

See `tools/README.md` for registry and log format.

## Project layout

- `src/agent_llm/` – agent package
  - `llm.py` – OpenRouter client wrapper (OpenAI-compatible API)
  - `tools.py` – tool definitions + `load_registry_tools()`
  - `tools_custom.py` – custom tools (Phase 2)
  - `agent.py` – agent wrapper and tool-call loop
  - `prompts/tool_designer_system.txt` – designer system prompt
- `notes/` – default root for file tools (safe sandbox)
- `tools/` – `registry.json`, `proposals_log.jsonl`, `proposals/` (Phase 2)
- `scripts/run_multi_agent.py` – multi-agent runner
- `tests/test_agent.py` – optional unit tests (mock LLM)
