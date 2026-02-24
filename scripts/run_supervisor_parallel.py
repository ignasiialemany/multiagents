#!/usr/bin/env python3
"""
Thin shim — kept for backward compatibility.
All logic now lives in src/agent_llm/_runner.py.

Usage (same as before):
    python scripts/run_supervisor_parallel.py "your task here"
    python scripts/run_supervisor_parallel.py --help
"""

try:
    from agent_llm._runner import main
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from agent_llm._runner import main

if __name__ == "__main__":
    main()
