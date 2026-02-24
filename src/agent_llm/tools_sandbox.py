import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from agent_llm.tools import _tool, Tool


def create_sandbox_tool(cwd_root: str | Path) -> Tool:
    """Create a tool that runs agent-provided code in a subprocess."""
    root = Path(cwd_root).resolve()
    # Ensure the root exists so we can create tempdirs inside it
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    def run_sandboxed_code(code: str, lang: str = "python") -> str:
        if lang.lower() != "python":
            return "Error: only python is supported for sandboxed execution."

        with tempfile.TemporaryDirectory(dir=root) as temp_dir:
            script_path = Path(temp_dir) / "script.py"
            # Unindent in case the agent wraps code in markdown code blocks or indented text
            clean_code = textwrap.dedent(code)
            script_path.write_text(clean_code, encoding="utf-8")

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,  # 10 second timeout
                )
                output = ""
                if result.stdout:
                    output += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    output += f"STDERR:\n{result.stderr}\n"
                output += f"Exit code: {result.returncode}"
                return output
            except subprocess.TimeoutExpired:
                return "Error: execution timed out after 10 seconds."
            except Exception as e:
                return f"Error executing code: {e}"

    return _tool(
        name="run_sandboxed_code",
        description="Run arbitrary python code in a restricted sandboxed directory with a 10s timeout. Useful as a fallback when you lack a specific tool.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                },
                "lang": {
                    "type": "string",
                    "description": "The programming language (default 'python'). Only 'python' is supported.",
                    "default": "python",
                },
            },
            "required": ["code"],
        },
        execute_fn=run_sandboxed_code,
    )
