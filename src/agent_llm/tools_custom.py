"""
Custom tools (Phase 2): implemented from agent proposals.
Each function takes kwargs matching its JSON Schema and returns a string.
Registry entries in tools/registry.json reference this module by name.
"""

import re
from pathlib import Path

# Default docs root (repo/notes), same as default tools
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ROOT = _REPO_ROOT / "notes"


def _resolve_safe(root: Path, path_str: str) -> Path | None:
    """Resolve path_str relative to root; return None if outside root."""
    try:
        path_str = path_str.strip() or "."
        resolved = (root / path_str).resolve()
        root_resolved = root.resolve()
        if not str(resolved).startswith(str(root_resolved)):
            return None
        return resolved
    except Exception:
        return None


def grep_repo(
    pattern: str,
    extension: str | None = None,
    root: str | Path | None = None,
) -> str:
    """
    Search for a text pattern in files under the docs root, optionally filtered by extension.
    Returns file paths and matching line snippets.
    """
    base = Path(root) if root else _DEFAULT_ROOT
    if not base.exists() or not base.is_dir():
        return f"Error: root directory does not exist: {base}"

    if not pattern or not pattern.strip():
        return "Error: pattern must be non-empty."

    pattern = pattern.strip()
    try:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    results: list[str] = []
    try:
        for fp in base.rglob("*"):
            if not fp.is_file():
                continue
            if extension is not None and extension.strip():
                ext = extension.strip().lower()
                if not ext.startswith("."):
                    ext = f".{ext}"
                if fp.suffix.lower() != ext:
                    continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            rel = fp.relative_to(base)
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(results) >= 30:
                        return "\n".join(results) + "\n... (truncated)"
    except OSError as e:
        return f"Error: {e}"
    if not results:
        return "No matches found."
    return "\n".join(results)
