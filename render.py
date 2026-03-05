"""Output rendering — thin wrapper for terminal display.

Swap this file to change how LLM responses look (TUI, web, etc.).
The rest of the project uses raw print() with ANSI codes for system
messages, labels, and prompts — this module only handles LLM response
content.

Currently uses ``rich`` for markdown rendering.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown

_console = Console(highlight=False)


def render_response(text: str) -> None:
    """Render a completed LLM response as formatted markdown."""
    if not text.strip():
        return
    _console.print(Markdown(text))


def create_streaming_renderer() -> tuple:
    """Create a simple token accumulator for streaming LLM output.

    Returns (context_manager, add_token, finish):
        context_manager — no-op, kept for API compatibility
        add_token(token) — prints each token inline as it arrives
        finish() — render final markdown, return accumulated text
    """
    buffer: list[str] = []

    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *_): pass

    def add_token(token: str) -> None:
        buffer.append(token)
        print(token, end="", flush=True)

    def finish() -> str:
        print()  # end the streaming line
        return "".join(buffer).strip()

    return _NullCtx(), add_token, finish
