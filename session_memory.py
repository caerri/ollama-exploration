"""Session memory — save/load/forget a compact session summary.

On quit, the local model generates a 3-5 line summary of the conversation.
On startup, that summary is auto-injected so the model knows what happened
last time. /forget clears it. One file, overwritten each session.

Exports:
  save_session_summary(conversation_history) — generate + write summary
  load_session_summary()                     — read existing summary or ""
  forget_session()                           — delete summary file
"""

from __future__ import annotations

from pathlib import Path

from config import SESSION_SUMMARY_PATH, DIM, RESET, get_env
from conversation import build_conversation_digest


# ---------------------------------------------------------------------------
# Summary prompt — sent to the local model at shutdown
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = (
    "Summarize this conversation in 3-5 lines. Focus on:\n"
    "- What the user was working on\n"
    "- Any decisions made\n"
    "- Anything left unfinished or explicitly planned for next time\n\n"
    "Be specific — use names, files, features, model names. "
    "Not generic. This will be injected into the next session so the AI "
    "knows what happened.\n\n"
    "Return ONLY the summary lines. No preamble, no JSON, no markdown headers."
)

# Minimum conversation exchanges before saving (don't save trivial sessions)
_MIN_EXCHANGES = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_session_summary() -> str:
    """Read the session summary file if it exists.

    Returns the summary text, or empty string if missing/empty/unreadable.
    Never raises — graceful degradation.
    """
    try:
        path = Path(SESSION_SUMMARY_PATH)
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            return text
    except (OSError, UnicodeDecodeError):
        pass
    return ""


def forget_session() -> bool:
    """Delete the session summary file.

    Returns True if deleted, False if it didn't exist or couldn't be deleted.
    """
    try:
        path = Path(SESSION_SUMMARY_PATH)
        if path.is_file():
            path.unlink()
            return True
    except OSError:
        pass
    return False


def save_session_summary(conversation_history: list[dict[str, str]]) -> bool:
    """Generate a session summary from conversation history and write it.

    Calls the local model silently (no streaming output) to produce a
    3-5 line summary, then writes it to SESSION_SUMMARY_PATH.

    Returns True on success, False on any failure. Never raises.
    """
    # Don't save trivial sessions
    user_msgs = [m for m in conversation_history if m["role"] == "user"]
    if len(user_msgs) < _MIN_EXCHANGES:
        return False

    # Build a plain-text digest of the conversation for the model
    digest = build_conversation_digest(max_turns=30)
    if not digest:
        return False

    prompt = f"{digest}\n\n{_SUMMARIZE_PROMPT}"

    try:
        # Import here to avoid circular imports (clients imports from conversation)
        from clients import call_ollama_direct

        model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        summary = call_ollama_direct(prompt, model=model, show_stream=False)

        if not summary or len(summary.strip()) < 20:
            return False  # model returned garbage

        # Ensure the parent directory exists
        path = Path(SESSION_SUMMARY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(summary.strip(), encoding="utf-8")
        return True

    except Exception:  # noqa: BLE001
        # Never crash the app on summary failure
        return False
