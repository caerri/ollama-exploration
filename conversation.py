"""Conversation history management, message building, and response parsing.

Owns the shared mutable state (conversation_history). Other modules import
the list by reference — mutations are visible everywhere.
"""

from __future__ import annotations

import json
import re

from config import (
    MAX_HISTORY,
    VALID_NEXT_STEPS,
    MODEL_SHORTCUTS,
    CYAN,
    RESET,
    RED,
    GREEN,
    MAGENTA,
    YELLOW,
)

# Words that indicate the user is giving a routing instruction, not asking a question.
# Used by build_remote_messages() to detect when user_input is "yup" or "opus please"
# rather than a real question, so we can send the actual question to the remote model.
_ROUTING_WORDS = set(MODEL_SHORTCUTS.keys()) | {
    "send", "remote", "please", "yup", "yeah", "yes", "do it",
    "go", "ahead", "send it", "try it", "use", "ok", "okay",
    "sure", "fine", "proceed", "just", "do", "it", "that",
}


# ---------------------------------------------------------------------------
# Shared conversation state
# ---------------------------------------------------------------------------
conversation_history: list[dict[str, str]] = []


def _trim_history() -> None:
    """Keep conversation_history within MAX_HISTORY entries."""
    if len(conversation_history) > MAX_HISTORY:
        conversation_history[:] = conversation_history[-MAX_HISTORY:]


def add_message(role: str, content: str) -> None:
    """Append a message and trim. Use this instead of raw .append() + _trim_history()."""
    conversation_history.append({"role": role, "content": content})
    _trim_history()


def clear_history() -> None:
    """Reset conversation history."""
    conversation_history.clear()


# ---------------------------------------------------------------------------
# Message builders — transform history for different consumers
# ---------------------------------------------------------------------------

def build_conversation_digest(max_turns: int = 10) -> str:
    """Build a plain-text digest of recent conversation for local models
    (deepseek, escalation) that don't support multi-turn message arrays.
    """
    if not conversation_history:
        return ""

    # Grab the last N entries (user + assistant pairs)
    recent = conversation_history[-max_turns * 2:]
    lines: list[str] = []
    for msg in recent:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            if content.startswith("[Remote model"):
                lines.append(f"Remote model: {content}")
            elif content.startswith("[Local escalation"):
                lines.append(content)
            else:
                # Try to extract just the output field from local JSON
                try:
                    parsed = json.loads(content)
                    local_answer = parsed.get("output", "")
                    if local_answer and parsed.get("next_step", "").upper() == "RESPOND_LOCALLY":
                        lines.append(f"Local model: {local_answer}")
                except (json.JSONDecodeError, AttributeError):
                    pass  # skip unparseable assistant turns
        elif role == "user":
            lines.append(f"User: {content}")

    if not lines:
        return ""
    return "--- Conversation so far ---\n" + "\n".join(lines) + "\n--- End of conversation ---"


def build_remote_messages(user_input: str, max_turns: int = 10) -> list[dict[str, str]]:
    """Build a proper multi-turn messages array for remote APIs.

    Sends real conversation history so the remote model sees the full picture:
    each user turn as a user message, each assistant response as an assistant
    message, and the current user question as the final user message.

    The user's actual words are always the final message — no llama rewrites.
    """
    messages: list[dict[str, str]] = []

    # Walk through conversation history, skipping the current turn's entries.
    # The current turn may have added: user_input (by call_ollama or manual append)
    # and possibly llama's routing JSON (by call_ollama).
    # Strip exactly those — NOT previous turns' entries.
    recent = conversation_history[-max_turns * 2:]

    # Strip at most one trailing llama routing JSON
    if (recent
            and recent[-1]["role"] == "assistant"):
        try:
            json.loads(recent[-1]["content"])
            # It's parseable JSON — likely llama's routing response for this turn
            recent = recent[:-1]
        except (json.JSONDecodeError, AttributeError):
            pass  # not JSON, it's a real response — keep it

    # Strip the current user_input (added by call_ollama or manual append)
    if (recent
            and recent[-1]["role"] == "user"
            and recent[-1]["content"] == user_input):
        recent = recent[:-1]

    for msg in recent:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            # Remote model responses — pass through as assistant messages
            if content.startswith("[Remote model"):
                # Strip the prefix: "[Remote model (MODEL) responded]: actual response"
                marker_end = content.find("]: ")
                clean = content[marker_end + 3:] if marker_end != -1 else content
                messages.append({"role": "assistant", "content": clean})
            elif content.startswith("[Local escalation"):
                marker_end = content.find("]: ")
                clean = content[marker_end + 3:] if marker_end != -1 else content
                messages.append({"role": "assistant", "content": clean})
            else:
                # Local model JSON — extract the output field as the "answer"
                try:
                    parsed = json.loads(content)
                    local_answer = parsed.get("output", "")
                    if local_answer and parsed.get("next_step", "").upper() == "RESPOND_LOCALLY":
                        messages.append({"role": "assistant", "content": local_answer})
                except (json.JSONDecodeError, AttributeError):
                    pass  # skip unparseable turns

    # Collapse consecutive same-role messages (APIs require alternating roles)
    collapsed: list[dict[str, str]] = []
    for msg in messages:
        if collapsed and collapsed[-1]["role"] == msg["role"]:
            collapsed[-1]["content"] += "\n\n" + msg["content"]
        else:
            collapsed.append(dict(msg))  # copy so we don't mutate history

    # Detect if user_input is a routing instruction ("yup", "opus please", "send it")
    # rather than a real question. If so, find the last substantive user message.
    effective_input = user_input
    _lower_input = user_input.lower().strip()
    _words = set(_lower_input.split())
    if len(_lower_input) < 30 and _words and _words.issubset(_ROUTING_WORDS):
        for msg in reversed(conversation_history):
            if msg["role"] == "user" and len(msg["content"]) >= 30:
                effective_input = msg["content"]
                break

    # Add the current user question as the final message.
    # If collapsed already ends with a user message, merge to maintain alternation.
    if collapsed and collapsed[-1]["role"] == "user":
        collapsed[-1]["content"] += "\n\n" + effective_input
    else:
        collapsed.append({"role": "user", "content": effective_input})

    # Ensure conversation starts with a user message (API requirement)
    if collapsed and collapsed[0]["role"] != "user":
        collapsed.insert(0, {"role": "user", "content": "(continuing conversation)"})

    return collapsed


# ---------------------------------------------------------------------------
# Response parsing — interpret the local model's structured JSON output
# ---------------------------------------------------------------------------

def parse_local_response(raw_text: str) -> dict[str, str]:
    """Parse JSON response from local model."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from the response
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Local model did not return valid JSON: {raw_text[:200]}")

    # Normalize keys to uppercase
    parsed: dict[str, str] = {}
    key_map = {
        "analysis": "ANALYSIS",
        "sensitive_data": "SENSITIVE_DATA",
        "next_step": "NEXT_STEP",
        "model": "MODEL",
        "output": "OUTPUT",
    }
    for json_key, field_key in key_map.items():
        value = data.get(json_key, data.get(field_key, ""))
        parsed[field_key] = str(value).strip()

    required = {"ANALYSIS", "SENSITIVE_DATA", "NEXT_STEP", "MODEL", "OUTPUT"}
    missing = required - {k for k, v in parsed.items() if v}
    if missing:
        raise ValueError(f"Local model response missing required fields: {', '.join(sorted(missing))}")

    # Fix invalid NEXT_STEP values
    next_step = parsed["NEXT_STEP"].upper().strip()
    if next_step not in VALID_NEXT_STEPS:
        # Default to RESPOND_LOCALLY if the model invented a value
        parsed["NEXT_STEP"] = "RESPOND_LOCALLY"

    return parsed


def print_local_summary(parsed: dict[str, str]) -> None:
    """Display the routing decision from the local model."""
    print(f"  {CYAN}Analysis:{RESET} {parsed['ANALYSIS']}")
    sensitive = parsed['SENSITIVE_DATA']
    sens_color = RED if sensitive.upper() == "YES" else CYAN
    print(f"  {CYAN}Sensitive:{RESET} {sens_color}{sensitive}{RESET}")
    route = parsed['NEXT_STEP']
    route_color = GREEN if route == "RESPOND_LOCALLY" else MAGENTA if route == "SEND_TO_REMOTE" else YELLOW
    print(f"  {CYAN}Route:{RESET} {route_color}{route}{RESET}", end="")
    if parsed.get("MODEL", "NONE").upper() != "NONE":
        print(f" {MAGENTA}→ {parsed['MODEL']}{RESET}")
    else:
        print()
