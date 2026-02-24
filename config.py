"""Configuration constants, model maps, and keyword lists.

This is the leaf module — it imports nothing from the project.
Every other module imports from here. Prompts live in prompts.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project directory (not the cwd).
# override=True ensures .env values win even if the shell has empty exports.
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


# --- ANSI color codes (UI layer — swap these out when moving to a GUI) ---
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[38;5;115m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[38;5;111m"
LOCAL_COLOR = "\033[38;5;252m"  # soft white — distinct from user text


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value or ""


# Keep last 40 entries (~20 exchanges) so remote models see enough context
MAX_HISTORY = 40

# Calendar: how many days ahead to fetch events
CALENDAR_LOOKAHEAD_DAYS = 7

# Keywords that trigger automatic calendar context injection
CALENDAR_KEYWORDS = [
    "calendar", "schedule", "meeting", "meetings",
    "free", "busy", "appointment", "what's on my",
    "what do i have", "am i free", "my week", "my day",
    "today's schedule", "tomorrow's schedule",
]

# Obsidian journal settings
OBSIDIAN_VAULT_PATH = get_env("OBSIDIAN_VAULT_PATH", "~/Documents/vault-zero")
JOURNAL_LOOKBACK_DAYS = 7

# Keywords that trigger journal context injection
JOURNAL_KEYWORDS = [
    "journal", "diary", "daily note", "daily notes",
    "how am i doing", "how's my week", "how is my week",
    "how was my week", "how's my day", "how was my day",
    "what did i do", "what have i done", "what have i been doing",
    "what did i work on", "what have i worked on",
    "my progress", "my mood", "my energy",
    "what got done", "what didn't get done",
    "my habits", "my routine",
    "my blockers", "what's blocking",
    "review my week", "weekly review", "week review",
    "how's it going", "how am i",
    "workload", "overwhelmed", "burned out", "burnout",
    "time management", "productivity",
    "should i take", "can i handle", "am i overloaded",
]


# --- Model configuration ---

# Models that require the Responses API instead of Chat Completions
RESPONSES_API_MODELS = {"gpt-5.2-pro"}

MODEL_MAP = {
    "HAIKU": "claude-haiku-4-5",
    "SONNET": "claude-sonnet-4-6",
    "OPUS": "claude-opus-4-6",
    "GPT_MINI": "gpt-5-mini",
    "GPT": "gpt-5.2",
    "GPT_PRO": "gpt-5.2-pro",
}

# Which provider handles each model
MODEL_PROVIDER = {
    "HAIKU": "anthropic",
    "SONNET": "anthropic",
    "OPUS": "anthropic",
    "GPT_MINI": "openai",
    "GPT": "openai",
    "GPT_PRO": "openai",
}

VALID_NEXT_STEPS = {"RESPOND_LOCALLY", "SEND_TO_REMOTE", "ASK_USER"}

COST_INFO = {
    "HAIKU": "cheap",
    "SONNET": "moderate",
    "OPUS": "expensive",
    "GPT_MINI": "cheap",
    "GPT": "moderate",
    "GPT_PRO": "very expensive",
}

# Shortcuts the user can type at the Phone a Friend prompt or as @mentions
MODEL_SHORTCUTS = {
    "haiku": "HAIKU",
    "sonnet": "SONNET",
    "opus": "OPUS",
    "gpt": "GPT",
    "mini": "GPT_MINI",
    "gpt_mini": "GPT_MINI",
    "gpt-mini": "GPT_MINI",
    "gpt_pro": "GPT_PRO",
    "gpt-pro": "GPT_PRO",
}
