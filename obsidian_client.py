"""Read Obsidian daily journal notes from the local vault.

Exports:
  get_journal_context(days, vault_path)  — formatted text block of recent entries
  get_recent_entries(days, vault_path)   — raw parsed entries (for testing/future use)

All filesystem access is isolated here. Nothing else in the project
reads from the vault directly.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from config import get_env

# Cache: (timestamp, result_string)
_cache: tuple[float, str] = (0.0, "")
_CACHE_TTL = 300  # 5 minutes

# Max characters per section before truncation
_SECTION_MAX_CHARS = 300


def _parse_daily_note(path: Path) -> dict | None:
    """Parse a daily note file into frontmatter + sections.

    Returns a dict like:
        {
            "frontmatter": {"mood": 4, "energy": 3, "sleep_hours": 7, ...},
            "sections": {"Plan": "- [x] thing\\n- [ ] other", "What Happened": "...", ...},
            "date": "2026-02-22",
        }
    Or None if the file can't be read.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    result: dict = {
        "frontmatter": {},
        "sections": {},
        "date": path.stem,  # filename like "2026-02-22"
    }

    # --- Extract YAML frontmatter ---
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if fm_match:
        try:
            fm = yaml.safe_load(fm_match.group(1))
            if isinstance(fm, dict):
                result["frontmatter"] = fm
        except yaml.YAMLError:
            pass  # malformed YAML — continue with body only
        body = text[fm_match.end():]
    else:
        body = text

    # --- Parse markdown sections by ## headers ---
    current_section = None
    section_lines: dict[str, list[str]] = {}

    for line in body.split("\n"):
        header_match = re.match(r"^##\s+(.+)$", line)
        if header_match:
            current_section = header_match.group(1).strip()
            section_lines[current_section] = []
        elif current_section is not None:
            section_lines[current_section].append(line)

    # Join and strip each section
    for name, lines in section_lines.items():
        content = "\n".join(lines).strip()
        if content:
            result["sections"][name] = content

    return result


def _format_plan_inline(plan_text: str) -> str:
    """Convert multi-line checkbox plan into comma-separated inline format.

    Input:  "- [x] study math\\n- [ ] go to gym\\n- [x] code"
    Output: "[x] study math, [ ] go to gym, [x] code"
    """
    items = []
    for line in plan_text.split("\n"):
        line = line.strip()
        # Match "- [x] thing" or "- [ ] thing"
        m = re.match(r"^-\s*(\[[ x]\])\s*(.*)", line)
        if m:
            items.append(f"{m.group(1)} {m.group(2).strip()}")
    return ", ".join(items) if items else plan_text[:_SECTION_MAX_CHARS]


def _indent(text: str, spaces: int = 4) -> str:
    """Indent each line of text by N spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


def _truncate(text: str, max_chars: int = _SECTION_MAX_CHARS) -> str:
    """Truncate text to max_chars, adding ... if cut off."""
    # Collapse to single line for compact display
    single = " ".join(text.split())
    if len(single) <= max_chars:
        return single
    return single[:max_chars].rstrip() + "..."


def _format_entry(entry: dict, rich: bool = False) -> str:
    """Format a single day's parsed note into a text block.

    If rich=False (default), sections are truncated to _SECTION_MAX_CHARS
    for local model context windows.

    If rich=True, full section content is preserved for remote models
    that can handle large context.
    """
    # Parse the date for a friendly header
    try:
        dt = datetime.strptime(entry["date"], "%Y-%m-%d")
        today = datetime.now().date()
        if dt.date() == today:
            header = f"Today ({dt.strftime('%a %b %-d')})"
        elif dt.date() == today - timedelta(days=1):
            header = f"Yesterday ({dt.strftime('%a %b %-d')})"
        else:
            header = dt.strftime("%a %b %-d")
    except ValueError:
        header = entry["date"]

    # Build the metadata parenthetical
    fm = entry.get("frontmatter", {})
    meta_parts = []
    if fm.get("mood") is not None and fm["mood"] != "":
        meta_parts.append(f"mood:{fm['mood']}")
    if fm.get("energy") is not None and fm["energy"] != "":
        meta_parts.append(f"energy:{fm['energy']}")
    if fm.get("sleep_hours") is not None and fm["sleep_hours"] != "":
        meta_parts.append(f"sleep:{fm['sleep_hours']}h")

    if meta_parts:
        header += f" ({' '.join(meta_parts)})"
    header += ":"

    # Build section lines
    lines = [header]
    sections = entry.get("sections", {})

    # Plan gets special inline formatting (compact) or full checklist (rich)
    if "Plan" in sections:
        if rich:
            lines.append(f"  Plan:\n{_indent(sections['Plan'], 4)}")
        else:
            lines.append(f"  Plan: {_format_plan_inline(sections['Plan'])}")

    # Other sections: full content (rich) or truncated (compact)
    section_map = [
        ("What Happened", "Happened"),
        ("Wins", "Wins"),
        ("Blockers", "Blockers"),
        ("Notes", "Notes"),
    ]
    for full_name, short_name in section_map:
        if full_name in sections:
            if rich:
                lines.append(f"  {short_name}:\n{_indent(sections[full_name], 4)}")
            else:
                lines.append(f"  {short_name}: {_truncate(sections[full_name])}")

    return "\n".join(lines)


def get_recent_entries(days: int | None = None,
                       vault_path: str | None = None) -> list[dict]:
    """Return parsed journal entries for the last N days.

    Each entry is a dict from _parse_daily_note(), or a stub
    {"date": "YYYY-MM-DD", "missing": True} for days with no file.

    Returned in reverse chronological order (most recent first).
    """
    if days is None:
        days = int(get_env("JOURNAL_LOOKBACK_DAYS", "7"))
    if vault_path is None:
        vault_path = get_env("OBSIDIAN_VAULT_PATH", "~/Documents/vault-zero")

    journal_dir = Path(vault_path).expanduser() / "Journal" / "Daily"
    if not journal_dir.is_dir():
        return []

    entries = []
    today = datetime.now().date()

    for i in range(days):
        day = today - timedelta(days=i)
        date_str = day.strftime("%Y-%m-%d")
        note_path = journal_dir / f"{date_str}.md"

        if note_path.is_file():
            parsed = _parse_daily_note(note_path)
            if parsed:
                entries.append(parsed)
            else:
                entries.append({"date": date_str, "missing": True})
        else:
            entries.append({"date": date_str, "missing": True})

    return entries


def get_journal_context(days: int | None = None,
                        vault_path: str | None = None,
                        rich: bool = False) -> str:
    """Fetch recent journal entries and return a formatted text block.

    If rich=False (default), returns compact format for local models.
    If rich=True, returns full untruncated entries over a longer window
    for remote models that can analyze trends.

    Returns empty string if vault not found, journal dir missing, or no entries.
    Results are cached for 5 minutes (compact and rich cached separately).
    """
    if days is None:
        days = int(get_env("JOURNAL_LOOKBACK_DAYS", "7"))
        if rich:
            days = max(days, 30)  # at least 30 days for trend detection

    cache_key = f"{'rich' if rich else 'compact'}_{days}"
    global _cache
    now = time.time()
    # Simple cache — only reuse if same mode
    if _cache[1] and (now - _cache[0]) < _CACHE_TTL and getattr(get_journal_context, "_cache_key", "") == cache_key:
        return _cache[1]

    entries = get_recent_entries(days, vault_path)
    if not entries:
        return ""

    # Check if ALL entries are missing — no journal at all
    if all(e.get("missing") for e in entries):
        result = (
            f"--- Your Journal (last {days} days) ---\n"
            f"No journal entries found.\n"
            f"--- End Journal ---"
        )
        _cache = (now, result)
        get_journal_context._cache_key = cache_key
        return result

    lines = [f"--- Your Journal (last {days} days) ---"]

    for entry in entries:
        if entry.get("missing"):
            if rich:
                continue  # skip missing days in rich mode to reduce noise
            # Show the date but note no entry
            try:
                dt = datetime.strptime(entry["date"], "%Y-%m-%d")
                today = datetime.now().date()
                if dt.date() == today:
                    header = f"Today ({dt.strftime('%a %b %-d')})"
                elif dt.date() == today - timedelta(days=1):
                    header = f"Yesterday ({dt.strftime('%a %b %-d')})"
                else:
                    header = dt.strftime("%a %b %-d")
            except ValueError:
                header = entry["date"]
            lines.append(f"{header}:")
            lines.append("  (no entry)")
        else:
            lines.append(_format_entry(entry, rich=rich))

    lines.append("--- End Journal ---")
    result = "\n".join(lines)
    _cache = (now, result)
    get_journal_context._cache_key = cache_key
    return result
