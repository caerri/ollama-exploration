"""Read-only Apple Calendar access via macOS EventKit.

Exports two functions:
  get_calendar_context(days)  — formatted text block of upcoming events
  list_calendars()            — available calendars (for debugging)

All EventKit/PyObjC code is isolated here. Nothing else in the project
imports from PyObjC.
"""

from __future__ import annotations

import time
from datetime import datetime

from config import get_env

# Cache: (timestamp, result_string)
_cache: tuple[float, str] = (0.0, "")
_CACHE_TTL = 300  # 5 minutes


def _request_access(store) -> bool:
    """Request calendar access, spinning the run loop until macOS responds."""
    from Foundation import NSDate, NSRunLoop
    from EventKit import EKEntityTypeEvent

    state = {"done": False, "granted": False}

    def handler(granted, err):
        state["done"] = True
        state["granted"] = bool(granted)

    store.requestAccessToEntityType_completion_(EKEntityTypeEvent, handler)

    timeout = time.time() + 10
    while not state["done"] and time.time() < timeout:
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )

    return state["granted"]


def _format_date_header(dt: datetime) -> str:
    """Format a date as 'Today (Sat Feb 22)' or 'Mon Feb 24'."""
    today = datetime.now().date()
    if dt.date() == today:
        return f"Today ({dt.strftime('%a %b %-d')})"
    return dt.strftime("%a %b %-d")


def _format_time(dt: datetime) -> str:
    """Format time as '3:00 PM', or 'all-day' placeholder handled separately."""
    return dt.strftime("%-I:%M %p")


def get_calendar_context(days: int | None = None) -> str:
    """Fetch upcoming events and return a formatted text block.

    Returns empty string if access denied, EventKit unavailable, or no events.
    Results are cached for 5 minutes.
    """
    if days is None:
        days = int(get_env("CALENDAR_LOOKAHEAD_DAYS", "7"))

    global _cache
    now = time.time()
    if _cache[1] and (now - _cache[0]) < _CACHE_TTL:
        return _cache[1]

    try:
        from EventKit import EKEventStore, EKEntityTypeEvent
        from Foundation import NSDate
    except ImportError:
        return ""

    store = EKEventStore.alloc().init()
    if not _request_access(store):
        return ""

    start = NSDate.date()
    end = NSDate.dateWithTimeIntervalSinceNow_(days * 24 * 3600)

    predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
        start, end, None
    )
    events = store.eventsMatchingPredicate_(predicate)
    if not events:
        result = f"--- Your Calendar (next {days} days) ---\nNo upcoming events.\n--- End Calendar ---"
        _cache = (now, result)
        return result

    # Group events by date
    by_day: dict[str, list[dict]] = {}
    for ev in events:
        # Convert NSDate to Python datetime
        ts = ev.startDate().timeIntervalSince1970()
        start_dt = datetime.fromtimestamp(ts)
        end_ts = ev.endDate().timeIntervalSince1970()
        end_dt = datetime.fromtimestamp(end_ts)

        day_key = start_dt.strftime("%Y-%m-%d")
        title = str(ev.title() or "(no title)")
        is_all_day = ev.isAllDay()

        if day_key not in by_day:
            by_day[day_key] = []
        by_day[day_key].append({
            "title": title,
            "start": start_dt,
            "end": end_dt,
            "all_day": is_all_day,
        })

    # Build output, sorted by date
    lines = [f"--- Your Calendar (next {days} days) ---"]

    # Fill in all days in range, including empty ones
    from datetime import timedelta
    today = datetime.now().date()
    for i in range(days):
        day = today + timedelta(days=i)
        day_key = day.strftime("%Y-%m-%d")
        header = _format_date_header(datetime.combine(day, datetime.min.time()))

        if day_key in by_day:
            lines.append(f"{header}:")
            day_events = sorted(by_day[day_key], key=lambda e: e["start"])
            for ev in day_events:
                if ev["all_day"]:
                    lines.append(f"  - All day: {ev['title']}")
                else:
                    lines.append(
                        f"  - {_format_time(ev['start'])} - {_format_time(ev['end'])}: {ev['title']}"
                    )
        # Skip empty days silently to keep output compact

    lines.append("--- End Calendar ---")
    result = "\n".join(lines)
    _cache = (now, result)
    return result


def list_calendars() -> list[dict]:
    """Return available calendars (name, id) for debugging."""
    try:
        from EventKit import EKEventStore, EKEntityTypeEvent
    except ImportError:
        return []

    store = EKEventStore.alloc().init()
    if not _request_access(store):
        return []

    cals = store.calendarsForEntityType_(EKEntityTypeEvent)
    return [
        {"name": str(c.title()), "id": str(c.calendarIdentifier())}
        for c in cals
    ]
