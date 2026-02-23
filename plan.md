# Apple Calendar Integration Plan

## Goal
Let the relay read your macOS Calendar so models can see your schedule and answer questions about it ("what's on my calendar today?", "am I free Thursday afternoon?", "evaluate my week").

## Approach: EventKit via PyObjC (read-only)

### New dependency
```bash
pip install pyobjc-framework-EventKit
```
Confirmed: PyObjC 12.1 has Python 3.14 wheels. Will also pull `pyobjc-core` and `pyobjc-framework-Cocoa`.

First run will trigger a macOS permission prompt for Calendar access (Terminal/iTerm needs to be allowed in System Settings > Privacy & Security > Calendars).

### New file: `calendar_client.py`
Keeps all EventKit/PyObjC code isolated. Exports two functions:

1. **`get_calendar_context(days=7) -> str`**
   - Requests calendar access (with run loop spin for async callback)
   - Fetches events from now through `days` days out
   - Returns a formatted plain-text block like:
     ```
     --- Your Calendar (next 7 days) ---
     Today (Sat Feb 22):
       - 3:00 PM - 4:00 PM: Dentist appointment
       - 7:00 PM - 9:00 PM: Dinner with Alex
     Mon Feb 24:
       - 9:00 AM - 10:00 AM: Team standup
       - 2:00 PM - 3:30 PM: Design review
     (no events on Tue Feb 25)
     --- End Calendar ---
     ```
   - Groups by day, sorted by start time
   - Includes all-day events
   - Returns empty string if access denied or no events
   - Caches for 5 minutes so we're not hammering EventKit every keystroke

2. **`list_calendars() -> list[dict]`**
   - Returns available calendars (name, id, color) for debugging/config

### Integration into the relay

**Where calendar context gets injected — two paths:**

1. **Local model (`call_ollama`)**: Inject calendar context into the conversation messages as a system-context block before the user's message. The local model sees the schedule and can answer questions about it directly.

2. **Remote models (`build_remote_messages`)**: When calendar context is relevant, include it so remote models can also reference the schedule.

**Trigger: on-demand, not always-on.**
- New command: `calendar` or `cal` — prints your upcoming schedule directly
- New `@calendar` trigger detection in the REPL loop — when detected, fetches calendar context and prepends it to the user's message before routing
- The local model doesn't need to "decide" to fetch calendar — the system detects the keyword and injects the data automatically
- Keywords that trigger injection: "calendar", "schedule", "meeting", "free", "busy", "appointment", "what's on my"

**No write access.** Read-only for now. Creating/modifying events from a CLI relay is risky — one bad parse and you've got phantom meetings. Can add later if wanted.

### Changes to existing files

1. **`config.py`** — Add `CALENDAR_LOOKAHEAD_DAYS = 7` constant
2. **`main.py`** — Add calendar trigger detection in the REPL loop (similar to @model triggers), import `get_calendar_context`, add `cal`/`calendar` command
3. **`clients.py`** — No changes needed
4. **`conversation.py`** — No changes needed

### File structure after
```
config.py              — constants (+ CALENDAR_LOOKAHEAD_DAYS)
conversation.py        — history management (unchanged)
clients.py             — API clients (unchanged)
calendar_client.py     — NEW: EventKit read-only access
main.py                — REPL loop (+ calendar trigger detection)
```

### Test plan
- `test_local.py` — Add tests for calendar context formatting (mock EventKit responses)
- Manual test: run `cal` command in the relay, verify events show up
- Manual test: ask "what's on my calendar this week?" and verify the model answers with real data
