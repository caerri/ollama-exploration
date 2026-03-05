"""Relay v2: Local-first multi-model LLM relay.

This is the entry point — REPL loop, routing orchestration, and user interaction.
Reusable building blocks live in config.py, conversation.py, and clients.py.
"""

from __future__ import annotations

import json
import sys
import termios
from datetime import datetime, timezone, timedelta

import requests

from config import (
    get_env,
    DIM, BOLD, RESET, CYAN, GREEN, YELLOW, RED, MAGENTA,
    MODEL_MAP, MODEL_PROVIDER, COST_INFO, MODEL_SHORTCUTS, RESPONSES_API_MODELS,
    CALENDAR_KEYWORDS, JOURNAL_KEYWORDS, CANVAS_KEYWORDS, PLANNING_KEYWORDS,
    STICKY_LOCAL_MAX,
)
from prompts import build_system_prompt, has_recent_context, REMOTE_SYSTEM_PROMPT, REMOTE_CONTEXT_PROMPT
from render import render_response
from calendar_client import get_calendar_context
from obsidian_client import get_journal_context
from canvas_client import get_canvas_context, sync_courses
from conversation import (
    conversation_history, add_message, clear_history,
    build_conversation_digest, build_remote_messages,
    parse_local_response, print_local_summary,
)
from clients import (
    call_ollama, call_ollama_direct, get_ollama_models,
    call_anthropic_stream, call_openai_stream, call_openai_responses_stream,
    unload_model,
)
from session_memory import load_session_summary, save_session_summary, forget_session


# ---------------------------------------------------------------------------
# Passphrase gate for expensive models
# ---------------------------------------------------------------------------
def _check_expensive_model(model_choice: str, user_input: str,
                           force_model: str | None) -> str:
    """If model_choice is OPUS or GPT_PRO, enforce passphrase or downgrade.
    Returns the (possibly downgraded) model_choice."""
    if model_choice not in ("OPUS", "GPT_PRO"):
        return model_choice

    provider_label = "Anthropic" if model_choice == "OPUS" else "OpenAI"
    downgrade = "SONNET" if model_choice == "OPUS" else "GPT"
    _lower = user_input.lower()
    user_requested = (
        force_model in ("OPUS", "GPT_PRO")
        or (model_choice == "OPUS" and "opus" in _lower)
        or (model_choice == "GPT_PRO" and ("gpt pro" in _lower or "gpt_pro" in _lower))
    )
    if user_requested:
        print(f"\n{RED}⚠  {model_choice} is the most expensive {provider_label} model.{RESET}")
        print(f"{YELLOW}Type the phrase exactly: {BOLD}Money grows on trees.{RESET}")
        passphrase = input(f"   > ").strip()
        if passphrase != "Money grows on trees.":
            print(f"{DIM}[Wrong phrase — downgrading to {downgrade}]{RESET}")
            return downgrade
        return model_choice
    else:
        print(f"\n{YELLOW}[GUARDRAIL] Local model picked {model_choice} without user request — downgrading to {downgrade}{RESET}")
        return downgrade


# ---------------------------------------------------------------------------
# Core remote call — used by both relay_once() and direct @model triggers
# ---------------------------------------------------------------------------
def send_to_remote(user_input: str, model_choice: str,
                    remote_system_prompt: str | None = None) -> None:
    """Send the user's question + full conversation history to a remote model.

    This is the single path for ALL remote calls — whether triggered by llama's
    routing, @model shortcuts, or Phone a Friend. It builds proper multi-turn
    messages and streams the response.
    """
    model_id = MODEL_MAP.get(model_choice, MODEL_MAP["HAIKU"])
    provider = MODEL_PROVIDER.get(model_choice, "anthropic")
    effective_system = remote_system_prompt or REMOTE_SYSTEM_PROMPT

    # Build multi-turn messages from real conversation history
    remote_messages = build_remote_messages(user_input)

    msg_count = len(remote_messages)
    total_chars = sum(len(m["content"]) for m in remote_messages)
    sys_chars = len(effective_system)
    print(f"\n{MAGENTA}--- {model_id} (remote — {provider}) ---{RESET}")
    print(f"{DIM}[Sending {msg_count} messages ({total_chars} chars) + system ({sys_chars} chars) to {model_id}]{RESET}")
    print()

    if provider == "openai":
        if model_id in RESPONSES_API_MODELS:
            remote_response = call_openai_responses_stream(
                model=model_id, messages=remote_messages, system_prompt=effective_system)
        else:
            remote_response = call_openai_stream(
                model=model_id, messages=remote_messages, system_prompt=effective_system)
    else:
        remote_response = call_anthropic_stream(
            model=model_id, messages=remote_messages, system_prompt=effective_system)

    # Auto-escalate: if cheap model says it doesn't know, offer upgrade
    dont_know_phrases = [
        "don't have information", "don't have reliable information",
        "my training data", "my knowledge cutoff", "not aware of",
        "knowledge was last updated", "i'm not sure about", "i cannot confirm",
        "as of my", "after my knowledge", "cannot provide", "i don't have",
    ]
    normalized = remote_response.lower().replace("\u2019", "'").replace("\u2018", "'")
    is_cheap = model_choice in ("HAIKU", "GPT_MINI")
    if is_cheap and any(phrase in normalized for phrase in dont_know_phrases):
        if provider == "openai":
            up_model, up_id, up_cost = "GPT", MODEL_MAP["GPT"], COST_INFO["GPT"]
        else:
            up_model, up_id, up_cost = "SONNET", MODEL_MAP["SONNET"], COST_INFO["SONNET"]
        escalate = input(f"\n{YELLOW}{model_choice} couldn't answer. Escalate to {up_model} ({up_cost})? [y/n] > {RESET}").strip().lower()
        if escalate in ("y", "yes"):
            print(f"\n{MAGENTA}--- {up_id} (remote — {provider}) ---{RESET}")
            if provider == "openai":
                if up_id in RESPONSES_API_MODELS:
                    remote_response = call_openai_responses_stream(
                        model=up_id, messages=remote_messages, system_prompt=effective_system)
                else:
                    remote_response = call_openai_stream(
                        model=up_id, messages=remote_messages, system_prompt=effective_system)
            else:
                remote_response = call_anthropic_stream(
                    model=up_id, messages=remote_messages, system_prompt=effective_system)
            model_choice = up_model
        else:
            print(f"{DIM}[Keeping {model_choice} response]{RESET}")

    # Inject response into conversation history so llama/deepseek see it later
    add_message("assistant", f"[Remote model ({model_choice}) responded]: {remote_response}")


# ---------------------------------------------------------------------------
# Main relay: llama routes, then dispatch
# ---------------------------------------------------------------------------
def relay_once(user_input: str, force_remote: bool = False,
               force_model: str | None = None,
               system_prompt: str | None = None,
               context_model: str | None = None,
               remote_system_prompt: str | None = None,
               rich_context_input: str | None = None) -> None:
    """Main relay: run llama for routing, then dispatch accordingly.

    Pass context_model to auto-escalate to a stronger local model when
    context data (calendar/journal/canvas) is injected.
    Pass remote_system_prompt to give remote models the context analysis instructions.
    Pass rich_context_input to send expanded context (full journal, assignment
    descriptions) to remote models instead of the compact local version.
    """
    _local_model_name = context_model or get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    label = f"{_local_model_name} (local)"
    if context_model:
        label = f"{_local_model_name} (local — context mode)"
    print(f"\n{CYAN}--- {label} ---{RESET}")
    local_raw = call_ollama(user_input, show_stream=True, system_prompt=system_prompt,
                            model=context_model)

    try:
        parsed = parse_local_response(local_raw)
    except ValueError:
        # JSON was likely truncated — retry with a shorter-answer instruction
        print(f"  {YELLOW}[TRUNCATED] Response too long for JSON — retrying with shorter format...{RESET}")
        retry_prompt = (f"The user asked: {user_input}\n\n"
                        f"Your previous answer was too long and got cut off. "
                        f"Give a SHORTER but still complete answer. "
                        f"Keep the output field under 1500 characters.")
        retry_raw = call_ollama(retry_prompt, system_prompt=system_prompt,
                               model=context_model)
        parsed = parse_local_response(retry_raw)

    # Python-side override: if the user is complaining about routing, force local
    complaint_phrases = ["stop prompting", "stop making remote", "stop sending",
                         "why did you send", "should not have sent",
                         "don't send", "don't use the api", "stop using the api",
                         "answer it yourself", "you should have answered"]
    if any(phrase in user_input.lower() for phrase in complaint_phrases):
        if parsed["NEXT_STEP"].upper() == "SEND_TO_REMOTE":
            parsed["NEXT_STEP"] = "RESPOND_LOCALLY"
            print(f"  {YELLOW}[OVERRIDE] User asked to stop remote calls — forcing local{RESET}")

    # If user explicitly requested remote via "phone a friend", override routing
    if force_remote and parsed["NEXT_STEP"].upper() != "SEND_TO_REMOTE":
        parsed["NEXT_STEP"] = "SEND_TO_REMOTE"
        if not parsed.get("MODEL") or parsed["MODEL"].upper() == "NONE":
            parsed["MODEL"] = "HAIKU"

    # If a specific model was forced via @mention, override whatever llama picked
    if force_model:
        parsed["MODEL"] = force_model
        if parsed["NEXT_STEP"].upper() != "SEND_TO_REMOTE":
            parsed["NEXT_STEP"] = "SEND_TO_REMOTE"

    print_local_summary(parsed)

    next_step = parsed["NEXT_STEP"].upper().strip()

    if next_step == "RESPOND_LOCALLY":
        output = parsed["OUTPUT"]
        # Detect truncated responses
        truncation_hints = ["here's how", "let's break", "step by step", "here are the steps",
                           "follow these", "here's a"]
        looks_truncated = False

        # Check 1: instructional opener that never delivered
        if (any(hint in output.lower() for hint in truncation_hints)
                and len(output) < 200
                and output.rstrip().endswith((".", ":", "!"))):
            looks_truncated = True

        # Check 2: response ends mid-sentence (no terminal punctuation)
        # But skip if response is long enough — it's probably a signature/name ending
        stripped = output.rstrip()
        if (stripped
                and 200 < len(stripped) < 800
                and stripped[-1] not in ".!?\"')]}\u201d\n"):
            looks_truncated = True

        if looks_truncated:
            print(f"\n{YELLOW}[NOTE] Response looks truncated — retrying with more space...{RESET}")
            retry_prompt = (f"The user asked: {user_input}\n"
                            f"You started to answer but got cut off. "
                            f"Please give the FULL complete answer in the output field. "
                            f"Make sure to finish your thought completely.")
            retry_raw = call_ollama(retry_prompt, system_prompt=system_prompt)
            retry_parsed = parse_local_response(retry_raw)
            output = retry_parsed["OUTPUT"]
        print(f"\n{DIM}[{_local_model_name}]{RESET}")
        render_response(output)
        return

    if next_step == "ASK_USER":
        print(f"\n{DIM}[{_local_model_name}]{RESET}")
        print(f"{YELLOW}{parsed['OUTPUT']}{RESET}")
        return

    if next_step == "SEND_TO_REMOTE":
        if parsed.get("SENSITIVE_DATA", "").upper() == "YES":
            print(f"\n{RED}[SAFETY] Sensitive data detected — sanitized prompt being sent.{RESET}")

        model_choice = parsed.get("MODEL", "HAIKU").upper().strip()
        model_choice = _check_expensive_model(model_choice, user_input, force_model)
        cost = COST_INFO.get(model_choice, "unknown cost")

        # --- Unified confirmation gate (deepseek + phone a friend in one prompt) ---
        escalation_model = get_env("OLLAMA_ESCALATION_MODEL", "")
        if not force_remote:
            print(f"\n{MAGENTA}📞 Phone a Friend?{RESET}  ({model_choice} — {cost})")
            options = f"   {BOLD}[y]{RESET} Send / {BOLD}[n]{RESET} Answer locally"
            if escalation_model:
                options += f" / {BOLD}[d]{RESET} Try {escalation_model} first"
            options += f" / {BOLD}[more]{RESET} Need more context"
            print(options)
            print(f"   {DIM}Or pick a model: haiku  sonnet  gpt  mini{RESET}")
            choice = input(f"   > ").strip().lower()

            # Check if the user typed a model name to override
            # Accept both "gpt_pro" and "@gpt_pro" syntax
            if choice.startswith("@"):
                choice = choice[1:]
            if choice in MODEL_SHORTCUTS:
                override_model = MODEL_SHORTCUTS[choice]
                override_model = _check_expensive_model(override_model, user_input, override_model)
                model_choice = override_model
                cost = COST_INFO.get(model_choice, "unknown cost")
                print(f"{DIM}[Switched to {model_choice} ({cost})]{RESET}")
                choice = "y"  # proceed with sending

            if choice == "d" and escalation_model:
                # Try local escalation (deepseek) before going remote
                print(f"\n{CYAN}--- {escalation_model} (local escalation) ---{RESET}")
                digest = build_conversation_digest()
                escalation_prompt_parts: list[str] = []
                if digest:
                    escalation_prompt_parts.append(digest)
                escalation_prompt_parts.append(
                    f"The user said: \"{user_input}\"\n\n"
                    f"Continue this conversation. The user is asking you to weigh in "
                    f"on what's been discussed above. Give a direct, thorough response. "
                    f"If it requires real-time data you don't have, say exactly: "
                    f"\"I cannot answer this.\""
                )
                escalation_prompt = "\n\n".join(escalation_prompt_parts)
                escalation_response = call_ollama_direct(
                    escalation_prompt, model=escalation_model, show_stream=True
                )
                punt_phrases = ["i cannot answer", "i don't have", "i'm not able to",
                                "i don't know", "beyond my knowledge", "no information",
                                "i'm unable to", "i cannot provide"]
                if escalation_response and not any(
                    phrase in escalation_response.lower() for phrase in punt_phrases
                ):
                    print(f"\n{DIM}[{escalation_model}]{RESET}")
                    return
                else:
                    print(f"{DIM}[Escalation model punted — proceeding to remote]{RESET}")
                    # Fall through to send_to_remote below

            elif choice in ("n", "no"):
                # Ask local model to answer it instead
                local_retry_prompt = (f"The user asked: \"{user_input}\"\n"
                                      f"You suggested sending this to the remote model, but the user "
                                      f"declined. Answer the question yourself to the best of your ability. "
                                      f"Set next_step to RESPOND_LOCALLY.")
                _retry_model = context_model or get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
                print(f"\n{CYAN}--- {_retry_model} (local retry) ---{RESET}")
                retry_raw = call_ollama(local_retry_prompt, show_stream=True,
                                       system_prompt=system_prompt,
                                       model=context_model)
                try:
                    retry_parsed = parse_local_response(retry_raw)
                    local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
                    print(f"\n{DIM}[{local_model}]{RESET}")
                    render_response(retry_parsed['OUTPUT'])
                except (ValueError, json.JSONDecodeError):
                    print(f"\n{YELLOW}[NOTE] Couldn't parse retry — here's the raw response{RESET}")
                    print(retry_raw)
                return

            elif choice in ("more", "m"):
                # Ask local model to generate a clarifying question
                clarify_prompt = (f"The user asked: \"{user_input}\"\n"
                                  f"You wanted to send this to the remote model, but the user wants "
                                  f"more context first. Ask the user a clarifying question that would "
                                  f"help you either answer locally or build a better remote prompt. "
                                  f"Set next_step to ASK_USER.")
                clarify_raw = call_ollama(clarify_prompt, system_prompt=system_prompt,
                                         model=context_model)
                try:
                    clarify_parsed = parse_local_response(clarify_raw)
                    print(f"\n{YELLOW}{clarify_parsed['OUTPUT']}{RESET}")
                except (ValueError, json.JSONDecodeError):
                    print(f"\n{YELLOW}What additional context would help here?{RESET}")
                return

            elif choice not in ("y", "yes"):
                print(f"{DIM}[Cancelled — press y to send]{RESET}")
                return

        # Use rich context for remote calls when available
        remote_input = rich_context_input if rich_context_input else user_input
        send_to_remote(remote_input, model_choice, remote_system_prompt=remote_system_prompt)
        return

    print(f"\n{RED}[ERROR] Unexpected routing: {next_step}{RESET}")


# ---------------------------------------------------------------------------
# REPL — the user-facing loop
# ---------------------------------------------------------------------------
def main() -> None:
    local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    escalation_model = get_env("OLLAMA_ESCALATION_MODEL", "")

    # Discover installed Ollama models and build @trigger shortcuts
    ollama_shortcuts = get_ollama_models()
    # For display, show shortest alias → full model name
    _seen_models: dict[str, str] = {}  # full_name → shortest shortcut
    for shortcut, full_name in ollama_shortcuts.items():
        if full_name not in _seen_models or len(shortcut) < len(_seen_models[full_name]):
            _seen_models[full_name] = shortcut
    local_triggers = "  ".join(
        f"@{s} ({fn.split(':')[1]})" if ':' in fn else f"@{s}"
        for fn, s in sorted(_seen_models.items(), key=lambda x: x[1])
    )

    print(f"{BOLD}{CYAN}Relay v2: Local (Ollama) → Remote (Claude / GPT){RESET}")
    print(f"{DIM}  Local:     {local_triggers}{RESET}")
    print(f"{DIM}  Anthropic: @haiku  @sonnet  @opus{RESET}")
    print(f"{DIM}  OpenAI:    @mini   @gpt     @gpt_pro{RESET}")
    print(f"{DIM}  Commands:  cal  journal  canvas  sync  clear  /forget  exit{RESET}\n")

    # --- Session memory: load previous session summary ---
    session_summary = load_session_summary()
    if session_summary:
        print(f"{DIM}Last session loaded. /forget to start fresh.{RESET}\n")

    # Sticky local model override — persists across loop iterations.
    # When the user switches to a non-default local model via @mention,
    # that model stays active for STICKY_LOCAL_MAX turns before reverting.
    sticky_local_model: str | None = None
    sticky_local_turns: int = 0

    while True:
        # Flush any leftover bytes in stdin (e.g. from a multi-line paste)
        # before reading new input, so stale data doesn't auto-feed
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        user_input = input("You > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            # Save session summary before exiting
            if save_session_summary(conversation_history):
                print(f"{DIM}[Session saved]{RESET}")
            print("Stopping relay.")
            unload_model()
            break
        if user_input.lower() == "clear":
            clear_history()
            sticky_local_model = None
            sticky_local_turns = 0
            print(f"{CYAN}[Conversation cleared]{RESET}")
            continue
        if user_input.lower() in {"cal", "calendar"}:
            cal_ctx = get_calendar_context()
            if cal_ctx:
                print(f"\n{CYAN}{cal_ctx}{RESET}")
            else:
                print(f"{YELLOW}[No calendar access — check System Settings → Privacy & Security → Calendars]{RESET}")
            continue
        if user_input.lower() in {"journal", "notes"}:
            journal_ctx = get_journal_context()
            if journal_ctx:
                print(f"\n{CYAN}{journal_ctx}{RESET}")
            else:
                print(f"{YELLOW}[No journal entries found — check your vault's Journal/Daily/ folder]{RESET}")
            continue
        if user_input.lower() in {"canvas", "assignments"}:
            canvas_ctx = get_canvas_context()
            if canvas_ctx:
                print(f"\n{CYAN}{canvas_ctx}{RESET}")
            else:
                print(f"{YELLOW}[No Canvas data — check CANVAS_API_URL and CANVAS_API_TOKEN in .env]{RESET}")
            continue
        if user_input.lower() in {"/forget", "forget"}:
            forget_session()
            session_summary = ""
            print(f"{CYAN}[Session memory cleared]{RESET}")
            continue
        if user_input.lower() == "sync":
            print(f"{DIM}[Syncing Canvas courses to vault...]{RESET}")
            result = sync_courses()
            n = result.get("files_written", 0)
            courses = result.get("courses", [])
            for c in courses:
                sections = ", ".join(c.get("sections", []))
                print(f"{CYAN}  {c['name']}: {sections}{RESET}")
            print(f"{GREEN}[Synced {n} files to vault]{RESET}")
            continue

        # Detect explicit model triggers
        force_remote = False
        force_local_model: str | None = None   # full Ollama model name (e.g. "qwen2.5:7b-instruct")
        force_model: str | None = None          # remote model canonical key (e.g. "HAIKU")
        lower_input = user_input.lower()

        # --- Check for @<local-ollama-model> triggers ---
        # Sort by shortcut length descending so "mistral-nemo" matches before "mistral"
        _local_matched = False
        for shortcut, full_model_name in sorted(
            ollama_shortcuts.items(), key=lambda x: len(x[0]), reverse=True
        ):
            trigger = f"@{shortcut}"
            if trigger in lower_input:
                idx = lower_input.index(trigger)
                user_input = (user_input[:idx] + user_input[idx + len(trigger):]).strip()
                if not user_input:
                    print(f"{YELLOW}Usage: {trigger} <your question>{RESET}")
                    _local_matched = True
                    break
                force_local_model = full_model_name
                _local_matched = True
                break
        if _local_matched and not force_local_model:
            continue  # empty usage — go back to prompt

        # --- Sticky local model: update state on explicit @local trigger ---
        default_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        if force_local_model:
            if force_local_model == default_model:
                # Tagging the default model clears sticky override
                sticky_local_model = None
                sticky_local_turns = 0
            else:
                sticky_local_model = force_local_model
                sticky_local_turns = STICKY_LOCAL_MAX

        # --- Check for @<remote-model> triggers (haiku, sonnet, gpt, etc.) ---
        if not force_local_model:
            if any(f"@{s}" in lower_input for s in MODEL_SHORTCUTS):
                _matched_empty = False
                # Sort by length descending so @gpt_pro matches before @gpt
                for shortcut, canonical in sorted(MODEL_SHORTCUTS.items(), key=lambda x: len(x[0]), reverse=True):
                    trigger = f"@{shortcut}"
                    if trigger in lower_input:
                        force_remote = True
                        force_model = canonical
                        user_input = user_input[:lower_input.index(trigger)] + user_input[lower_input.index(trigger) + len(trigger):]
                        user_input = user_input.strip()
                        if not user_input:
                            # No question provided — resend last substantive user message
                            for msg in reversed(conversation_history):
                                if msg["role"] == "user" and len(msg["content"]) >= 10:
                                    user_input = msg["content"]
                                    print(f"{DIM}[Resending: \"{user_input[:60]}{'...' if len(user_input) > 60 else ''}\"]{RESET}")
                                    break
                            if not user_input:
                                print(f"{YELLOW}Usage: {trigger} <your question>{RESET}")
                                _matched_empty = True
                        break
                if _matched_empty:
                    continue

            # "phone a friend" — force remote (llama still picks the model)
            elif "phone a friend" in lower_input:
                user_input = user_input[:lower_input.index("phone a friend")] + user_input[lower_input.index("phone a friend") + len("phone a friend"):]
                user_input = user_input.strip()
                if not user_input:
                    print(f"{YELLOW}Usage: phone a friend <your question>{RESET}")
                    print(f"{DIM}Or tag a model directly: @haiku @sonnet @gpt @mini{RESET}")
                else:
                    force_remote = True
                if not force_remote:
                    continue

        # --- Sticky local model: apply when no explicit trigger this turn ---
        if not force_local_model and not force_model and sticky_local_turns > 0:
            force_local_model = sticky_local_model
            sticky_local_turns -= 1
            if sticky_local_turns == 0:
                print(f"{DIM}[Sticky model expired — back to {default_model}]{RESET}")
                sticky_local_model = None

        # --- Session memory: inject previous session context (first turn only) ---
        # After first turn, the summary is in conversation history and doesn't
        # need re-injection.  Placed after keyword detection (which uses the
        # original lower_input) so session text can't trigger false context matches.
        if session_summary and not conversation_history:
            _session_block = (
                "--- Previous Session ---\n"
                f"{session_summary}\n"
                "--- End Previous Session ---"
            )
            user_input = _session_block + "\n\n" + user_input

        # --- Direct @<local-model>: skip routing, go straight to that Ollama model ---
        if force_local_model:
            if force_local_model == default_model:
                # User tagged the default model — use call_ollama (JSON routing mode)
                try:
                    print(f"\n{CYAN}--- {default_model} (local) ---{RESET}")
                    local_raw = call_ollama(user_input, show_stream=True)
                    parsed = parse_local_response(local_raw)
                    print(f"\n{DIM}[{default_model}]{RESET}")
                    render_response(parsed['OUTPUT'])
                except (ValueError, json.JSONDecodeError):
                    print(f"{YELLOW}[Couldn't parse response]{RESET}")
            else:
                # Non-default local model — use call_ollama_direct (free-form)
                add_message("user", user_input)
                print(f"\n{CYAN}--- {force_local_model} (local) ---{RESET}")
                digest = build_conversation_digest()
                prompt_parts: list[str] = []
                if digest:
                    prompt_parts.append(digest)
                prompt_parts.append(user_input)
                full_prompt = "\n\n".join(prompt_parts)
                resp = call_ollama_direct(full_prompt, model=force_local_model, show_stream=True)
                if resp:
                    print(f"{DIM}[{force_local_model}]{RESET}")
                else:
                    print(f"{YELLOW}[No response from {force_local_model}]{RESET}")
            continue

        # --- Direct @model: skip llama entirely, go straight to remote ---
        if force_model:
            model_choice = _check_expensive_model(force_model, user_input, force_model)
            # Record the user input in conversation history (llama didn't run to do it)
            add_message("user", user_input)
            try:
                send_to_remote(user_input, model_choice)
            except (requests.RequestException, ValueError) as err:
                print(f"{RED}[ERROR] {err}{RESET}")
            except Exception as err:  # noqa: BLE001
                print(f"{RED}[UNEXPECTED ERROR] {err}{RESET}")
            continue

        # --- Context injection (calendar + journal) ---
        # Either keyword list triggers loading BOTH context sources together,
        # so the model sees schedule + journal entries side by side.
        # Sticky: if recent history already has context data, stay in context mode.
        cal_match = any(kw in lower_input for kw in CALENDAR_KEYWORDS)
        journal_match = any(kw in lower_input for kw in JOURNAL_KEYWORDS)
        canvas_match = any(kw in lower_input for kw in CANVAS_KEYWORDS)
        planning_match = any(kw in lower_input for kw in PLANNING_KEYWORDS)
        sticky = has_recent_context(conversation_history)
        has_context = cal_match or journal_match or canvas_match or planning_match or sticky

        # --- Routing trace ---
        _triggers = []
        if cal_match:
            _kw = next(kw for kw in CALENDAR_KEYWORDS if kw in lower_input)
            _triggers.append(f"cal(\"{_kw}\")")
        if journal_match:
            _kw = next(kw for kw in JOURNAL_KEYWORDS if kw in lower_input)
            _triggers.append(f"journal(\"{_kw}\")")
        if canvas_match:
            _kw = next(kw for kw in CANVAS_KEYWORDS if kw in lower_input)
            _triggers.append(f"canvas(\"{_kw}\")")
        if planning_match:
            _kw = next(kw for kw in PLANNING_KEYWORDS if kw in lower_input)
            _triggers.append(f"planning(\"{_kw}\")")
        if sticky:
            _triggers.append("sticky(history)")
        if _triggers:
            fresh = cal_match or journal_match or canvas_match or planning_match
            _dest = "haiku" if fresh else "local (sticky)"
            print(f"{DIM}[route: {' + '.join(_triggers)} → {_dest}]{RESET}")

        # Build context: compact version for local model, rich version for remote
        _rich_context_input: str | None = None
        if cal_match or journal_match or canvas_match or planning_match:
            # Date/timezone header so remote models know when "today" is
            _mst = timezone(timedelta(hours=-7))
            _now = datetime.now(_mst)
            _date_header = (
                f"--- Today: {_now.strftime('%A, %B %-d, %Y at %-I:%M %p')} MST ---\n"
                "All calendar times are in MST (Mountain Standard Time).\n"
                "The user works a salaried office job, roughly 8 AM–5 PM weekdays. "
                "Assume that block is unavailable for personal tasks."
            )

            # Compact context (for local 14b — limited context window)
            context_parts = [_date_header]
            cal_ctx = get_calendar_context()
            if cal_ctx:
                context_parts.append(cal_ctx)
            journal_ctx = get_journal_context()
            if journal_ctx:
                context_parts.append(journal_ctx)
            canvas_ctx = get_canvas_context()
            if canvas_ctx:
                context_parts.append(canvas_ctx)
            if context_parts:
                _sources = []
                if cal_ctx:
                    _sources.append(f"cal({len(cal_ctx)}ch)")
                if journal_ctx:
                    _sources.append(f"journal({len(journal_ctx)}ch)")
                if canvas_ctx:
                    _sources.append(f"canvas({len(canvas_ctx)}ch)")
                print(f"{DIM}[context: {' + '.join(_sources)} → {len(user_input) + sum(len(p) for p in context_parts)}ch total]{RESET}")
                user_input = "\n\n".join(context_parts) + "\n\n" + user_input

            # Rich context (for remote models — full journal, assignment descriptions)
            rich_parts = [_date_header]
            if cal_ctx:
                rich_parts.append(cal_ctx)  # calendar is already full detail
            rich_journal = get_journal_context(rich=True)
            if rich_journal:
                rich_parts.append(rich_journal)
            rich_canvas = get_canvas_context(rich=True)
            if rich_canvas:
                rich_parts.append(rich_canvas)
            if rich_parts:
                _original_question = user_input.split("\n\n")[-1]  # extract original question
                _rich_context_input = "\n\n".join(rich_parts) + "\n\n" + _original_question

        # --- Context routing: fresh keywords → Haiku (accurate date analysis) ---
        fresh_context = cal_match or journal_match or canvas_match or planning_match
        if fresh_context and not force_remote:
            remote_input = _rich_context_input if _rich_context_input else user_input
            cost = COST_INFO.get("HAIKU", "cheap")
            print(f"{DIM}[model: → haiku ({cost}, context auto-route)]{RESET}")
            add_message("user", user_input)
            try:
                send_to_remote(remote_input, "HAIKU",
                               remote_system_prompt=REMOTE_CONTEXT_PROMPT)
            except (requests.RequestException, ValueError) as err:
                print(f"{RED}[ERROR] {err}{RESET}")
            except Exception as err:  # noqa: BLE001
                print(f"{RED}[UNEXPECTED ERROR] {err}{RESET}")
            continue

        # --- Sticky context: follow-ups stay on local escalation model ---
        system_prompt = build_system_prompt(has_context=has_context) if has_context else None
        _escalation = get_env("OLLAMA_ESCALATION_MODEL", "")
        _context_model = _escalation if (sticky and _escalation) else None
        _remote_prompt = REMOTE_CONTEXT_PROMPT if sticky else None
        if _context_model:
            _default = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
            print(f"{DIM}[model: {_default} → {_context_model} (sticky escalation)]{RESET}")

        # --- Default path: local model routes ---
        try:
            relay_once(user_input, force_remote=force_remote,
                       system_prompt=system_prompt, context_model=_context_model,
                       remote_system_prompt=_remote_prompt,
                       rich_context_input=_rich_context_input)
        except (requests.RequestException, ValueError) as err:
            print(f"{RED}[ERROR] {err}{RESET}")
        except Exception as err:  # noqa: BLE001
            print(f"{RED}[UNEXPECTED ERROR] {err}{RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        from conversation import conversation_history as _hist
        if save_session_summary(_hist):
            print(f"{DIM}[Session saved]{RESET}")
        print("Unloading model...")
        unload_model()
