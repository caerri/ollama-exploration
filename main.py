"""Relay v2: Local-first multi-model LLM relay.

This is the entry point — REPL loop, routing orchestration, and user interaction.
Reusable building blocks live in config.py, conversation.py, and clients.py.
"""

from __future__ import annotations

import json
import sys
import termios
import requests

from config import (
    get_env,
    DIM, BOLD, RESET, CYAN, GREEN, YELLOW, RED, MAGENTA, BLUE,
    MODEL_MAP, MODEL_PROVIDER, COST_INFO, MODEL_SHORTCUTS, RESPONSES_API_MODELS,
    REMOTE_SYSTEM_PROMPT,
)
from conversation import (
    conversation_history, add_message, clear_history,
    build_conversation_digest, build_remote_messages,
    parse_local_response, print_local_summary,
)
from clients import (
    call_ollama, call_ollama_direct,
    call_anthropic_stream, call_openai_stream, call_openai_responses_stream,
    unload_model,
)


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
def send_to_remote(user_input: str, model_choice: str) -> None:
    """Send the user's question + full conversation history to a remote model.

    This is the single path for ALL remote calls — whether triggered by llama's
    routing, @model shortcuts, or Phone a Friend. It builds proper multi-turn
    messages and streams the response.
    """
    model_id = MODEL_MAP.get(model_choice, MODEL_MAP["HAIKU"])
    provider = MODEL_PROVIDER.get(model_choice, "anthropic")

    # Build multi-turn messages from real conversation history
    remote_messages = build_remote_messages(user_input)

    msg_count = len(remote_messages)
    total_chars = sum(len(m["content"]) for m in remote_messages)
    print(f"\n{MAGENTA}--- {model_id} (remote — {provider}) ---{RESET}")
    print(f"{DIM}[Sending {msg_count} messages ({total_chars} chars) to {model_id}]{RESET}")
    print()

    if provider == "openai":
        if model_id in RESPONSES_API_MODELS:
            remote_response = call_openai_responses_stream(
                model=model_id, messages=remote_messages, system_prompt=REMOTE_SYSTEM_PROMPT)
        else:
            remote_response = call_openai_stream(
                model=model_id, messages=remote_messages, system_prompt=REMOTE_SYSTEM_PROMPT)
    else:
        remote_response = call_anthropic_stream(
            model=model_id, messages=remote_messages, system_prompt=REMOTE_SYSTEM_PROMPT)

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
                        model=up_id, messages=remote_messages, system_prompt=REMOTE_SYSTEM_PROMPT)
                else:
                    remote_response = call_openai_stream(
                        model=up_id, messages=remote_messages, system_prompt=REMOTE_SYSTEM_PROMPT)
            else:
                remote_response = call_anthropic_stream(
                    model=up_id, messages=remote_messages, system_prompt=REMOTE_SYSTEM_PROMPT)
            model_choice = up_model
        else:
            print(f"{DIM}[Keeping {model_choice} response]{RESET}")

    # Inject response into conversation history so llama/deepseek see it later
    add_message("assistant", f"[Remote model ({model_choice}) responded]: {remote_response}")


# ---------------------------------------------------------------------------
# Main relay: llama routes, then dispatch
# ---------------------------------------------------------------------------
def relay_once(user_input: str, force_remote: bool = False, force_model: str | None = None) -> None:
    """Main relay: run llama for routing, then dispatch accordingly."""
    _local_model_name = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    print(f"\n{CYAN}--- {_local_model_name} (local) ---{RESET}")
    local_raw = call_ollama(user_input, show_stream=True)

    try:
        parsed = parse_local_response(local_raw)
    except ValueError:
        # JSON was likely truncated — retry with a shorter-answer instruction
        print(f"  {YELLOW}[TRUNCATED] Response too long for JSON — retrying with shorter format...{RESET}")
        retry_prompt = (f"The user asked: {user_input}\n\n"
                        f"Your previous answer was too long and got cut off. "
                        f"Give a SHORTER but still complete answer. "
                        f"Keep the output field under 1500 characters.")
        retry_raw = call_ollama(retry_prompt)
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
        stripped = output.rstrip()
        if (stripped
                and len(stripped) > 200
                and stripped[-1] not in ".!?\"')]}\u201d"):
            looks_truncated = True

        if looks_truncated:
            print(f"\n{YELLOW}[NOTE] Response looks truncated — retrying with more space...{RESET}")
            retry_prompt = (f"The user asked: {user_input}\n"
                            f"You started to answer but got cut off. "
                            f"Please give the FULL complete answer in the output field. "
                            f"Make sure to finish your thought completely.")
            retry_raw = call_ollama(retry_prompt)
            retry_parsed = parse_local_response(retry_raw)
            output = retry_parsed["OUTPUT"]
        local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        print(f"\n{DIM}[{local_model}]{RESET}")
        print(f"{BOLD}{output}{RESET}")
        return

    if next_step == "ASK_USER":
        local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        print(f"\n{DIM}[{local_model}]{RESET}")
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
                _retry_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
                print(f"\n{CYAN}--- {_retry_model} (local retry) ---{RESET}")
                retry_raw = call_ollama(local_retry_prompt, show_stream=True)
                try:
                    retry_parsed = parse_local_response(retry_raw)
                    local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
                    print(f"\n{DIM}[{local_model}]{RESET}")
                    print(f"{BOLD}{retry_parsed['OUTPUT']}{RESET}")
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
                clarify_raw = call_ollama(clarify_prompt)
                try:
                    clarify_parsed = parse_local_response(clarify_raw)
                    print(f"\n{YELLOW}{clarify_parsed['OUTPUT']}{RESET}")
                except (ValueError, json.JSONDecodeError):
                    print(f"\n{YELLOW}What additional context would help here?{RESET}")
                return

            elif choice not in ("y", "yes"):
                print(f"{DIM}[Cancelled — press y to send]{RESET}")
                return

        send_to_remote(user_input, model_choice)
        return

    print(f"\n{RED}[ERROR] Unexpected routing: {next_step}{RESET}")


# ---------------------------------------------------------------------------
# REPL — the user-facing loop
# ---------------------------------------------------------------------------
def main() -> None:
    local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    escalation_model = get_env("OLLAMA_ESCALATION_MODEL", "")
    print(f"{BOLD}{CYAN}Relay v2: Local (Ollama) → Remote (Claude / GPT){RESET}")
    print(f"{DIM}  Local:     @llama  @deepseek{RESET}")
    print(f"{DIM}  Anthropic: @haiku  @sonnet  @opus{RESET}")
    print(f"{DIM}  OpenAI:    @mini   @gpt     @gpt_pro{RESET}")
    print(f"{DIM}  Commands:  clear   exit{RESET}\n")

    while True:
        # Flush any leftover bytes in stdin (e.g. from a multi-line paste)
        # before reading new input, so stale data doesn't auto-feed
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        user_input = input("You > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Stopping relay.")
            unload_model()
            break
        if user_input.lower() == "clear":
            clear_history()
            print(f"{CYAN}[Conversation cleared]{RESET}")
            continue

        # Detect explicit model triggers
        force_remote = False
        force_deepseek = False
        force_llama = False
        force_model: str | None = None
        lower_input = user_input.lower()

        # @deepseek — skip llama, go straight to deepseek
        if "@deepseek" in lower_input:
            force_deepseek = True
            user_input = user_input[:lower_input.index("@deepseek")] + user_input[lower_input.index("@deepseek") + len("@deepseek"):]
            user_input = user_input.strip()
            if not user_input:
                print(f"{YELLOW}Usage: @deepseek <your question>{RESET}")
                continue

        # @llama — force local model (no escalation, no remote)
        elif "@llama" in lower_input:
            force_llama = True
            user_input = user_input[:lower_input.index("@llama")] + user_input[lower_input.index("@llama") + len("@llama"):]
            user_input = user_input.strip()
            if not user_input:
                print(f"{YELLOW}Usage: @llama <your question>{RESET}")
                continue

        # @haiku, @sonnet, @gpt, @mini etc — skip llama, go straight to remote
        elif any(f"@{s}" in lower_input for s in MODEL_SHORTCUTS):
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

        # --- Direct @deepseek: skip everything, go straight to deepseek ---
        if force_deepseek:
            escalation_model = get_env("OLLAMA_ESCALATION_MODEL", "")
            if not escalation_model:
                print(f"{YELLOW}No escalation model configured (OLLAMA_ESCALATION_MODEL is empty){RESET}")
            else:
                # Record the user's question so future turns have context
                add_message("user", user_input)
                print(f"\n{CYAN}--- {escalation_model} (local) ---{RESET}")
                digest = build_conversation_digest()
                prompt_parts: list[str] = []
                if digest:
                    prompt_parts.append(digest)
                prompt_parts.append(user_input)
                full_prompt = "\n\n".join(prompt_parts)
                response = call_ollama_direct(full_prompt, model=escalation_model, show_stream=True)
                if response:
                    print(f"{DIM}[{escalation_model}]{RESET}")
                else:
                    print(f"{YELLOW}[No response from {escalation_model}]{RESET}")
            continue

        # --- Direct @llama: force local model ---
        if force_llama:
            try:
                local_model_name = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
                print(f"\n{CYAN}--- {local_model_name} (local) ---{RESET}")
                local_raw = call_ollama(user_input, show_stream=True)
                parsed = parse_local_response(local_raw)
                print(f"\n{DIM}[{local_model_name}]{RESET}")
                print(f"{BOLD}{parsed['OUTPUT']}{RESET}")
            except (ValueError, json.JSONDecodeError):
                print(f"{YELLOW}[Couldn't parse response]{RESET}")
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

        # --- Default path: llama routes ---
        try:
            relay_once(user_input, force_remote=force_remote)
        except (requests.RequestException, ValueError) as err:
            print(f"{RED}[ERROR] {err}{RESET}")
        except Exception as err:  # noqa: BLE001
            print(f"{RED}[UNEXPECTED ERROR] {err}{RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Unloading model...")
        unload_model()
