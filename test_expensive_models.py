"""Test suite for expensive gated models — Opus and GPT Pro.

Run this SEPARATELY from the main test suite since these cost real money.
Uses the same simulated multi-turn conversation as test_remote_models.py.

Usage:
    python test_expensive_models.py              # test both
    python test_expensive_models.py opus          # test Opus only
    python test_expensive_models.py gpt_pro       # test GPT Pro only
"""

from __future__ import annotations

import json
import sys

from config import (
    get_env, MODEL_MAP, REMOTE_SYSTEM_PROMPT, RESPONSES_API_MODELS,
    BOLD, RESET, DIM, CYAN, GREEN, YELLOW, RED, MAGENTA,
)
from conversation import (
    conversation_history, add_message, clear_history,
    build_remote_messages,
)
from clients import (
    call_anthropic_stream, call_openai_stream, call_openai_responses_stream,
)


def simulate_conversation() -> str:
    """Build a realistic multi-turn conversation history as the app would."""
    clear_history()

    # Turn 1: User greets, local model responds
    add_message("user", "hey! i'm working on a python web scraper and running into issues")
    add_message("assistant", json.dumps({
        "analysis": "user needs help with a python web scraper",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Hey! I'd be happy to help with your web scraper. What kind of issues are you running into? Are you getting errors, or is the scraper not pulling the data you expect?"
    }))

    # Turn 2: User gives details, local model responds
    add_message("user", "i'm using beautifulsoup to scrape product prices from an ecommerce site but the prices show up as empty. the page loads them with javascript.")
    add_message("assistant", json.dumps({
        "analysis": "user's scraper can't get JS-rendered content with beautifulsoup",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "That's a classic issue! BeautifulSoup only parses the static HTML — it can't execute JavaScript. The prices are probably loaded dynamically after the page loads. You have a few options: you could use Selenium or Playwright to render the JS first, or you could check if the site has an API endpoint that returns the price data directly (check the Network tab in dev tools). Which approach sounds better for your use case?"
    }))

    # Turn 3: User asks follow-up, local model routes to remote
    add_message("user", "i want to try playwright. can you give me a working example that scrapes prices from a page with dynamic content? also how do i handle pagination?")
    add_message("assistant", json.dumps({
        "analysis": "user wants a complete playwright example with pagination",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "SONNET",
        "output": "can you give me a working example that scrapes prices from a page with dynamic content? also how do i handle pagination?"
    }))

    return "i want to try playwright. can you give me a working example that scrapes prices from a page with dynamic content? also how do i handle pagination?"


def test_model(name: str, call_fn, model_id: str, messages: list[dict[str, str]]):
    """Call a single model and capture its response."""
    print(f"\n{'='*70}")
    print(f"{BOLD}Testing: {name} ({model_id}){RESET}")
    print(f"{'='*70}")
    print(f"{DIM}Messages being sent ({len(messages)} messages, {sum(len(m['content']) for m in messages)} chars):{RESET}")
    for msg in messages:
        role_color = CYAN if msg["role"] == "user" else GREEN
        content_preview = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
        print(f"  {role_color}[{msg['role']}]{RESET} {content_preview}")
    print()

    try:
        response = call_fn(model=model_id, messages=messages, system_prompt=REMOTE_SYSTEM_PROMPT)
        print(f"\n{DIM}Response length: {len(response)} chars{RESET}")
        return response
    except Exception as e:
        print(f"\n{YELLOW}ERROR: {e}{RESET}")
        return f"ERROR: {e}"


def evaluate(name: str, response: str):
    """Evaluate a model response for relevance."""
    if response.startswith("ERROR"):
        return f"{RED}FAILED{RESET}"
    if len(response) < 100:
        return f"{YELLOW}SUSPICIOUSLY SHORT{RESET}"

    keywords = ["playwright", "scrape", "pric", "pagination", "page"]
    hits = sum(1 for kw in keywords if kw in response.lower())
    if hits >= 3:
        return f"{GREEN}GOOD — addressed the question ({hits}/5 keywords){RESET}"
    elif hits >= 1:
        return f"{YELLOW}PARTIAL — only {hits}/5 keywords{RESET}"
    else:
        return f"{YELLOW}MISSED — didn't address the question{RESET}"


def main():
    # Parse CLI args
    run_opus = True
    run_gpt_pro = True
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower().strip()
        if arg in ("opus",):
            run_gpt_pro = False
        elif arg in ("gpt_pro", "gpt-pro", "gptpro", "pro"):
            run_opus = False
        else:
            print(f"{YELLOW}Usage: python test_expensive_models.py [opus|gpt_pro]{RESET}")
            print(f"{DIM}  No args = test both. Pass 'opus' or 'gpt_pro' to test one.{RESET}")
            sys.exit(1)

    which = []
    if run_opus:
        which.append("Opus")
    if run_gpt_pro:
        which.append("GPT Pro")
    label = " & ".join(which)

    print(f"{BOLD}{RED}💰 Expensive Model Test: {label}{RESET}")
    print(f"{YELLOW}These models cost real money. Ctrl+C to abort.{RESET}\n")

    # Build the conversation
    user_question = simulate_conversation()
    remote_messages = build_remote_messages(user_question)
    print(f"{CYAN}Conversation: {len(conversation_history)} history entries → {len(remote_messages)} remote messages{RESET}")

    results = {}

    # --- Opus ---
    if run_opus:
        results["opus"] = test_model(
            "Claude Opus", call_anthropic_stream,
            MODEL_MAP["OPUS"], remote_messages
        )

    # --- GPT Pro ---
    if run_gpt_pro:
        model_id = MODEL_MAP["GPT_PRO"]
        if model_id in RESPONSES_API_MODELS:
            call_fn = call_openai_responses_stream
        else:
            call_fn = call_openai_stream
        results["gpt_pro"] = test_model(
            "GPT Pro", call_fn,
            model_id, remote_messages
        )

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'='*70}")

    for name, response in results.items():
        status = evaluate(name, response)
        print(f"  {BOLD}{name:10}{RESET}: {status} ({len(response)} chars)")

    clear_history()
    print(f"\n{DIM}Done. Conversation history cleared.{RESET}")


if __name__ == "__main__":
    main()
