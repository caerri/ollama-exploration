"""Test script: Simulate a multi-turn conversation and send it to each remote model.

This mimics what the app does — builds conversation_history as if a real session
happened, then uses build_remote_messages() to construct what each model receives.
One call per paid model. Evaluates whether they get enough context.
"""

from __future__ import annotations

import json
from config import get_env, MODEL_MAP, REMOTE_SYSTEM_PROMPT, BOLD, RESET, DIM, CYAN, GREEN, BLUE, MAGENTA, YELLOW
from conversation import (
    conversation_history, add_message, clear_history,
    build_remote_messages,
)
from clients import call_anthropic_stream, call_openai_stream, call_openai_responses_stream


def simulate_conversation():
    """Build a realistic multi-turn conversation history as the app would."""
    clear_history()

    # Turn 1: User greets, llama responds locally
    add_message("user", "hey! i'm working on a python web scraper and running into issues")
    add_message("assistant", json.dumps({
        "analysis": "user needs help with a python web scraper",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Hey! I'd be happy to help with your web scraper. What kind of issues are you running into? Are you getting errors, or is the scraper not pulling the data you expect?"
    }))

    # Turn 2: User gives details, llama responds locally
    add_message("user", "i'm using beautifulsoup to scrape product prices from an ecommerce site but the prices show up as empty. the page loads them with javascript.")
    add_message("assistant", json.dumps({
        "analysis": "user's scraper can't get JS-rendered content with beautifulsoup",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "That's a classic issue! BeautifulSoup only parses the static HTML — it can't execute JavaScript. The prices are probably loaded dynamically after the page loads. You have a few options: you could use Selenium or Playwright to render the JS first, or you could check if the site has an API endpoint that returns the price data directly (check the Network tab in dev tools). Which approach sounds better for your use case?"
    }))

    # Turn 3: User asks follow-up, llama routes to remote
    add_message("user", "i want to try playwright. can you give me a working example that scrapes prices from a page with dynamic content? also how do i handle pagination?")
    add_message("assistant", json.dumps({
        "analysis": "user wants a complete playwright example with pagination",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "SONNET",
        "output": "can you give me a working example that scrapes prices from a page with dynamic content? also how do i handle pagination?"
    }))

    # The key question that the remote model should answer
    return "i want to try playwright. can you give me a working example that scrapes prices from a page with dynamic content? also how do i handle pagination?"


def test_model(name: str, call_fn, model_id: str, messages: list[dict[str, str]]):
    """Call a single model and capture its response."""
    print(f"\n{'='*70}")
    print(f"{BOLD}Testing: {name} ({model_id}){RESET}")
    print(f"{'='*70}")
    print(f"{DIM}Messages being sent ({len(messages)} messages, {sum(len(m['content']) for m in messages)} chars):{RESET}")
    for i, msg in enumerate(messages):
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


def main():
    print(f"{BOLD}{CYAN}Remote Model Test: Simulated Multi-Turn Conversation{RESET}")
    print(f"{DIM}Testing whether each model gets proper context from build_remote_messages(){RESET}\n")

    # Build the conversation
    user_question = simulate_conversation()
    print(f"{CYAN}Simulated conversation history: {len(conversation_history)} entries{RESET}")
    for msg in conversation_history:
        role = msg["role"]
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  [{role}] {content}")

    # Build remote messages (same function the app uses)
    remote_messages = build_remote_messages(user_question)
    print(f"\n{CYAN}build_remote_messages() produced: {len(remote_messages)} messages{RESET}")
    for i, msg in enumerate(remote_messages):
        content = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
        print(f"  [{msg['role']}] {content}")

    results = {}

    # --- Test 1: Haiku (Anthropic, cheap) ---
    results["haiku"] = test_model(
        "Claude Haiku", call_anthropic_stream,
        MODEL_MAP["HAIKU"], remote_messages
    )

    # --- Test 2: Sonnet (Anthropic, moderate) ---
    results["sonnet"] = test_model(
        "Claude Sonnet", call_anthropic_stream,
        MODEL_MAP["SONNET"], remote_messages
    )

    # --- Test 3: GPT Mini (OpenAI, cheap) ---
    results["gpt_mini"] = test_model(
        "GPT Mini", call_openai_stream,
        MODEL_MAP["GPT_MINI"], remote_messages
    )

    # --- Test 4: GPT (OpenAI, moderate) ---
    results["gpt"] = test_model(
        "GPT", call_openai_stream,
        MODEL_MAP["GPT"], remote_messages
    )

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'='*70}")

    for name, response in results.items():
        if response.startswith("ERROR"):
            status = f"{YELLOW}FAILED{RESET}"
        elif len(response) < 100:
            status = f"{YELLOW}SUSPICIOUSLY SHORT{RESET}"
        else:
            # Check if the response actually addresses the question
            keywords = ["playwright", "scrape", "pric", "pagination", "page"]
            hits = sum(1 for kw in keywords if kw in response.lower())
            if hits >= 3:
                status = f"{GREEN}GOOD — addressed the question ({hits}/5 keywords){RESET}"
            elif hits >= 1:
                status = f"{YELLOW}PARTIAL — only {hits}/5 keywords{RESET}"
            else:
                status = f"{YELLOW}MISSED — didn't address the question{RESET}"
        print(f"  {BOLD}{name:10}{RESET}: {status} ({len(response)} chars)")

    clear_history()  # clean up
    print(f"\n{DIM}Done. Conversation history cleared.{RESET}")


if __name__ == "__main__":
    main()
