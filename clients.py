"""API client functions for Ollama, Anthropic, and OpenAI.

Each function handles streaming I/O (printing tokens as they arrive)
and returns the collected response text.
"""

from __future__ import annotations

import json
import requests
from anthropic import Anthropic
from openai import OpenAI

from config import get_env, LOCAL_SYSTEM_PROMPT, DIM, RESET, GREEN, BLUE
from conversation import conversation_history, add_message, _trim_history


# ---------------------------------------------------------------------------
# Ollama (local models)
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, show_stream: bool = False) -> str:
    """Send a message to Ollama using the chat API (supports conversation history).

    If show_stream is True, tokens are printed to stdout as they arrive so the
    user can see the model "thinking".
    """
    base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    model = get_env("OLLAMA_MODEL", "gemma2:9b")
    url = f"{base_url.rstrip('/')}/api/chat"

    # Add current message to history
    conversation_history.append({"role": "user", "content": prompt})

    messages = [{"role": "system", "content": LOCAL_SYSTEM_PROMPT}] + conversation_history

    response = requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "format": "json",
            "stream": True,
            "options": {
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        },
        timeout=300,
        stream=True,
    )
    response.raise_for_status()

    # Collect streamed tokens
    chunks: list[str] = []
    if show_stream:
        print(DIM, end="", flush=True)  # dim the thinking stream
    for line in response.iter_lines():
        if not line:
            continue
        payload = json.loads(line)
        token = payload.get("message", {}).get("content", "")
        if token:
            chunks.append(token)
            if show_stream:
                print(token, end="", flush=True)
        if payload.get("done", False):
            break
    if show_stream:
        print(RESET)  # end dim + newline

    assistant_content = "".join(chunks).strip()

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": assistant_content})
    _trim_history()

    return assistant_content


def call_ollama_direct(prompt: str, model: str | None = None, show_stream: bool = False) -> str:
    """Call a specific Ollama model directly — no JSON format, no routing,
    no conversation history injection.  Used for local escalation (e.g. deepseek)."""
    base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    if not model:
        model = get_env("OLLAMA_ESCALATION_MODEL", "")
    if not model:
        return ""  # no escalation model configured
    url = f"{base_url.rstrip('/')}/api/chat"

    messages = [{"role": "user", "content": prompt}]

    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_ctx": 8192,
                    "num_predict": 4096,
                },
            },
            timeout=300,
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""  # model not available, skip escalation

    chunks: list[str] = []
    if show_stream:
        print(DIM, end="", flush=True)
    for line in response.iter_lines():
        if not line:
            continue
        payload = json.loads(line)
        token = payload.get("message", {}).get("content", "")
        if token:
            chunks.append(token)
            if show_stream:
                print(token, end="", flush=True)
        if payload.get("done", False):
            break
    if show_stream:
        print(RESET)

    result = "".join(chunks).strip()

    # Inject into conversation history so the local model knows what deepseek said
    if result:
        add_message("assistant", f"[Local escalation ({model}) responded]: {result}")

    return result


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

_anthropic_client: Anthropic | None = None


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = get_env("ANTHROPIC_API_KEY", required=True)
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def call_anthropic_stream(model: str | None = None,
                          messages: list[dict[str, str]] | None = None,
                          prompt: str | None = None,
                          system_prompt: str | None = None) -> str:
    """Stream an Anthropic response, printing tokens as they arrive in green.

    Accepts a full messages array for multi-turn conversations,
    or a single prompt string for simple one-off calls.
    """
    if not model:
        model = "claude-haiku-4-5"
    client = _get_anthropic_client()

    if messages is None:
        messages = [{"role": "user", "content": prompt or ""}]

    collected: list[str] = []
    print(GREEN, end="", flush=True)
    stream_kwargs: dict = dict(
        model=model,
        max_tokens=4096,
        messages=messages,
    )
    if system_prompt:
        stream_kwargs["system"] = system_prompt
    with client.messages.stream(**stream_kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            collected.append(text)
    print(RESET)
    return "".join(collected).strip()


# ---------------------------------------------------------------------------
# OpenAI (GPT)
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = get_env("OPENAI_API_KEY", required=True)
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_openai_stream(model: str | None = None,
                       messages: list[dict[str, str]] | None = None,
                       prompt: str | None = None,
                       system_prompt: str | None = None) -> str:
    """Stream an OpenAI Chat Completions response in blue.

    Accepts a full messages array for multi-turn conversations,
    or a single prompt string for simple one-off calls.
    """
    if not model:
        model = "gpt-5-mini"
    client = _get_openai_client()

    if messages is None:
        messages = [{"role": "user", "content": prompt or ""}]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    collected: list[str] = []
    print(BLUE, end="", flush=True)
    stream = client.chat.completions.create(
        model=model,
        max_completion_tokens=4096,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            collected.append(delta.content)
    print(RESET)
    return "".join(collected).strip()


def call_openai_responses_stream(model: str | None = None,
                                 messages: list[dict[str, str]] | None = None,
                                 prompt: str | None = None,
                                 system_prompt: str | None = None) -> str:
    """Stream an OpenAI Responses API response (for models like gpt-5.2-pro
    that don't support Chat Completions). Prints tokens in blue.

    Accepts a messages array (preferred) or a single prompt string.
    The Responses API accepts the same [{role, content}] format as Chat Completions.
    """
    if not model:
        model = "gpt-5.2-pro"
    client = _get_openai_client()

    if messages is not None:
        api_input = messages
    else:
        api_input = prompt or ""

    if system_prompt and isinstance(api_input, list):
        api_input = [{"role": "system", "content": system_prompt}] + api_input

    collected: list[str] = []
    print(BLUE, end="", flush=True)
    stream = client.responses.create(
        model=model,
        input=api_input,
        stream=True,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
            collected.append(event.delta)
    print(RESET)
    return "".join(collected).strip()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def unload_model() -> None:
    """Unload the Ollama model on exit to free memory and clear KV cache."""
    try:
        base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
        model = get_env("OLLAMA_MODEL", "llama3.1:8b")
        requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
        print(f"\n{DIM}[Model unloaded]{RESET}")
    except requests.RequestException:
        pass
