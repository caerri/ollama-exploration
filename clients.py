"""API client functions for Ollama, Anthropic, and OpenAI.

Each function handles streaming I/O (printing tokens as they arrive)
and returns the collected response text.
"""

from __future__ import annotations

import json
import re
from contextlib import nullcontext

import requests
from anthropic import Anthropic
from openai import OpenAI

from config import get_env, DIM, RESET
from render import create_streaming_renderer
from prompts import build_system_prompt
from conversation import conversation_history, add_message, _trim_history


# ---------------------------------------------------------------------------
# Ollama (local models)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Strip DeepSeek R1 <think>...</think> reasoning blocks from output."""
    return _THINK_RE.sub("", text).strip()


def get_ollama_models() -> dict[str, str]:
    """Query Ollama for installed models and build a shortcut map.

    Returns a dict like {"qwen": "qwen2.5:7b-instruct", "gemma": "gemma2:9b", ...}
    where keys are short trigger names derived from the model name.
    """
    base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        return {}

    shortcuts: dict[str, str] = {}
    for model_info in resp.json().get("models", []):
        full_name = model_info["name"]          # e.g. "mistral-nemo:12b"
        base = full_name.split(":")[0]          # e.g. "mistral-nemo"

        # Add the full base as a shortcut: "mistral-nemo" → "mistral-nemo:12b"
        shortcuts[base] = full_name

        # Add the first word as a shortcut: "mistral" → "mistral-nemo:12b"
        # But only if it doesn't collide with an existing (longer) name
        first_word = base.split("-")[0]
        if first_word not in shortcuts:
            shortcuts[first_word] = full_name

        # Also handle dots: "llama3.1" → first word "llama3" and "llama"
        if "." in first_word:
            prefix = first_word.split(".")[0]   # "llama3"
            if prefix not in shortcuts:
                shortcuts[prefix] = full_name
            # Strip trailing digits from the prefix too: "llama3" → "llama"
            clean_prefix = prefix.rstrip("0123456789")
            if clean_prefix and clean_prefix != prefix and clean_prefix not in shortcuts:
                shortcuts[clean_prefix] = full_name
        else:
            # Strip trailing digits for a clean shortcut: "gemma2" → "gemma"
            clean = first_word.rstrip("0123456789")
            if clean and clean != first_word and clean not in shortcuts:
                shortcuts[clean] = full_name

    return shortcuts

def call_ollama(prompt: str, show_stream: bool = False,
                system_prompt: str | None = None,
                model: str | None = None) -> str:
    """Send a message to Ollama using the chat API (supports conversation history).

    If show_stream is True, tokens are printed to stdout as they arrive so the
    user can see the model "thinking".  Pass system_prompt to override the
    default base prompt (e.g. with context-aware prompt).
    Pass model to override the default OLLAMA_MODEL (e.g. for auto-escalation).
    """
    base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    if not model:
        model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    url = f"{base_url.rstrip('/')}/api/chat"

    # Add current message to history
    conversation_history.append({"role": "user", "content": prompt})

    effective_prompt = system_prompt or build_system_prompt()
    messages = [{"role": "system", "content": effective_prompt}] + conversation_history

    response = requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "format": "json",
            "stream": True,
            "options": {
                "num_ctx": 16384,
                "num_predict": 4096,
            },
        },
        timeout=300,
        stream=True,
    )
    response.raise_for_status()

    # Collect streamed tokens
    chunks: list[str] = []
    in_thinking = False
    if show_stream:
        print(DIM, end="", flush=True)  # dim the thinking stream
    for line in response.iter_lines():
        if not line:
            continue
        payload = json.loads(line)
        token = payload.get("message", {}).get("content", "")
        if token:
            chunks.append(token)
            # Suppress <think>...</think> blocks from streaming display
            if "<think>" in token:
                in_thinking = True
            if show_stream and not in_thinking:
                print(token, end="", flush=True)
            if "</think>" in token:
                in_thinking = False
        if payload.get("done", False):
            break
    if show_stream:
        print(RESET)  # end dim + newline

    assistant_content = _strip_thinking("".join(chunks))

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
                    "num_ctx": 16384,
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
    in_thinking = False

    if show_stream:
        live, add_token, finish = create_streaming_renderer()
        display_ctx = live
    else:
        display_ctx = nullcontext()
        add_token = lambda t: None  # noqa: E731

    with display_ctx:
        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            token = payload.get("message", {}).get("content", "")
            if token:
                chunks.append(token)
                if "<think>" in token:
                    in_thinking = True
                if not in_thinking:
                    add_token(token)
                if "</think>" in token:
                    in_thinking = False
            if payload.get("done", False):
                break
    if show_stream:
        finish()

    result = _strip_thinking("".join(chunks))

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

    stream_kwargs: dict = dict(
        model=model,
        max_tokens=8192,
        messages=messages,
    )
    if system_prompt:
        stream_kwargs["system"] = system_prompt

    live, add_token, finish = create_streaming_renderer()
    with live:
        with client.messages.stream(**stream_kwargs) as stream:
            for text in stream.text_stream:
                add_token(text)
    result = finish()
    return result


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

    live, add_token, finish = create_streaming_renderer()
    with live:
        stream = client.chat.completions.create(
            model=model,
            max_completion_tokens=8192,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                add_token(delta.content)
    result = finish()
    return result


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

    live, add_token, finish = create_streaming_renderer()
    with live:
        stream = client.responses.create(
            model=model,
            input=api_input,
            stream=True,
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                add_token(event.delta)
    result = finish()
    return result


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def unload_model() -> None:
    """Unload the Ollama model on exit to free memory and clear KV cache."""
    try:
        base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
        model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
        print(f"\n{DIM}[Model unloaded]{RESET}")
    except requests.RequestException:
        pass
