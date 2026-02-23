"""Test suite for local-only functionality — no remote API calls, no money spent.

Tests cover:
  1. parse_local_response() — JSON parsing, fallback regex, key normalization, validation
  2. build_conversation_digest() — plain-text digest for escalation models
  3. build_remote_messages() — message building, JSON stripping, routing word detection
  4. Conversation history management — add_message, clear_history, trimming
  5. get_ollama_models() — shortcut generation from Ollama API response
  6. Config sanity — MODEL_MAP, MODEL_SHORTCUTS, COST_INFO consistency
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from config import (
    MODEL_MAP, MODEL_PROVIDER, MODEL_SHORTCUTS, COST_INFO,
    VALID_NEXT_STEPS, MAX_HISTORY,
    BOLD, RESET, DIM, CYAN, GREEN, YELLOW, RED,
)
from conversation import (
    conversation_history, add_message, clear_history,
    build_conversation_digest, build_remote_messages,
    parse_local_response,
)
from clients import get_ollama_models


# ======================================================================
# Helpers
# ======================================================================

def _reset():
    """Clear conversation history before each logical test group."""
    clear_history()


def _pass(label: str):
    print(f"  {GREEN}✓{RESET} {label}")


def _fail(label: str, detail: str = ""):
    msg = f"  {RED}✗{RESET} {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)


def _section(title: str):
    print(f"\n{BOLD}{CYAN}{title}{RESET}")


# ======================================================================
# 1. parse_local_response()
# ======================================================================

def test_parse_local_response():
    _section("1. parse_local_response()")
    passed = 0
    failed = 0

    # --- 1a. Valid JSON, lowercase keys ---
    raw = json.dumps({
        "analysis": "user said hi",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Hey there!"
    })
    try:
        parsed = parse_local_response(raw)
        assert parsed["ANALYSIS"] == "user said hi"
        assert parsed["SENSITIVE_DATA"] == "NO"
        assert parsed["NEXT_STEP"] == "RESPOND_LOCALLY"
        assert parsed["MODEL"] == "NONE"
        assert parsed["OUTPUT"] == "Hey there!"
        _pass("Valid JSON with lowercase keys"); passed += 1
    except Exception as e:
        _fail("Valid JSON with lowercase keys", str(e)); failed += 1

    # --- 1b. Valid JSON, uppercase keys ---
    raw = json.dumps({
        "ANALYSIS": "checking caps",
        "SENSITIVE_DATA": "NO",
        "NEXT_STEP": "SEND_TO_REMOTE",
        "MODEL": "HAIKU",
        "OUTPUT": "sending it out"
    })
    try:
        parsed = parse_local_response(raw)
        assert parsed["NEXT_STEP"] == "SEND_TO_REMOTE"
        assert parsed["MODEL"] == "HAIKU"
        _pass("Valid JSON with uppercase keys"); passed += 1
    except Exception as e:
        _fail("Valid JSON with uppercase keys", str(e)); failed += 1

    # --- 1c. JSON embedded in surrounding text (fallback regex) ---
    raw = 'Sure, here is my response:\n' + json.dumps({
        "analysis": "embedded json",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "extracted!"
    }) + '\nHope that helps!'
    try:
        parsed = parse_local_response(raw)
        assert parsed["OUTPUT"] == "extracted!"
        _pass("Fallback regex extracts embedded JSON"); passed += 1
    except Exception as e:
        _fail("Fallback regex extracts embedded JSON", str(e)); failed += 1

    # --- 1d. Invalid next_step defaults to RESPOND_LOCALLY ---
    raw = json.dumps({
        "analysis": "bad routing",
        "sensitive_data": "NO",
        "next_step": "YEET_TO_MARS",
        "model": "NONE",
        "output": "should default"
    })
    try:
        parsed = parse_local_response(raw)
        assert parsed["NEXT_STEP"] == "RESPOND_LOCALLY", f"got {parsed['NEXT_STEP']}"
        _pass("Invalid next_step defaults to RESPOND_LOCALLY"); passed += 1
    except Exception as e:
        _fail("Invalid next_step defaults to RESPOND_LOCALLY", str(e)); failed += 1

    # --- 1e. Missing required fields raises ValueError ---
    raw = json.dumps({"analysis": "incomplete", "output": "missing fields"})
    try:
        parse_local_response(raw)
        _fail("Missing fields should raise ValueError", "no exception raised"); failed += 1
    except ValueError:
        _pass("Missing required fields raises ValueError"); passed += 1
    except Exception as e:
        _fail("Missing fields should raise ValueError", f"wrong exception: {e}"); failed += 1

    # --- 1f. Total garbage raises ValueError ---
    try:
        parse_local_response("lol not json at all bro")
        _fail("Garbage input should raise ValueError", "no exception raised"); failed += 1
    except ValueError:
        _pass("Garbage input raises ValueError"); passed += 1
    except Exception as e:
        _fail("Garbage input should raise ValueError", f"wrong exception: {e}"); failed += 1

    # --- 1g. All three valid next_step values are accepted ---
    for step in VALID_NEXT_STEPS:
        raw = json.dumps({
            "analysis": f"testing {step}",
            "sensitive_data": "NO",
            "next_step": step,
            "model": "HAIKU" if step == "SEND_TO_REMOTE" else "NONE",
            "output": f"step is {step}"
        })
        try:
            parsed = parse_local_response(raw)
            assert parsed["NEXT_STEP"] == step
            _pass(f"next_step={step} accepted"); passed += 1
        except Exception as e:
            _fail(f"next_step={step} accepted", str(e)); failed += 1

    # --- 1h. Whitespace around values is stripped ---
    raw = json.dumps({
        "analysis": "  padded  ",
        "sensitive_data": "  NO  ",
        "next_step": "  RESPOND_LOCALLY  ",
        "model": "  NONE  ",
        "output": "  trimmed  "
    })
    try:
        parsed = parse_local_response(raw)
        assert parsed["ANALYSIS"] == "padded"
        assert parsed["OUTPUT"] == "trimmed"
        assert parsed["NEXT_STEP"] == "RESPOND_LOCALLY"
        _pass("Whitespace around values is stripped"); passed += 1
    except Exception as e:
        _fail("Whitespace around values is stripped", str(e)); failed += 1

    return passed, failed


# ======================================================================
# 2. Conversation history management
# ======================================================================

def test_conversation_history():
    _section("2. Conversation History Management")
    passed = 0
    failed = 0

    # --- 2a. add_message appends correctly ---
    _reset()
    add_message("user", "hello")
    try:
        assert len(conversation_history) == 1
        assert conversation_history[0] == {"role": "user", "content": "hello"}
        _pass("add_message appends correctly"); passed += 1
    except Exception as e:
        _fail("add_message appends correctly", str(e)); failed += 1

    # --- 2b. clear_history wipes everything ---
    add_message("assistant", "hi back")
    clear_history()
    try:
        assert len(conversation_history) == 0
        _pass("clear_history wipes everything"); passed += 1
    except Exception as e:
        _fail("clear_history wipes everything", str(e)); failed += 1

    # --- 2c. Trimming keeps last MAX_HISTORY entries ---
    _reset()
    for i in range(MAX_HISTORY + 10):
        add_message("user", f"message {i}")
    try:
        assert len(conversation_history) == MAX_HISTORY, f"got {len(conversation_history)}"
        # Oldest should be trimmed, newest should be last
        assert conversation_history[-1]["content"] == f"message {MAX_HISTORY + 9}"
        assert conversation_history[0]["content"] == f"message 10"
        _pass(f"Trimming keeps last {MAX_HISTORY} entries"); passed += 1
    except Exception as e:
        _fail(f"Trimming keeps last {MAX_HISTORY} entries", str(e)); failed += 1

    # --- 2d. History is shared across imports ---
    _reset()
    add_message("user", "shared state test")
    from conversation import conversation_history as h2
    try:
        assert h2 is conversation_history
        assert len(h2) == 1
        _pass("History is shared across imports (same reference)"); passed += 1
    except Exception as e:
        _fail("History is shared across imports", str(e)); failed += 1

    _reset()
    return passed, failed


# ======================================================================
# 3. build_conversation_digest()
# ======================================================================

def test_build_conversation_digest():
    _section("3. build_conversation_digest()")
    passed = 0
    failed = 0

    # --- 3a. Empty history returns empty string ---
    _reset()
    try:
        assert build_conversation_digest() == ""
        _pass("Empty history returns empty string"); passed += 1
    except Exception as e:
        _fail("Empty history returns empty string", str(e)); failed += 1

    # --- 3b. User messages are included ---
    _reset()
    add_message("user", "what's a good recipe for pasta?")
    add_message("assistant", json.dumps({
        "analysis": "cooking question",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Try aglio e olio — simple and delicious."
    }))
    try:
        digest = build_conversation_digest()
        assert "User: what's a good recipe for pasta?" in digest
        assert "Local model: Try aglio e olio" in digest
        _pass("User messages + local output extracted into digest"); passed += 1
    except Exception as e:
        _fail("User messages + local output extracted", str(e)); failed += 1

    # --- 3c. Remote model responses are included ---
    _reset()
    add_message("user", "tell me about quantum computing")
    add_message("assistant", "[Remote model (HAIKU) responded]: Quantum computing uses qubits...")
    try:
        digest = build_conversation_digest()
        assert "Remote model:" in digest
        assert "Quantum computing uses qubits" in digest
        _pass("Remote model responses included in digest"); passed += 1
    except Exception as e:
        _fail("Remote model responses included", str(e)); failed += 1

    # --- 3d. SEND_TO_REMOTE output is NOT included as local answer ---
    _reset()
    add_message("user", "what's the latest news?")
    add_message("assistant", json.dumps({
        "analysis": "needs current info",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "HAIKU",
        "output": "what's the latest news?"
    }))
    try:
        digest = build_conversation_digest()
        # The routing JSON with SEND_TO_REMOTE should NOT appear as "Local model: ..."
        assert "Local model: what's the latest news?" not in digest
        _pass("SEND_TO_REMOTE output excluded from digest"); passed += 1
    except Exception as e:
        _fail("SEND_TO_REMOTE output excluded from digest", str(e)); failed += 1

    # --- 3e. max_turns limits output ---
    _reset()
    for i in range(20):
        add_message("user", f"turn {i}")
        add_message("assistant", json.dumps({
            "analysis": f"turn {i}",
            "sensitive_data": "NO",
            "next_step": "RESPOND_LOCALLY",
            "model": "NONE",
            "output": f"response {i}"
        }))
    try:
        digest = build_conversation_digest(max_turns=2)
        # Should only have the last 2 turns (4 messages)
        assert "turn 19" in digest
        assert "turn 18" in digest
        assert "turn 0" not in digest
        _pass("max_turns limits digest to recent turns"); passed += 1
    except Exception as e:
        _fail("max_turns limits digest", str(e)); failed += 1

    # --- 3f. Unparseable assistant messages are skipped gracefully ---
    _reset()
    add_message("user", "test")
    add_message("assistant", "not json at all")
    add_message("user", "follow up")
    try:
        digest = build_conversation_digest()
        assert "User: test" in digest
        assert "User: follow up" in digest
        assert "not json at all" not in digest
        _pass("Unparseable assistant messages skipped"); passed += 1
    except Exception as e:
        _fail("Unparseable assistant messages skipped", str(e)); failed += 1

    # --- 3g. Local escalation responses are included ---
    _reset()
    add_message("user", "hard question")
    add_message("assistant", "[Local escalation (deepseek-r1:14b) responded]: Here's my take...")
    try:
        digest = build_conversation_digest()
        assert "Here's my take" in digest
        _pass("Local escalation responses included"); passed += 1
    except Exception as e:
        _fail("Local escalation responses included", str(e)); failed += 1

    _reset()
    return passed, failed


# ======================================================================
# 4. build_remote_messages()
# ======================================================================

def test_build_remote_messages():
    _section("4. build_remote_messages()")
    passed = 0
    failed = 0

    # --- 4a. Simple conversation produces clean alternating messages ---
    _reset()
    add_message("user", "hey I need help with python")
    add_message("assistant", json.dumps({
        "analysis": "greeting",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Sure, what do you need help with?"
    }))
    add_message("user", "how do I read a CSV file?")
    # Simulate llama's routing JSON for current turn
    add_message("assistant", json.dumps({
        "analysis": "csv question",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "HAIKU",
        "output": "how do I read a CSV file?"
    }))
    try:
        msgs = build_remote_messages("how do I read a CSV file?")
        # Should have: user greeting → assistant answer → user question (current)
        assert len(msgs) == 3, f"expected 3, got {len(msgs)}"
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Sure, what do you need help with?"
        assert msgs[2]["role"] == "user"
        assert "CSV" in msgs[2]["content"] or "csv" in msgs[2]["content"].lower()
        _pass("Simple conversation → clean alternating messages"); passed += 1
    except Exception as e:
        _fail("Simple conversation → clean alternating messages", str(e)); failed += 1

    # --- 4b. Routing JSON is stripped (not sent to remote) ---
    try:
        for msg in msgs:
            assert "next_step" not in msg["content"].lower(), "routing JSON leaked through"
            assert "SEND_TO_REMOTE" not in msg["content"]
        _pass("Routing JSON stripped from remote messages"); passed += 1
    except Exception as e:
        _fail("Routing JSON stripped from remote messages", str(e)); failed += 1

    # --- 4c. Remote model response prefixes are stripped ---
    _reset()
    add_message("user", "what is rust?")
    add_message("assistant", "[Remote model (HAIKU) responded]: Rust is a systems programming language.")
    add_message("user", "how does ownership work?")
    add_message("assistant", json.dumps({
        "analysis": "follow up on rust",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "SONNET",
        "output": "how does ownership work?"
    }))
    try:
        msgs = build_remote_messages("how does ownership work?")
        # Find the assistant message — it should NOT have the [Remote model...] prefix
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(asst_msgs) >= 1
        assert not asst_msgs[0]["content"].startswith("[Remote model")
        assert "Rust is a systems programming language" in asst_msgs[0]["content"]
        _pass("Remote model response prefixes stripped"); passed += 1
    except Exception as e:
        _fail("Remote model response prefixes stripped", str(e)); failed += 1

    # --- 4d. Routing word detection — "yup" finds real question ---
    _reset()
    add_message("user", "can you explain how neural networks learn through backpropagation?")
    add_message("assistant", json.dumps({
        "analysis": "complex ML question",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "SONNET",
        "output": "can you explain how neural networks learn through backpropagation?"
    }))
    # User confirms with "yup"
    try:
        msgs = build_remote_messages("yup")
        last_msg = msgs[-1]
        assert last_msg["role"] == "user"
        # Should contain the real question, not "yup"
        assert "backpropagation" in last_msg["content"].lower()
        _pass("Routing word 'yup' → finds real question"); passed += 1
    except Exception as e:
        _fail("Routing word 'yup' → finds real question", str(e)); failed += 1

    # --- 4e. Real question is NOT replaced by routing detection ---
    _reset()
    add_message("user", "hello")
    add_message("assistant", json.dumps({
        "analysis": "greeting",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Hi!"
    }))
    add_message("user", "explain quantum entanglement in detail please")
    add_message("assistant", json.dumps({
        "analysis": "physics question",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "SONNET",
        "output": "explain quantum entanglement in detail please"
    }))
    try:
        msgs = build_remote_messages("explain quantum entanglement in detail please")
        last_msg = msgs[-1]
        assert "quantum entanglement" in last_msg["content"].lower()
        _pass("Real question preserved (not replaced by routing detection)"); passed += 1
    except Exception as e:
        _fail("Real question preserved", str(e)); failed += 1

    # --- 4f. First message is always user role ---
    _reset()
    # Weird state: history starts with assistant
    add_message("assistant", "[Remote model (HAIKU) responded]: here's some context")
    add_message("user", "thanks, now explain more")
    add_message("assistant", json.dumps({
        "analysis": "follow up",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "HAIKU",
        "output": "explain more"
    }))
    try:
        msgs = build_remote_messages("thanks, now explain more")
        assert msgs[0]["role"] == "user", f"first message role: {msgs[0]['role']}"
        _pass("First message is always user role (API requirement)"); passed += 1
    except Exception as e:
        _fail("First message is always user role", str(e)); failed += 1

    # --- 4g. Consecutive same-role messages are collapsed ---
    _reset()
    add_message("user", "first question")
    add_message("user", "oh wait, also this")  # two user messages in a row
    add_message("assistant", json.dumps({
        "analysis": "double question",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "HAIKU",
        "output": "both questions"
    }))
    try:
        msgs = build_remote_messages("oh wait, also this")
        # Should not have consecutive user messages
        for i in range(len(msgs) - 1):
            assert not (msgs[i]["role"] == msgs[i+1]["role"]), \
                f"consecutive {msgs[i]['role']} messages at index {i} and {i+1}"
        _pass("Consecutive same-role messages collapsed"); passed += 1
    except Exception as e:
        _fail("Consecutive same-role messages collapsed", str(e)); failed += 1

    _reset()
    return passed, failed


# ======================================================================
# 5. get_ollama_models() — shortcut generation
# ======================================================================

def test_get_ollama_models():
    _section("5. get_ollama_models() — Shortcut Generation")
    passed = 0
    failed = 0

    # Mock the Ollama /api/tags response
    fake_models = {
        "models": [
            {"name": "mistral-nemo:12b"},
            {"name": "gemma2:9b"},
            {"name": "llama3.1:8b"},
            {"name": "deepseek-r1:14b"},
            {"name": "qwen2.5:7b-instruct"},
        ]
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_models
    mock_resp.raise_for_status = MagicMock()

    with patch("clients.requests.get", return_value=mock_resp):
        shortcuts = get_ollama_models()

    # --- 5a. Full base names are shortcuts ---
    for base in ["mistral-nemo", "gemma2", "llama3.1", "deepseek-r1", "qwen2.5"]:
        try:
            assert base in shortcuts, f"'{base}' not in shortcuts"
            _pass(f"Full base '{base}' is a shortcut"); passed += 1
        except Exception as e:
            _fail(f"Full base '{base}' is a shortcut", str(e)); failed += 1

    # --- 5b. First-word shortcuts exist ---
    for word in ["mistral", "gemma", "deepseek", "qwen"]:
        try:
            assert word in shortcuts, f"'{word}' not in shortcuts"
            _pass(f"First word '{word}' is a shortcut"); passed += 1
        except Exception as e:
            _fail(f"First word '{word}' is a shortcut", str(e)); failed += 1

    # --- 5c. Digit-stripped shortcuts exist ---
    try:
        assert "gemma" in shortcuts
        assert shortcuts["gemma"] == "gemma2:9b"
        _pass("'gemma' shortcut (digit-stripped from 'gemma2')"); passed += 1
    except Exception as e:
        _fail("'gemma' shortcut", str(e)); failed += 1

    # --- 5d. Dot-split + digit-stripped shortcuts for llama ---
    try:
        assert "llama3" in shortcuts, "'llama3' not found"
        assert "llama" in shortcuts, "'llama' not found"
        assert shortcuts["llama"] == "llama3.1:8b"
        _pass("'llama3' and 'llama' shortcuts from 'llama3.1:8b'"); passed += 1
    except Exception as e:
        _fail("'llama' shortcuts from 'llama3.1:8b'", str(e)); failed += 1

    # --- 5e. No trailing dots in shortcuts ---
    try:
        bad_keys = [k for k in shortcuts if k.endswith(".")]
        assert len(bad_keys) == 0, f"found trailing dots: {bad_keys}"
        _pass("No trailing dots in any shortcut"); passed += 1
    except Exception as e:
        _fail("No trailing dots in shortcuts", str(e)); failed += 1

    # --- 5f. No trailing digits in clean shortcuts (gemma, llama, qwen) ---
    clean_shortcuts = ["gemma", "llama", "qwen"]
    try:
        for s in clean_shortcuts:
            if s in shortcuts:
                assert not s[-1].isdigit(), f"'{s}' ends with digit"
        _pass("Clean shortcuts have no trailing digits"); passed += 1
    except Exception as e:
        _fail("Clean shortcuts have no trailing digits", str(e)); failed += 1

    # --- 5g. Shortcuts map to correct full model names ---
    expected = {
        "mistral-nemo": "mistral-nemo:12b",
        "mistral": "mistral-nemo:12b",
        "gemma2": "gemma2:9b",
        "gemma": "gemma2:9b",
        "llama3.1": "llama3.1:8b",
        "llama3": "llama3.1:8b",
        "llama": "llama3.1:8b",
        "deepseek-r1": "deepseek-r1:14b",
        "deepseek": "deepseek-r1:14b",
    }
    all_match = True
    mismatches = []
    for key, expected_val in expected.items():
        actual = shortcuts.get(key)
        if actual != expected_val:
            all_match = False
            mismatches.append(f"  {key}: expected '{expected_val}', got '{actual}'")
    try:
        assert all_match, "\n".join(mismatches)
        _pass("All shortcuts map to correct full model names"); passed += 1
    except Exception as e:
        _fail("All shortcuts map to correct full model names", str(e)); failed += 1

    # --- 5h. Ollama offline returns empty dict ---
    import requests as _requests
    with patch("clients.requests.get", side_effect=_requests.RequestException("connection refused")):
        try:
            result = get_ollama_models()
            assert result == {}
            _pass("Ollama offline → empty dict (graceful fallback)"); passed += 1
        except Exception as e:
            _fail("Ollama offline → empty dict", str(e)); failed += 1

    return passed, failed


# ======================================================================
# 6. Config sanity checks
# ======================================================================

def test_config_sanity():
    _section("6. Config Sanity Checks")
    passed = 0
    failed = 0

    # --- 6a. Every MODEL_MAP key has a provider ---
    try:
        for key in MODEL_MAP:
            assert key in MODEL_PROVIDER, f"{key} missing from MODEL_PROVIDER"
        _pass("Every MODEL_MAP key has a provider"); passed += 1
    except Exception as e:
        _fail("Every MODEL_MAP key has a provider", str(e)); failed += 1

    # --- 6b. Every MODEL_MAP key has cost info ---
    try:
        for key in MODEL_MAP:
            assert key in COST_INFO, f"{key} missing from COST_INFO"
        _pass("Every MODEL_MAP key has cost info"); passed += 1
    except Exception as e:
        _fail("Every MODEL_MAP key has cost info", str(e)); failed += 1

    # --- 6c. All MODEL_SHORTCUTS resolve to valid MODEL_MAP keys ---
    try:
        for shortcut, canonical in MODEL_SHORTCUTS.items():
            assert canonical in MODEL_MAP, f"shortcut '{shortcut}' → '{canonical}' not in MODEL_MAP"
        _pass("All MODEL_SHORTCUTS resolve to valid MODEL_MAP keys"); passed += 1
    except Exception as e:
        _fail("All MODEL_SHORTCUTS resolve to valid MODEL_MAP keys", str(e)); failed += 1

    # --- 6d. Provider values are valid ---
    try:
        valid_providers = {"anthropic", "openai"}
        for key, provider in MODEL_PROVIDER.items():
            assert provider in valid_providers, f"{key} has invalid provider '{provider}'"
        _pass("All providers are 'anthropic' or 'openai'"); passed += 1
    except Exception as e:
        _fail("Provider values are valid", str(e)); failed += 1

    # --- 6e. VALID_NEXT_STEPS has exactly the expected values ---
    try:
        expected = {"RESPOND_LOCALLY", "SEND_TO_REMOTE", "ASK_USER"}
        assert VALID_NEXT_STEPS == expected, f"got {VALID_NEXT_STEPS}"
        _pass("VALID_NEXT_STEPS has expected values"); passed += 1
    except Exception as e:
        _fail("VALID_NEXT_STEPS has expected values", str(e)); failed += 1

    # --- 6f. MAX_HISTORY is a reasonable number ---
    try:
        assert 10 <= MAX_HISTORY <= 200, f"MAX_HISTORY={MAX_HISTORY} seems off"
        _pass(f"MAX_HISTORY={MAX_HISTORY} is reasonable"); passed += 1
    except Exception as e:
        _fail(f"MAX_HISTORY={MAX_HISTORY} is reasonable", str(e)); failed += 1

    return passed, failed


# ======================================================================
# Main
# ======================================================================

def main():
    print(f"{BOLD}{CYAN}Local Test Suite — No API Calls{RESET}")
    print(f"{DIM}Testing parse, conversation, digest, shortcuts, and config{RESET}")

    total_passed = 0
    total_failed = 0

    for test_fn in [
        test_parse_local_response,
        test_conversation_history,
        test_build_conversation_digest,
        test_build_remote_messages,
        test_get_ollama_models,
        test_config_sanity,
    ]:
        p, f = test_fn()
        total_passed += p
        total_failed += f

    print(f"\n{'='*50}")
    color = GREEN if total_failed == 0 else RED
    print(f"{BOLD}Results: {color}{total_passed} passed{RESET}, ", end="")
    if total_failed:
        print(f"{RED}{total_failed} failed{RESET}")
    else:
        print(f"{GREEN}0 failed{RESET}")
    print(f"{'='*50}")

    _reset()  # clean up


if __name__ == "__main__":
    main()
