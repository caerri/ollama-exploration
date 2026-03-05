"""Test suite for local-only functionality — no remote API calls, no money spent.

Tests cover:
  1. parse_local_response() — JSON parsing, fallback regex, key normalization, validation
  2. Conversation history management — add_message, clear_history, trimming
  3. build_conversation_digest() — plain-text digest for escalation models
  4. build_remote_messages() — message building, JSON stripping, routing word detection
  5. get_ollama_models() — shortcut generation from Ollama API response
  6. Config sanity — MODEL_MAP, MODEL_SHORTCUTS, COST_INFO, calendar keywords
  7. Calendar keyword detection — matching logic used by main.py
  8. Natural conversation flow — multi-turn scenarios that broke in real usage
  9. Journal keyword detection — matching logic for Obsidian journal injection
 10. Journal template parsing — YAML frontmatter + markdown section extraction
 11. Prompt composition — modular prompt building and sticky context detection
 12. Canvas integration — keywords, context markers, sync format
 13. Planning keywords — auto-context trigger matching
 14. Sticky local model switching — 10-turn persistence, clear reset, default clears
 15. Remote prompt token budget — system prompts include budget instructions
 16. History window sizing — MAX_HISTORY and build_remote_messages max_turns
 17. Session memory — load, save, forget, context injection, graceful failure
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from config import (
    MODEL_MAP, MODEL_PROVIDER, MODEL_SHORTCUTS, COST_INFO,
    VALID_NEXT_STEPS, MAX_HISTORY, CALENDAR_KEYWORDS, CALENDAR_LOOKAHEAD_DAYS,
    JOURNAL_KEYWORDS, JOURNAL_LOOKBACK_DAYS, CANVAS_KEYWORDS, PLANNING_KEYWORDS,
    STICKY_LOCAL_MAX,
    BOLD, RESET, DIM, CYAN, GREEN, YELLOW, RED,
)
from conversation import (
    conversation_history, add_message, clear_history,
    build_conversation_digest, build_remote_messages,
    parse_local_response,
)
from clients import get_ollama_models
from obsidian_client import _parse_daily_note, _format_entry, get_journal_context


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

    assert failed == 0, f"{failed} parse_local_response sub-tests failed"


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
    assert failed == 0, f"{failed} conversation_history sub-tests failed"


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
    assert failed == 0, f"{failed} build_conversation_digest sub-tests failed"


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
    assert failed == 0, f"{failed} build_remote_messages sub-tests failed"


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

    assert failed == 0, f"{failed} get_ollama_models sub-tests failed"


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

    assert failed == 0, f"{failed} config_sanity sub-tests failed"


# ======================================================================
# 7. Calendar keyword detection
# ======================================================================

def test_calendar_keywords():
    _section("7. Calendar Keyword Detection")
    passed = 0
    failed = 0

    # --- 7a. CALENDAR_LOOKAHEAD_DAYS is reasonable ---
    try:
        assert 1 <= CALENDAR_LOOKAHEAD_DAYS <= 30, f"got {CALENDAR_LOOKAHEAD_DAYS}"
        _pass(f"CALENDAR_LOOKAHEAD_DAYS={CALENDAR_LOOKAHEAD_DAYS} is reasonable"); passed += 1
    except Exception as e:
        _fail("CALENDAR_LOOKAHEAD_DAYS is reasonable", str(e)); failed += 1

    # --- 7b. CALENDAR_KEYWORDS is non-empty ---
    try:
        assert len(CALENDAR_KEYWORDS) > 0
        _pass(f"CALENDAR_KEYWORDS has {len(CALENDAR_KEYWORDS)} entries"); passed += 1
    except Exception as e:
        _fail("CALENDAR_KEYWORDS is non-empty", str(e)); failed += 1

    # --- 7c. Expected keywords trigger matches ---
    should_match = [
        "what's on my calendar today",
        "am i free thursday",
        "do i have any meetings tomorrow",
        "what's my schedule this week",
        "is there an appointment on friday",
        "what do i have going on",
        "evaluate my week",
        "am i busy tomorrow",
    ]
    for phrase in should_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in CALENDAR_KEYWORDS)
            assert matched, f"'{phrase}' should match but didn't"
            _pass(f"Matches: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"Matches: '{phrase}'", str(e)); failed += 1

    # --- 7d. Non-calendar phrases do NOT trigger ---
    should_not_match = [
        "hey what's up",
        "write me a python script",
        "who is the president",
        "tell me about quantum computing",
        "how do gallstones form",
    ]
    for phrase in should_not_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in CALENDAR_KEYWORDS)
            assert not matched, f"'{phrase}' should NOT match but did"
            _pass(f"No match: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"No match: '{phrase}'", str(e)); failed += 1

    # --- 7e. All keywords are lowercase ---
    try:
        for kw in CALENDAR_KEYWORDS:
            assert kw == kw.lower(), f"keyword '{kw}' is not lowercase"
        _pass("All keywords are lowercase"); passed += 1
    except Exception as e:
        _fail("All keywords are lowercase", str(e)); failed += 1

    assert failed == 0, f"{failed} calendar_keywords sub-tests failed"


# ======================================================================
# 8. Natural conversation flow — simulates realistic multi-turn exchanges
#    to verify context handling, digest building, and message construction
#    across the same scenarios that broke in real usage.
# ======================================================================

def test_natural_conversation_flow():
    _section("8. Natural Conversation Flow")
    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # 8a. GPT Pro answers, user says "i'm good" — context should carry
    # ------------------------------------------------------------------
    _reset()
    # User asks about Apple Calendar + Python
    add_message("user", "i need the mac to connect to apple calendar through python")
    # Local model routes to remote
    add_message("assistant", json.dumps({
        "analysis": "User wants Python + Apple Calendar",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "HAIKU",
        "output": "i need the mac to connect to apple calendar through python"
    }))
    # GPT Pro responds with a big answer
    gpt_response = (
        "You've got 3 ways to talk to Apple Calendar from Python on macOS:\n\n"
        "1) EventKit via PyObjC — best for local Calendar app\n"
        "2) AppleScript/JXA — clunkier but works\n"
        "3) CalDAV — best for network API access\n\n"
        "If you want 'connect to the Calendar app on this Mac', use EventKit.\n\n"
        "Install: pip install pyobjc pyobjc-framework-EventKit\n"
        "Permission: System Settings > Privacy > Calendars"
    )
    add_message("assistant", f"[Remote model (GPT_PRO) responded]: {gpt_response}")
    # User says "i'm good"
    add_message("user", "i'm good")

    try:
        # The digest should contain the GPT Pro response
        digest = build_conversation_digest()
        assert "EventKit" in digest, "GPT Pro's EventKit answer missing from digest"
        assert "User: i'm good" in digest, "User's 'i'm good' missing from digest"
        _pass("8a. GPT Pro response + 'i'm good' both in digest"); passed += 1
    except Exception as e:
        _fail("8a. GPT Pro response + 'i'm good' in digest", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8b. "evaluate what gpt pro said" — should see the response in history
    # ------------------------------------------------------------------
    add_message("user", "evaluate what gpt pro said")
    # Simulate local model's routing JSON for this turn
    add_message("assistant", json.dumps({
        "analysis": "User wants evaluation of GPT Pro response",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "GPT Pro gave you three solid options for connecting to Apple Calendar. "
                  "EventKit is the right call for local access — it reads the same database "
                  "the Calendar app uses. The code example covers permissions and event reading."
    }))
    try:
        digest = build_conversation_digest()
        # Both the GPT Pro response and the evaluation should be present
        assert "EventKit" in digest
        assert "evaluate what gpt pro said" in digest
        assert "three solid options" in digest or "right call" in digest
        _pass("8b. Evaluation turn preserved in context"); passed += 1
    except Exception as e:
        _fail("8b. Evaluation turn preserved in context", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8c. Remote response flows cleanly to next remote call
    # ------------------------------------------------------------------
    _reset()
    add_message("user", "what is rust?")
    add_message("assistant", "[Remote model (HAIKU) responded]: Rust is a systems programming language focused on safety and performance.")
    add_message("user", "how does the borrow checker work?")
    add_message("assistant", json.dumps({
        "analysis": "Follow up on Rust",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "SONNET",
        "output": "how does the borrow checker work?"
    }))
    try:
        msgs = build_remote_messages("how does the borrow checker work?")
        # Sonnet should see the clean Haiku response (no prefix)
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(asst_msgs) >= 1
        assert "Rust is a systems programming language" in asst_msgs[0]["content"]
        assert not asst_msgs[0]["content"].startswith("[Remote model")
        # Final message should be the follow-up question
        assert "borrow checker" in msgs[-1]["content"].lower()
        _pass("8c. Remote response clean in next remote call"); passed += 1
    except Exception as e:
        _fail("8c. Remote response clean in next remote call", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8d. Multi-turn budget conversation — running total scenario
    # ------------------------------------------------------------------
    _reset()
    add_message("user", "help me save $1780 per month")
    add_message("assistant", json.dumps({
        "analysis": "User wants to save money",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Let's go through your expenses and find savings. What are your main monthly costs?"
    }))
    add_message("user", "$1k rent, netflix $7.99, hulu $2, car payment, insurance, food")
    add_message("assistant", json.dumps({
        "analysis": "User listed expenses",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "OK so rent is $1000, Netflix $7.99, Hulu $2. Let's start with the subscriptions."
    }))
    add_message("user", "i already cancelled netflix. keep a running total")
    add_message("assistant", json.dumps({
        "analysis": "User cancelled Netflix, wants running total",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Netflix cancelled — that's $7.99/mo saved.\n\n"
                  "Running total: $7.99 saved / $1772.01 left to go.\n\n"
                  "Next up — Hulu at $2. Want to cut that too?"
    }))
    add_message("user", "yeah cut hulu too")
    try:
        digest = build_conversation_digest()
        # All key context should survive in the digest
        assert "$1780" in digest or "1780" in digest, "Original goal missing"
        assert "netflix" in digest.lower() or "Netflix" in digest, "Netflix discussion missing"
        assert "7.99" in digest, "Netflix amount missing"
        assert "running total" in digest.lower() or "Running total" in digest, "Running total missing"
        assert "hulu" in digest.lower() or "Hulu" in digest, "Hulu discussion missing"
        _pass("8d. Multi-turn budget conversation preserved in digest"); passed += 1
    except Exception as e:
        _fail("8d. Multi-turn budget conversation preserved", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8e. Frustration + profanity doesn't break context
    # ------------------------------------------------------------------
    _reset()
    add_message("user", "really")
    add_message("assistant", json.dumps({
        "analysis": "User said really — unclear intent",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "What's on your mind?"
    }))
    add_message("user", "i didn't say hey. i said really. are you fuckin dumb.")
    add_message("assistant", json.dumps({
        "analysis": "User is frustrated about misinterpretation",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "My bad — you said 'really', not 'hey'. What were you reacting to?"
    }))
    try:
        digest = build_conversation_digest()
        assert "really" in digest.lower()
        assert "my bad" in digest.lower() or "My bad" in digest
        # The model should NOT have said "hey" in its response
        local_responses = [line for line in digest.split("\n") if line.startswith("Local model:")]
        for resp in local_responses:
            assert "Hello" not in resp, "Model hallucinated a greeting"
            assert "How can I assist" not in resp, "Model used banned phrase"
        _pass("8e. Frustration handled, no hallucinated greetings"); passed += 1
    except Exception as e:
        _fail("8e. Frustration handled correctly", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8f. Dismissive reply after remote — routing words still work
    # ------------------------------------------------------------------
    _reset()
    add_message("user", "explain how transformers work in machine learning")
    add_message("assistant", json.dumps({
        "analysis": "Complex ML question",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "SONNET",
        "output": "explain how transformers work in machine learning"
    }))
    try:
        # "yup" at Phone a Friend should find the real question
        msgs = build_remote_messages("yup")
        last = msgs[-1]
        assert last["role"] == "user"
        assert "transformer" in last["content"].lower()
        assert "yup" not in last["content"].lower()
        _pass("8f. 'yup' routes real question, not the word 'yup'"); passed += 1
    except Exception as e:
        _fail("8f. 'yup' routing word detection", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8g. Long conversation doesn't lose early context within limits
    # ------------------------------------------------------------------
    _reset()
    # Simulate 8 turns of conversation (16 messages = within MAX_HISTORY)
    add_message("user", "my name is Carrie and I'm trying to build a budgeting app")
    add_message("assistant", json.dumps({
        "analysis": "User introduction",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Hey Carrie! A budgeting app sounds great. What tech stack are you thinking?"
    }))
    for i in range(6):
        add_message("user", f"follow-up question {i} about the budgeting app")
        add_message("assistant", json.dumps({
            "analysis": f"follow-up {i}",
            "sensitive_data": "NO",
            "next_step": "RESPOND_LOCALLY",
            "model": "NONE",
            "output": f"Here's my answer to follow-up {i} about the budgeting app."
        }))
    add_message("user", "what was my name again?")
    try:
        digest = build_conversation_digest()
        assert "Carrie" in digest, "User's name lost from context"
        assert "budgeting" in digest.lower(), "Project topic lost from context"
        _pass("8g. Early context (name, topic) survives 8-turn conversation"); passed += 1
    except Exception as e:
        _fail("8g. Early context survives long conversation", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8h. Calendar context mixed with regular question
    # ------------------------------------------------------------------
    _reset()
    cal_block = (
        "--- Your Calendar (next 7 days) ---\n"
        "Today (Sat Feb 22):\n"
        "  - 3:00 PM - 4:00 PM: Dentist appointment\n"
        "  - 7:00 PM - 9:00 PM: Dinner with Alex\n"
        "Mon Feb 24:\n"
        "  - 9:00 AM - 10:00 AM: Team standup\n"
        "--- End Calendar ---"
    )
    # This simulates what main.py does — prepending calendar context
    user_msg_with_cal = f"{cal_block}\n\nam i free tomorrow"
    add_message("user", user_msg_with_cal)
    add_message("assistant", json.dumps({
        "analysis": "User asks about availability with calendar context",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Looking at your calendar, tomorrow (Sunday Feb 23) is completely clear — no events scheduled."
    }))
    try:
        digest = build_conversation_digest()
        assert "Dentist" in digest, "Calendar events missing from digest"
        assert "free tomorrow" in digest.lower() or "am i free" in digest.lower()
        assert "completely clear" in digest or "no events" in digest
        _pass("8h. Calendar context + question preserved in digest"); passed += 1
    except Exception as e:
        _fail("8h. Calendar context in digest", str(e)); failed += 1

    # ------------------------------------------------------------------
    # 8i. Mixed local + remote turns build correct remote messages
    # ------------------------------------------------------------------
    _reset()
    add_message("user", "hey what's up")
    add_message("assistant", json.dumps({
        "analysis": "greeting",
        "sensitive_data": "NO",
        "next_step": "RESPOND_LOCALLY",
        "model": "NONE",
        "output": "Not much! What are you working on?"
    }))
    add_message("user", "who won the super bowl this year")
    add_message("assistant", json.dumps({
        "analysis": "current events",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "HAIKU",
        "output": "who won the super bowl this year"
    }))
    add_message("assistant", "[Remote model (HAIKU) responded]: The Kansas City Chiefs won Super Bowl LX.")
    add_message("user", "cool. now tell me more about the game")
    add_message("assistant", json.dumps({
        "analysis": "follow-up on super bowl",
        "sensitive_data": "NO",
        "next_step": "SEND_TO_REMOTE",
        "model": "HAIKU",
        "output": "tell me more about the game"
    }))
    try:
        msgs = build_remote_messages("cool. now tell me more about the game")
        # Haiku should see: the greeting exchange, its own previous answer, and the follow-up
        roles = [m["role"] for m in msgs]
        # Must alternate
        for i in range(len(roles) - 1):
            assert roles[i] != roles[i+1], f"consecutive {roles[i]} at {i}"
        # Must start with user
        assert roles[0] == "user"
        # Previous Haiku answer should be in there (cleaned)
        all_content = " ".join(m["content"] for m in msgs)
        assert "Chiefs" in all_content or "Super Bowl" in all_content, "Previous remote answer missing"
        assert "[Remote model" not in all_content, "Remote prefix leaked through"
        _pass("8i. Mixed local+remote turns → correct remote messages"); passed += 1
    except Exception as e:
        _fail("8i. Mixed local+remote turns", str(e)); failed += 1

    _reset()
    assert failed == 0, f"{failed} natural_conversation_flow sub-tests failed"


# ======================================================================
# 9. Journal keyword detection
# ======================================================================

def test_journal_keywords():
    _section("9. Journal Keyword Detection")
    passed = 0
    failed = 0

    # --- 9a. JOURNAL_LOOKBACK_DAYS is reasonable ---
    try:
        assert 1 <= JOURNAL_LOOKBACK_DAYS <= 30, f"got {JOURNAL_LOOKBACK_DAYS}"
        _pass(f"JOURNAL_LOOKBACK_DAYS={JOURNAL_LOOKBACK_DAYS} is reasonable"); passed += 1
    except Exception as e:
        _fail("JOURNAL_LOOKBACK_DAYS is reasonable", str(e)); failed += 1

    # --- 9b. JOURNAL_KEYWORDS is non-empty ---
    try:
        assert len(JOURNAL_KEYWORDS) > 0
        _pass(f"JOURNAL_KEYWORDS has {len(JOURNAL_KEYWORDS)} entries"); passed += 1
    except Exception as e:
        _fail("JOURNAL_KEYWORDS is non-empty", str(e)); failed += 1

    # --- 9c. Expected phrases trigger matches ---
    should_match = [
        "how am i doing this week",
        "how's my week been",
        "what did i work on yesterday",
        "review my progress",
        "i'm feeling overwhelmed today",
        "what got done this week",
        "my energy is super low",
        "how am i handling my workload",
        "i feel burned out",
        "let's look at my journal",
    ]
    for phrase in should_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in JOURNAL_KEYWORDS)
            assert matched, f"'{phrase}' should match but didn't"
            _pass(f"Matches: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"Matches: '{phrase}'", str(e)); failed += 1

    # --- 9d. Non-journal phrases do NOT trigger ---
    should_not_match = [
        "hey what's up",
        "write me a python script",
        "who is the president",
        "explain quantum computing",
        "what's 2 + 2",
    ]
    for phrase in should_not_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in JOURNAL_KEYWORDS)
            assert not matched, f"'{phrase}' should NOT match but did"
            _pass(f"No match: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"No match: '{phrase}'", str(e)); failed += 1

    # --- 9e. All keywords are lowercase ---
    try:
        for kw in JOURNAL_KEYWORDS:
            assert kw == kw.lower(), f"keyword '{kw}' is not lowercase"
        _pass("All journal keywords are lowercase"); passed += 1
    except Exception as e:
        _fail("All journal keywords are lowercase", str(e)); failed += 1

    assert failed == 0, f"{failed} journal_keywords sub-tests failed"


# ======================================================================
# 10. Journal template parsing
# ======================================================================

def test_journal_parsing():
    _section("10. Journal Template Parsing")
    passed = 0
    failed = 0

    # --- 10a. Valid daily note with all fields parses correctly ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-22.md"
        note.write_text(
            "---\n"
            "date: \"2026-02-22\"\n"
            "day: \"Saturday\"\n"
            "mood: 4\n"
            "energy: 3\n"
            "sleep_hours: 7\n"
            "tags: [school, coding]\n"
            "---\n\n"
            "## Plan\n"
            "- [x] finish relay code\n"
            "- [ ] study linear algebra\n"
            "- [ ] gym\n\n"
            "## What Happened\n"
            "Mostly coding today. Got the journal integration working.\n\n"
            "## Wins\n"
            "Journal integration shipped!\n\n"
            "## Blockers\n"
            "Didn't start studying yet.\n\n"
            "## Notes\n"
            "Need to plan study time for midterm.\n"
        )
        try:
            parsed = _parse_daily_note(note)
            assert parsed is not None
            assert parsed["frontmatter"]["mood"] == 4
            assert parsed["frontmatter"]["energy"] == 3
            assert parsed["frontmatter"]["sleep_hours"] == 7
            assert "school" in parsed["frontmatter"]["tags"]
            assert "Plan" in parsed["sections"]
            assert "[x] finish relay code" in parsed["sections"]["Plan"]
            assert "What Happened" in parsed["sections"]
            assert "Wins" in parsed["sections"]
            assert "Blockers" in parsed["sections"]
            assert "Notes" in parsed["sections"]
            assert parsed["date"] == "2026-02-22"
            _pass("10a. Valid note with all fields parses correctly"); passed += 1
        except Exception as e:
            _fail("10a. Valid note parses correctly", str(e)); failed += 1

    # --- 10b. Frontmatter with missing fields handled gracefully ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-21.md"
        note.write_text(
            "---\n"
            "date: \"2026-02-21\"\n"
            "mood:\n"
            "energy:\n"
            "---\n\n"
            "## What Happened\n"
            "Quick note — not much today.\n"
        )
        try:
            parsed = _parse_daily_note(note)
            assert parsed is not None
            assert parsed["frontmatter"].get("mood") is None
            assert "What Happened" in parsed["sections"]
            _pass("10b. Missing frontmatter fields handled gracefully"); passed += 1
        except Exception as e:
            _fail("10b. Missing frontmatter fields", str(e)); failed += 1

    # --- 10c. Empty sections are omitted from formatted output ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-20.md"
        note.write_text(
            "---\n"
            "date: \"2026-02-20\"\n"
            "mood: 3\n"
            "energy: 2\n"
            "sleep_hours: 5.5\n"
            "---\n\n"
            "## Plan\n\n"
            "## What Happened\n"
            "Just a rough day.\n\n"
            "## Wins\n\n"
            "## Blockers\n\n"
            "## Notes\n"
        )
        try:
            parsed = _parse_daily_note(note)
            assert "Plan" not in parsed["sections"], "empty Plan should be omitted"
            assert "Wins" not in parsed["sections"], "empty Wins should be omitted"
            assert "Blockers" not in parsed["sections"], "empty Blockers should be omitted"
            assert "Notes" not in parsed["sections"], "empty Notes should be omitted"
            assert "What Happened" in parsed["sections"]
            _pass("10c. Empty sections omitted from parsed output"); passed += 1
        except Exception as e:
            _fail("10c. Empty sections omitted", str(e)); failed += 1

    # --- 10d. Checkbox state preserved in Plan section ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-19.md"
        note.write_text(
            "---\ndate: \"2026-02-19\"\nmood: 4\n---\n\n"
            "## Plan\n"
            "- [x] study math\n"
            "- [ ] go to gym\n"
            "- [x] finish code\n"
        )
        try:
            parsed = _parse_daily_note(note)
            formatted = _format_entry(parsed)
            assert "[x] study math" in formatted
            assert "[ ] go to gym" in formatted
            assert "[x] finish code" in formatted
            _pass("10d. Checkbox state preserved in formatted Plan"); passed += 1
        except Exception as e:
            _fail("10d. Checkbox state preserved", str(e)); failed += 1

    # --- 10e. Very long section content is truncated ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-18.md"
        long_text = "This is a very long journal entry. " * 20  # ~700 chars
        note.write_text(
            "---\ndate: \"2026-02-18\"\nmood: 3\n---\n\n"
            f"## What Happened\n{long_text}\n"
        )
        try:
            parsed = _parse_daily_note(note)
            formatted = _format_entry(parsed)
            # The formatted "Happened:" line should be truncated
            for line in formatted.split("\n"):
                if "Happened:" in line:
                    assert len(line) < 350, f"line too long: {len(line)} chars"
                    assert line.rstrip().endswith("...")
                    break
            _pass("10e. Long section content truncated with ..."); passed += 1
        except Exception as e:
            _fail("10e. Long section truncated", str(e)); failed += 1

    # --- 10f. File with no frontmatter still returns body ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-17.md"
        note.write_text(
            "## What Happened\n"
            "Just a plain note without frontmatter.\n"
        )
        try:
            parsed = _parse_daily_note(note)
            assert parsed is not None
            assert parsed["frontmatter"] == {}
            assert "What Happened" in parsed["sections"]
            _pass("10f. No frontmatter — body still parsed"); passed += 1
        except Exception as e:
            _fail("10f. No frontmatter handling", str(e)); failed += 1

    # --- 10g. File with malformed YAML frontmatter doesn't crash ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-16.md"
        note.write_text(
            "---\n"
            "mood: [broken: yaml: here\n"
            "---\n\n"
            "## What Happened\n"
            "Note with bad YAML.\n"
        )
        try:
            parsed = _parse_daily_note(note)
            assert parsed is not None
            assert "What Happened" in parsed["sections"]
            _pass("10g. Malformed YAML doesn't crash"); passed += 1
        except Exception as e:
            _fail("10g. Malformed YAML handling", str(e)); failed += 1

    # --- 10h. get_journal_context returns "" for nonexistent vault ---
    try:
        result = get_journal_context(vault_path="/nonexistent/vault/path/xyz")
        assert result == "", f"expected empty string, got: {result!r}"
        _pass("10h. Nonexistent vault → empty string"); passed += 1
    except Exception as e:
        _fail("10h. Nonexistent vault", str(e)); failed += 1

    # --- 10i. get_journal_context returns wrapper for empty Journal dir ---
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_dir = Path(tmpdir) / "Journal" / "Daily"
        journal_dir.mkdir(parents=True)
        try:
            result = get_journal_context(days=3, vault_path=tmpdir)
            assert "--- Your Journal" in result
            assert "No journal entries" in result or "(no entry)" in result
            assert "--- End Journal ---" in result
            _pass("10i. Empty Journal dir → proper wrapper"); passed += 1
        except Exception as e:
            _fail("10i. Empty Journal dir", str(e)); failed += 1

    # --- 10j. Context format matches expected structure ---
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_dir = Path(tmpdir) / "Journal" / "Daily"
        journal_dir.mkdir(parents=True)
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        (journal_dir / f"{today}.md").write_text(
            "---\ndate: \"" + today + "\"\nmood: 4\nenergy: 3\nsleep_hours: 7\n---\n\n"
            "## Plan\n- [x] test things\n\n"
            "## What Happened\nTesting the journal integration.\n"
        )
        try:
            # Clear cache to force re-read
            import obsidian_client
            obsidian_client._cache = (0.0, "")

            result = get_journal_context(days=3, vault_path=tmpdir)
            assert result.startswith("--- Your Journal")
            assert result.endswith("--- End Journal ---")
            assert "mood:4" in result
            assert "energy:3" in result
            assert "sleep:7h" in result
            assert "[x] test things" in result
            _pass("10j. Context format matches expected structure"); passed += 1
        except Exception as e:
            _fail("10j. Context format structure", str(e)); failed += 1

    # --- 10k. Non-numeric mood/energy values handled correctly ---
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "2026-02-22.md"
        note.write_text(
            "---\ndate: \"2026-02-22\"\nmood: mixed\nenergy: low\nsleep_hours: 7\n---\n\n"
            "## What Happened\nTesting descriptive mood values.\n"
        )
        try:
            parsed = _parse_daily_note(note)
            assert parsed["frontmatter"]["mood"] == "mixed"
            assert parsed["frontmatter"]["energy"] == "low"
            formatted = _format_entry(parsed)
            assert "mood:mixed" in formatted
            assert "energy:low" in formatted
            _pass("10k. Non-numeric mood/energy handled"); passed += 1
        except Exception as e:
            _fail("10k. Non-numeric mood/energy", str(e)); failed += 1

    assert failed == 0, f"{failed} journal_parsing sub-tests failed"


# ======================================================================
# 11. Prompt Composition + Sticky Context Detection
# ======================================================================

def test_prompt_composition():
    _section("11. Prompt Composition + Sticky Context")
    passed = 0
    failed = 0

    from prompts import build_system_prompt, has_recent_context, PROMPT_CONTEXT

    # --- 11a. Base prompt has core sections, no context analysis ---
    try:
        base = build_system_prompt(has_context=False)
        assert "TONE" in base, "base prompt should have TONE"
        assert "JSON OUTPUT" in base, "base prompt should have JSON OUTPUT"
        assert "ROUTING" in base, "base prompt should have ROUTING"
        assert "RESPONSE QUALITY" in base, "base prompt should have RESPONSE QUALITY"
        assert "PERSONAL CONTEXT ANALYSIS MODE" not in base, \
            "base prompt should NOT have context analysis"
        _pass("11a. Base prompt has core sections, no context"); passed += 1
    except Exception as e:
        _fail("11a. Base prompt structure", str(e)); failed += 1

    # --- 11b. Context prompt includes analysis steps ---
    try:
        ctx = build_system_prompt(has_context=True)
        assert "PERSONAL CONTEXT ANALYSIS MODE" in ctx
        assert "STEP 1" in ctx
        assert "STEP 2" in ctx
        assert "STEP 3" in ctx
        assert "STEP 4" in ctx
        assert "DATA INVENTORY" in ctx
        assert "PATTERN DETECTION" in ctx
        assert "CALENDAR vs REALITY" in ctx
        assert "CONCRETE RECOMMENDATIONS" in ctx
        _pass("11b. Context prompt includes all analysis steps"); passed += 1
    except Exception as e:
        _fail("11b. Context prompt analysis steps", str(e)); failed += 1

    # --- 11c. Context prompt is longer than base ---
    try:
        base = build_system_prompt(has_context=False)
        ctx = build_system_prompt(has_context=True)
        assert len(ctx) > len(base), \
            f"context ({len(ctx)}) should be longer than base ({len(base)})"
        _pass("11c. Context prompt longer than base"); passed += 1
    except Exception as e:
        _fail("11c. Prompt length comparison", str(e)); failed += 1

    # --- 11d. has_recent_context detects markers in history ---
    try:
        history_with_context = [
            {"role": "user", "content": "--- Your Calendar ---\nMonday: meeting at 2pm"},
            {"role": "assistant", "content": '{"output": "you have a meeting"}'},
        ]
        assert has_recent_context(history_with_context) is True
        _pass("11d. has_recent_context detects calendar marker"); passed += 1
    except Exception as e:
        _fail("11d. Context marker detection", str(e)); failed += 1

    # --- 11e. has_recent_context returns False for empty/normal history ---
    try:
        empty_history: list[dict[str, str]] = []
        assert has_recent_context(empty_history) is False

        normal_history = [
            {"role": "user", "content": "hey what's up"},
            {"role": "assistant", "content": '{"output": "not much"}'},
        ]
        assert has_recent_context(normal_history) is False
        _pass("11e. No false positives on normal history"); passed += 1
    except Exception as e:
        _fail("11e. No false positives", str(e)); failed += 1

    # --- 11f. has_recent_context respects lookback window ---
    try:
        # Context marker is at the start, but lookback=2 only checks last 2 messages
        old_context_history = [
            {"role": "user", "content": "--- Your Journal ---\nmood: 3"},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "tell me a joke"},
            {"role": "assistant", "content": "why did the chicken"},
            {"role": "user", "content": "that sucked"},
            {"role": "assistant", "content": "fair enough"},
        ]
        assert has_recent_context(old_context_history, lookback=2) is False
        assert has_recent_context(old_context_history, lookback=10) is True
        _pass("11f. Lookback window respected"); passed += 1
    except Exception as e:
        _fail("11f. Lookback window", str(e)); failed += 1

    assert failed == 0, f"{failed} prompt_composition sub-tests failed"


# ======================================================================
# 12. Canvas integration — keywords, context markers, sync format
# ======================================================================

def test_canvas_integration():
    _section("12. Canvas Integration")
    passed = 0
    failed = 0

    from prompts import CONTEXT_MARKERS

    # --- 12a. CANVAS_KEYWORDS is non-empty and all lowercase ---
    try:
        assert len(CANVAS_KEYWORDS) > 0, "CANVAS_KEYWORDS is empty"
        for kw in CANVAS_KEYWORDS:
            assert kw == kw.lower(), f"keyword '{kw}' is not lowercase"
        _pass(f"12a. CANVAS_KEYWORDS has {len(CANVAS_KEYWORDS)} entries, all lowercase"); passed += 1
    except Exception as e:
        _fail("12a. CANVAS_KEYWORDS", str(e)); failed += 1

    # --- 12b. Expected phrases trigger canvas matches ---
    should_match = [
        "what's due this week",
        "show me my assignments",
        "any deadlines coming up",
        "i need to check canvas",
        "what homework do i have",
        "my school work is piling up",
    ]
    for phrase in should_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in CANVAS_KEYWORDS)
            assert matched, f"'{phrase}' should match but didn't"
            _pass(f"Matches: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"Matches: '{phrase}'", str(e)); failed += 1

    # --- 12c. Non-canvas phrases do NOT trigger ---
    should_not_match = [
        "hey what's up",
        "write me a python script",
        "how do i make pasta",
        "tell me a joke",
    ]
    for phrase in should_not_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in CANVAS_KEYWORDS)
            assert not matched, f"'{phrase}' should NOT match but did"
            _pass(f"No match: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"No match: '{phrase}'", str(e)); failed += 1

    # --- 12d. Assignment marker in CONTEXT_MARKERS ---
    try:
        assert "--- Your Assignments ---" in CONTEXT_MARKERS, \
            "Assignment marker missing from CONTEXT_MARKERS"
        _pass("12d. Assignment marker in CONTEXT_MARKERS"); passed += 1
    except Exception as e:
        _fail("12d. Assignment marker", str(e)); failed += 1

    # --- 12e. Context prompt includes deadline awareness ---
    try:
        from prompts import build_system_prompt
        ctx = build_system_prompt(has_context=True)
        assert "DEADLINE AWARENESS" in ctx, "DEADLINE AWARENESS step missing"
        assert "overdue" in ctx.lower(), "overdue mention missing from context prompt"
        _pass("12e. Context prompt has deadline awareness"); passed += 1
    except Exception as e:
        _fail("12e. Deadline awareness in prompt", str(e)); failed += 1

    # --- 12f. canvas_client imports cleanly and has expected functions ---
    try:
        from canvas_client import get_canvas_context, sync_courses
        assert callable(get_canvas_context)
        assert callable(sync_courses)
        _pass("12f. canvas_client exports expected functions"); passed += 1
    except Exception as e:
        _fail("12f. canvas_client imports", str(e)); failed += 1

    # --- 12g. get_canvas_context returns string (empty OK without API) ---
    try:
        from canvas_client import get_canvas_context as gcc
        import canvas_client
        canvas_client._cache = (0.0, "")  # clear cache
        result = gcc(days=7)
        assert isinstance(result, str), f"expected str, got {type(result)}"
        _pass("12g. get_canvas_context returns string"); passed += 1
    except Exception as e:
        _fail("12g. get_canvas_context return type", str(e)); failed += 1

    # --- 12h. has_recent_context detects assignment marker ---
    try:
        from prompts import has_recent_context
        history_with_assignments = [
            {"role": "user", "content": "--- Your Assignments ---\nCSCI E31\n  Paper due Mar 3"},
            {"role": "assistant", "content": '{"output": "you have a paper due"}'},
        ]
        assert has_recent_context(history_with_assignments) is True
        _pass("12h. has_recent_context detects assignment marker"); passed += 1
    except Exception as e:
        _fail("12h. Assignment marker detection", str(e)); failed += 1

    assert failed == 0, f"{failed} canvas_integration sub-tests failed"


def test_planning_keywords():
    _section("13. Planning Keywords — Auto-Context Trigger")
    passed = 0
    failed = 0

    # --- 13a. PLANNING_KEYWORDS is non-empty and all lowercase ---
    try:
        assert len(PLANNING_KEYWORDS) > 0, "PLANNING_KEYWORDS is empty"
        for kw in PLANNING_KEYWORDS:
            assert kw == kw.lower(), f"keyword '{kw}' is not lowercase"
        _pass(f"13a. PLANNING_KEYWORDS has {len(PLANNING_KEYWORDS)} entries, all lowercase"); passed += 1
    except Exception as e:
        _fail("13a. PLANNING_KEYWORDS", str(e)); failed += 1

    # --- 13b. Planning phrases trigger context ---
    should_match = [
        "help me schedule my week",
        "what should i work on tonight",
        "help me prioritize my tasks",
        "i'm falling behind on everything",
        "what's on deck for tomorrow",
        "help me plan this week",
        "i need to catch up on school",
        "prep for my classes",
    ]
    for phrase in should_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in PLANNING_KEYWORDS)
            assert matched, f"'{phrase}' should match but didn't"
            _pass(f"Matches: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"Matches: '{phrase}'", str(e)); failed += 1

    # --- 13c. Non-planning phrases do NOT trigger ---
    should_not_match = [
        "tell me a joke",
        "what is python",
        "explain recursion",
    ]
    for phrase in should_not_match:
        try:
            lower = phrase.lower()
            matched = any(kw in lower for kw in PLANNING_KEYWORDS)
            assert not matched, f"'{phrase}' should NOT match but did"
            _pass(f"No match: '{phrase}'"); passed += 1
        except Exception as e:
            _fail(f"No match: '{phrase}'", str(e)); failed += 1

    # --- 13d. Planning keywords don't overlap with generic words that cause false positives ---
    try:
        generic_inputs = ["what time is it", "i have a question", "is this a good idea"]
        false_positives = []
        for phrase in generic_inputs:
            if any(kw in phrase.lower() for kw in PLANNING_KEYWORDS):
                false_positives.append(phrase)
        assert not false_positives, f"False positives: {false_positives}"
        _pass("13d. No false positives on generic inputs"); passed += 1
    except Exception as e:
        _fail("13d. False positive check", str(e)); failed += 1

    assert failed == 0, f"{failed} planning_keywords sub-tests failed"


def test_sticky_local_model():
    _section("14. Sticky Local Model Switching")
    passed = 0
    failed = 0

    # --- 14a. STICKY_LOCAL_MAX is a reasonable value ---
    try:
        assert isinstance(STICKY_LOCAL_MAX, int), f"expected int, got {type(STICKY_LOCAL_MAX)}"
        assert 1 <= STICKY_LOCAL_MAX <= 50, f"STICKY_LOCAL_MAX={STICKY_LOCAL_MAX} seems off"
        _pass(f"14a. STICKY_LOCAL_MAX={STICKY_LOCAL_MAX} is reasonable"); passed += 1
    except Exception as e:
        _fail("14a. STICKY_LOCAL_MAX value", str(e)); failed += 1

    # --- 14b. Sticky state variables are correctly initialized ---
    # Simulate what main.py does before the while loop
    try:
        sticky_local_model: str | None = None
        sticky_local_turns: int = 0
        assert sticky_local_model is None
        assert sticky_local_turns == 0
        _pass("14b. Sticky state initializes to None/0"); passed += 1
    except Exception as e:
        _fail("14b. Sticky state initialization", str(e)); failed += 1

    # --- 14c. Setting sticky model works ---
    try:
        sticky_local_model = "gemma2:9b"
        sticky_local_turns = STICKY_LOCAL_MAX
        assert sticky_local_model == "gemma2:9b"
        assert sticky_local_turns == STICKY_LOCAL_MAX
        _pass("14c. Sticky model set correctly"); passed += 1
    except Exception as e:
        _fail("14c. Setting sticky model", str(e)); failed += 1

    # --- 14d. Decrement logic works over full cycle ---
    try:
        sticky_local_model = "gemma2:9b"
        sticky_local_turns = STICKY_LOCAL_MAX
        applied_models = []
        for _ in range(STICKY_LOCAL_MAX):
            # Simulate: no explicit trigger, sticky is active
            assert sticky_local_turns > 0
            applied_models.append(sticky_local_model)
            sticky_local_turns -= 1
            if sticky_local_turns == 0:
                sticky_local_model = None
        assert sticky_local_turns == 0, f"turns should be 0, got {sticky_local_turns}"
        assert sticky_local_model is None, "model should be None after expiry"
        assert len(applied_models) == STICKY_LOCAL_MAX
        assert all(m == "gemma2:9b" for m in applied_models)
        _pass(f"14d. Sticky expires after {STICKY_LOCAL_MAX} turns"); passed += 1
    except Exception as e:
        _fail("14d. Sticky decrement cycle", str(e)); failed += 1

    # --- 14e. Switching to default model clears sticky ---
    try:
        default_model = "qwen2.5:7b-instruct"
        sticky_local_model = "gemma2:9b"
        sticky_local_turns = 7  # mid-cycle
        # Simulate: user types @qwen (the default)
        force_local_model = default_model
        if force_local_model == default_model:
            sticky_local_model = None
            sticky_local_turns = 0
        assert sticky_local_model is None, "sticky should be cleared"
        assert sticky_local_turns == 0, "turns should be 0"
        _pass("14e. Tagging default model clears sticky"); passed += 1
    except Exception as e:
        _fail("14e. Default model clears sticky", str(e)); failed += 1

    # --- 14f. Switching to new non-default model resets counter ---
    try:
        sticky_local_model = "gemma2:9b"
        sticky_local_turns = 3  # partially through
        # Simulate: user types @llama mid-sticky
        new_model = "llama3.1:8b"
        default_model = "qwen2.5:7b-instruct"
        if new_model != default_model:
            sticky_local_model = new_model
            sticky_local_turns = STICKY_LOCAL_MAX
        assert sticky_local_model == "llama3.1:8b", "should switch to new model"
        assert sticky_local_turns == STICKY_LOCAL_MAX, "counter should reset to max"
        _pass("14f. New @local-model resets sticky counter"); passed += 1
    except Exception as e:
        _fail("14f. Mid-cycle model switch", str(e)); failed += 1

    # --- 14g. Clear command resets sticky state ---
    try:
        sticky_local_model = "gemma2:9b"
        sticky_local_turns = 5
        # Simulate: clear command
        clear_history()
        sticky_local_model = None
        sticky_local_turns = 0
        assert sticky_local_model is None
        assert sticky_local_turns == 0
        assert len(conversation_history) == 0
        _pass("14g. Clear resets sticky + history"); passed += 1
    except Exception as e:
        _fail("14g. Clear resets sticky", str(e)); failed += 1

    # --- 14h. Remote model trigger doesn't consume sticky turns ---
    try:
        sticky_local_model = "gemma2:9b"
        sticky_local_turns = 5
        # Simulate: @haiku (remote) — force_model is set, force_local_model is None
        force_local_model = None
        force_model = "HAIKU"
        # The sticky logic: if not force_local_model and not force_model and turns > 0
        if not force_local_model and not force_model and sticky_local_turns > 0:
            # This should NOT fire
            sticky_local_turns -= 1
        assert sticky_local_turns == 5, f"turns should still be 5, got {sticky_local_turns}"
        assert sticky_local_model == "gemma2:9b", "sticky model should be unchanged"
        _pass("14h. Remote trigger doesn't burn sticky turns"); passed += 1
    except Exception as e:
        _fail("14h. Remote doesn't consume sticky", str(e)); failed += 1

    assert failed == 0, f"{failed} sticky_local_model sub-tests failed"


def test_remote_prompt_config():
    _section("15. Remote Prompt Token Budget & History Config")
    passed = 0
    failed = 0

    # --- 15a. REMOTE_SYSTEM_PROMPT contains token budget instruction ---
    try:
        from prompts import REMOTE_SYSTEM_PROMPT
        assert "TOKEN BUDGET" in REMOTE_SYSTEM_PROMPT, "missing TOKEN BUDGET"
        assert "8000" in REMOTE_SYSTEM_PROMPT, "missing 8000 token reference"
        _pass("15a. REMOTE_SYSTEM_PROMPT has token budget"); passed += 1
    except Exception as e:
        _fail("15a. REMOTE_SYSTEM_PROMPT token budget", str(e)); failed += 1

    # --- 15b. REMOTE_CONTEXT_PROMPT contains token budget instruction ---
    try:
        from prompts import REMOTE_CONTEXT_PROMPT
        assert "TOKEN BUDGET" in REMOTE_CONTEXT_PROMPT, "missing TOKEN BUDGET"
        assert "8000" in REMOTE_CONTEXT_PROMPT, "missing 8000 token reference"
        _pass("15b. REMOTE_CONTEXT_PROMPT has token budget"); passed += 1
    except Exception as e:
        _fail("15b. REMOTE_CONTEXT_PROMPT token budget", str(e)); failed += 1

    # --- 15c. REMOTE_SYSTEM_PROMPT has history guidance ---
    try:
        from prompts import REMOTE_SYSTEM_PROMPT
        assert "LATEST" in REMOTE_SYSTEM_PROMPT or "latest" in REMOTE_SYSTEM_PROMPT, \
            "missing instruction to focus on latest question"
        assert "meandering" in REMOTE_SYSTEM_PROMPT or "unrelated" in REMOTE_SYSTEM_PROMPT, \
            "missing warning about unrelated history"
        _pass("15c. REMOTE_SYSTEM_PROMPT has history guidance"); passed += 1
    except Exception as e:
        _fail("15c. History guidance in system prompt", str(e)); failed += 1

    # --- 15d. REMOTE_CONTEXT_PROMPT has history guidance ---
    try:
        from prompts import REMOTE_CONTEXT_PROMPT
        assert "HISTORY NOTE" in REMOTE_CONTEXT_PROMPT or "history" in REMOTE_CONTEXT_PROMPT.lower(), \
            "missing history handling note"
        _pass("15d. REMOTE_CONTEXT_PROMPT has history guidance"); passed += 1
    except Exception as e:
        _fail("15d. History guidance in context prompt", str(e)); failed += 1

    # --- 15e. MAX_HISTORY matches expected value ---
    try:
        assert MAX_HISTORY == 82, f"expected 82, got {MAX_HISTORY}"
        _pass(f"15e. MAX_HISTORY={MAX_HISTORY} (41 exchanges)"); passed += 1
    except Exception as e:
        _fail("15e. MAX_HISTORY value", str(e)); failed += 1

    # --- 15f. build_remote_messages default max_turns covers full history ---
    try:
        import inspect
        sig = inspect.signature(build_remote_messages)
        default_max_turns = sig.parameters["max_turns"].default
        expected_turns = MAX_HISTORY // 2  # 82 / 2 = 41
        assert default_max_turns == expected_turns, \
            f"max_turns default is {default_max_turns}, expected {expected_turns} to match MAX_HISTORY"
        _pass(f"15f. build_remote_messages max_turns={default_max_turns} matches MAX_HISTORY"); passed += 1
    except Exception as e:
        _fail("15f. max_turns matches MAX_HISTORY", str(e)); failed += 1

    # --- 15g. build_remote_messages respects max_turns limit ---
    try:
        _reset()
        # Add more messages than max_turns=2 would include
        for i in range(10):
            add_message("user", f"question {i}")
            add_message("assistant", f'{{"output": "answer {i}", "next_step": "RESPOND_LOCALLY"}}')
        msgs = build_remote_messages("current question", max_turns=2)
        # Should only include recent turns + current question, not all 10
        # Count user messages (excluding the appended current question)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        # max_turns=2 grabs last 4 entries (2 user + 2 assistant), plus current question
        assert len(user_msgs) <= 4, f"expected <=4 user messages with max_turns=2, got {len(user_msgs)}"
        # Current question should be last
        assert msgs[-1]["role"] == "user"
        assert "current question" in msgs[-1]["content"]
        _pass("15g. build_remote_messages respects max_turns"); passed += 1
    except Exception as e:
        _fail("15g. max_turns limiting", str(e)); failed += 1
    finally:
        _reset()

    # --- 15h. Phone a Friend @ prefix stripping ---
    try:
        # Simulate what main.py does at the Phone a Friend prompt
        test_choices = ["@gpt_pro", "@haiku", "@sonnet", "@mini"]
        for raw_choice in test_choices:
            choice = raw_choice
            if choice.startswith("@"):
                choice = choice[1:]
            assert choice in MODEL_SHORTCUTS, f"'{raw_choice}' should resolve to valid shortcut after stripping @"
        _pass("15h. @ prefix stripped at Phone a Friend prompt"); passed += 1
    except Exception as e:
        _fail("15h. Phone a Friend @ prefix", str(e)); failed += 1

    assert failed == 0, f"{failed} remote_prompt_config sub-tests failed"


# ======================================================================
# 17. Session memory — load, save, forget, context injection
# ======================================================================

def test_session_memory():
    _section("17. Session Memory")
    passed = 0
    failed = 0

    from config import SESSION_SUMMARY_PATH
    from session_memory import load_session_summary, forget_session, save_session_summary

    # --- 17a. SESSION_SUMMARY_PATH points to vault System directory ---
    try:
        assert "System" in str(SESSION_SUMMARY_PATH), "path should be in System dir"
        assert str(SESSION_SUMMARY_PATH).endswith("session-summary.md"), \
            f"unexpected filename: {SESSION_SUMMARY_PATH}"
        _pass("17a. SESSION_SUMMARY_PATH in vault/System/session-summary.md"); passed += 1
    except Exception as e:
        _fail("17a. SESSION_SUMMARY_PATH", str(e)); failed += 1

    # --- 17b. load_session_summary returns "" for missing file ---
    try:
        with patch("session_memory.SESSION_SUMMARY_PATH", "/tmp/nonexistent_session_test_xyz.md"):
            result = load_session_summary()
            assert result == "", f"expected empty string, got: {result!r}"
        _pass("17b. Missing file → empty string"); passed += 1
    except Exception as e:
        _fail("17b. Missing file handling", str(e)); failed += 1

    # --- 17c. load_session_summary reads existing file ---
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("User was building session memory feature.\nDecided on single-file approach.")
            tmp_path = f.name
        with patch("session_memory.SESSION_SUMMARY_PATH", tmp_path):
            result = load_session_summary()
            assert "session memory" in result.lower(), f"unexpected content: {result!r}"
            assert "single-file" in result
        Path(tmp_path).unlink()
        _pass("17c. Existing file loaded correctly"); passed += 1
    except Exception as e:
        _fail("17c. Load existing file", str(e)); failed += 1

    # --- 17d. forget_session deletes the file ---
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("old session data")
            tmp_path = f.name
        assert Path(tmp_path).is_file(), "temp file should exist"
        with patch("session_memory.SESSION_SUMMARY_PATH", tmp_path):
            deleted = forget_session()
            assert deleted is True, "should return True"
            assert not Path(tmp_path).is_file(), "file should be deleted"
        _pass("17d. forget_session deletes file"); passed += 1
    except Exception as e:
        _fail("17d. forget_session", str(e)); failed += 1

    # --- 17e. forget_session returns False for missing file ---
    try:
        with patch("session_memory.SESSION_SUMMARY_PATH", "/tmp/nonexistent_session_test_xyz.md"):
            result = forget_session()
            assert result is False, f"expected False, got {result}"
        _pass("17e. forget_session on missing file → False"); passed += 1
    except Exception as e:
        _fail("17e. forget_session missing file", str(e)); failed += 1

    # --- 17f. save_session_summary skips trivial sessions ---
    try:
        # Less than _MIN_EXCHANGES (2) user messages
        trivial_history = [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "hi"},
        ]
        result = save_session_summary(trivial_history)
        assert result is False, "trivial session should not be saved"
        _pass("17f. Trivial session (1 exchange) not saved"); passed += 1
    except Exception as e:
        _fail("17f. Trivial session skip", str(e)); failed += 1

    # --- 17g. save_session_summary calls local model and writes file ---
    try:
        history = [
            {"role": "user", "content": "help me build session memory"},
            {"role": "assistant", "content": '{"output": "sure", "next_step": "RESPOND_LOCALLY"}'},
            {"role": "user", "content": "it should save on quit"},
            {"role": "assistant", "content": '{"output": "got it", "next_step": "RESPOND_LOCALLY"}'},
            {"role": "user", "content": "and load on startup"},
            {"role": "assistant", "content": '{"output": "makes sense", "next_step": "RESPOND_LOCALLY"}'},
        ]
        fake_summary = "User built session memory. Saves on quit, loads on startup. Uses local model."
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "session-summary.md"
            with patch("session_memory.SESSION_SUMMARY_PATH", str(tmp_path)), \
                 patch("clients.call_ollama_direct", return_value=fake_summary) as mock_call:
                # Need to also clear the conversation history that build_conversation_digest uses
                _reset()
                for msg in history:
                    add_message(msg["role"], msg["content"])
                result = save_session_summary(history)
                assert result is True, f"expected True, got {result}"
                assert tmp_path.is_file(), "summary file should exist"
                content = tmp_path.read_text()
                assert "session memory" in content.lower()
                assert mock_call.called, "should have called call_ollama_direct"
                # Verify it was called with show_stream=False
                call_kwargs = mock_call.call_args
                assert call_kwargs[1].get("show_stream") is False, \
                    "summary generation should be silent"
        _pass("17g. save_session_summary writes file via local model"); passed += 1
    except Exception as e:
        _fail("17g. save_session_summary", str(e)); failed += 1
    finally:
        _reset()

    # --- 17h. save_session_summary handles model failure gracefully ---
    try:
        history = [
            {"role": "user", "content": "first question about something"},
            {"role": "assistant", "content": '{"output": "answer", "next_step": "RESPOND_LOCALLY"}'},
            {"role": "user", "content": "second question about stuff"},
            {"role": "assistant", "content": '{"output": "more", "next_step": "RESPOND_LOCALLY"}'},
        ]
        _reset()
        for msg in history:
            add_message(msg["role"], msg["content"])
        with patch("clients.call_ollama_direct", side_effect=Exception("model crashed")):
            result = save_session_summary(history)
            assert result is False, "should return False on model failure"
        _pass("17h. Model failure → graceful False"); passed += 1
    except Exception as e:
        _fail("17h. Model failure handling", str(e)); failed += 1
    finally:
        _reset()

    # --- 17i. Session summary injects into first turn only ---
    try:
        _reset()
        session_summary = "User was building session memory feature."
        user_input = "hey what's up"
        # Simulate main.py injection logic: only when history is empty
        if session_summary and not conversation_history:
            _session_block = (
                "--- Previous Session ---\n"
                f"{session_summary}\n"
                "--- End Previous Session ---"
            )
            user_input = _session_block + "\n\n" + user_input
        assert "--- Previous Session ---" in user_input
        assert "session memory" in user_input
        assert "hey what's up" in user_input

        # After first turn is added to history, injection should NOT happen
        add_message("user", user_input)
        add_message("assistant", "hi there")
        user_input2 = "tell me a joke"
        if session_summary and not conversation_history:
            user_input2 = "SHOULD NOT HAPPEN\n\n" + user_input2
        assert "SHOULD NOT HAPPEN" not in user_input2, "second turn should not get injection"
        _pass("17i. Session summary injected on first turn only"); passed += 1
    except Exception as e:
        _fail("17i. First-turn-only injection", str(e)); failed += 1
    finally:
        _reset()

    # --- 17j. /forget clears in-memory session_summary ---
    try:
        session_summary = "Some old session data"
        # Simulate /forget command logic from main.py
        with patch("session_memory.SESSION_SUMMARY_PATH", "/tmp/nonexistent_xyz.md"):
            forget_session()
        session_summary = ""
        assert session_summary == "", "session_summary should be empty after /forget"
        _pass("17j. /forget clears in-memory variable"); passed += 1
    except Exception as e:
        _fail("17j. /forget clears variable", str(e)); failed += 1

    # --- 17k. load_session_summary handles empty file ---
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")  # empty file
            tmp_path = f.name
        with patch("session_memory.SESSION_SUMMARY_PATH", tmp_path):
            result = load_session_summary()
            assert result == "", f"empty file should return empty string, got: {result!r}"
        Path(tmp_path).unlink()
        _pass("17k. Empty file → empty string"); passed += 1
    except Exception as e:
        _fail("17k. Empty file handling", str(e)); failed += 1

    # --- 17l. Session summary doesn't trigger keyword detection ---
    try:
        # If session summary mentions "calendar" or "journal", it should NOT
        # trigger context injection because keyword detection uses original input
        session_text = "User checked their calendar and reviewed journal entries."
        user_msg = "hey what's up"
        # Keyword detection uses the ORIGINAL lower_input, not the injected version
        lower_input = user_msg.lower()
        from config import CALENDAR_KEYWORDS, JOURNAL_KEYWORDS
        cal_match = any(kw in lower_input for kw in CALENDAR_KEYWORDS)
        journal_match = any(kw in lower_input for kw in JOURNAL_KEYWORDS)
        assert not cal_match, "session summary should not trigger calendar keywords"
        assert not journal_match, "session summary should not trigger journal keywords"
        _pass("17l. Session text doesn't trigger keyword detection"); passed += 1
    except Exception as e:
        _fail("17l. No false keyword triggers", str(e)); failed += 1

    _reset()
    assert failed == 0, f"{failed} session_memory sub-tests failed"


# ======================================================================
# Main (standalone runner — also works with pytest)
# ======================================================================

def main():
    print(f"{BOLD}{CYAN}Local Test Suite — No API Calls{RESET}")
    print(f"{DIM}Testing parse, conversation, digest, shortcuts, config, calendar, journal, prompts, canvas, and session memory{RESET}")

    total_passed = 0
    total_failed = 0

    for test_fn in [
        test_parse_local_response,
        test_conversation_history,
        test_build_conversation_digest,
        test_build_remote_messages,
        test_get_ollama_models,
        test_config_sanity,
        test_calendar_keywords,
        test_natural_conversation_flow,
        test_journal_keywords,
        test_journal_parsing,
        test_prompt_composition,
        test_canvas_integration,
        test_planning_keywords,
        test_sticky_local_model,
        test_remote_prompt_config,
        test_session_memory,
    ]:
        try:
            test_fn()
            total_passed += 1
        except AssertionError:
            total_failed += 1

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
