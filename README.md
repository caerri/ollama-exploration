# Phase 1: Local-First Relay (Baby Steps)

This project is now a minimal relay orchestrator:

1. Send user input to a local Ollama model for sanitized analysis.
2. Parse structured local output.
3. If local says `SEND_TO_REMOTE`, forward sanitized text to Anthropic.
4. Print remote response in terminal.

No UI automation is executed in this phase.

## Files

- `main.py` — interactive relay loop.
- `system_check.py` — validates Python, Ollama availability, and Anthropic API key presence.
- `.env.example` — environment variable template.

## Setup

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (or export manually):

```bash
export ANTHROPIC_API_KEY="your_key_here"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3"
export ANTHROPIC_MODEL="claude-3-5-sonnet-latest"
```

## Verify environment

```bash
python system_check.py
```

## Run relay

```bash
python main.py
```

## Example session

```text
You > summarize this local note and suggest next actions

--- Step 1: Local model (Ollama) ---
[LOCAL] Raw response:
ANALYSIS: User asks for a concise summary and recommended actions.
SENSITIVE_DATA: NO
NEXT_STEP: SEND_TO_REMOTE
OUTPUT: Summarize the following note and provide 3 prioritized next actions: ...

--- Step 2: Remote model (Anthropic) ---
[RELAY] Sending sanitized prompt to remote model...

[REMOTE] Response:
1) ...
2) ...
3) ...
```
