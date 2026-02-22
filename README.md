# Local-First Multi-Model Relay

A terminal-based AI relay that routes conversations through a chain of models — local first, remote only when needed.

## How it works

```
You type a question
        |
  llama3.1:8b (local, fast, free)
  Tries to answer. If it can't:
        |
  deepseek-r1:14b (local, bigger brain, free)
  Second opinion. If it still can't:
        |
  Phone a Friend? [y/n]
  You choose whether to spend money.
        |
  Claude (Anthropic) or GPT (OpenAI)
  Remote API call with full conversation context.
```

- Local models handle greetings, factual Q&A, code snippets, and anything they know well.
- Remote calls only happen with your explicit approval ("Phone a Friend" confirmation gate).
- Say "phone a friend" or "!remote" in your message to skip straight to remote.
- Responses stream in real-time with color-coded output.

## Files

- `main.py` — the relay app (routing, streaming, escalation, all of it).
- `system_check.py` — validates Python, Ollama, and API key availability.
- `.env.example` — environment variable template.
- `run.sh` — launches the app with the venv.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy and fill in your env file:

```bash
cp .env.example .env
```

Then edit `.env` with your keys:

```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_ESCALATION_MODEL=deepseek-r1:14b
```

4. Pull the Ollama models:

```bash
ollama pull llama3.1:8b
ollama pull deepseek-r1:14b
```

## Run

```bash
./run.sh
```

Or directly:

```bash
python main.py
```

## Commands

| Command | What it does |
|---|---|
| `exit` / `quit` | Stop the relay and unload the model |
| `clear` | Reset conversation history |
| `phone a friend <question>` | Force a remote API call (skips confirmation) |
| `!remote <question>` | Same as above |

## Model routing

The local model decides where to send each request:

| Provider | Model | Cost | Best for |
|---|---|---|---|
| **Anthropic** | HAIKU | cheap | Factual Q&A, summaries, light code (default) |
| **Anthropic** | SONNET | moderate | Complex code, multi-step reasoning |
| **Anthropic** | OPUS | expensive | Research-grade analysis (rarely needed) |
| **OpenAI** | GPT_MINI | cheap | Brainstorming, creative writing (default for GPT) |
| **OpenAI** | GPT | moderate | Polished writing, stronger coding |
| **OpenAI** | GPT_PRO | very expensive | Most precise (rarely needed) |

## Color scheme

- **Dim** — local model's JSON thinking stream
- **Bold** — local model's final answer
- **Sage green** — Anthropic (Claude) responses
- **Blue** — OpenAI (GPT) responses
- **Magenta** — Phone a Friend prompts and remote model headers
- **Yellow** — warnings, clarifying questions
- **Red** — errors, sensitive data alerts

## Escalation chain

1. **llama3.1:8b** — fast local router, handles most things
2. **deepseek-r1:14b** — local escalation, bigger reasoning model (set via `OLLAMA_ESCALATION_MODEL`, leave empty to skip)
3. **Remote API** — Claude or GPT, with user confirmation required

Empty Enter at the Phone a Friend prompt cancels (doesn't send). Only explicit `y` sends.

If a cheap remote model (Haiku/GPT Mini) can't answer, you'll be asked before escalating to a more expensive one.
