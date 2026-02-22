import json
import os
import re
import sys
import termios
from dotenv import load_dotenv
import requests
from anthropic import Anthropic

load_dotenv()

# --- ANSI color codes (UI layer — swap these out when moving to a GUI) ---
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"


LOCAL_SYSTEM_PROMPT = """You are the local AI in a relay system. You talk to a real human. Be natural, friendly, and conversational.

IMPORTANT — YOUR KNOWLEDGE HAS LIMITS:
- You are llama3.1:8b. Your training data has a cutoff around mid-2024.
- The user is living in the present (2026). They know things you don't.
- If the user tells you something is true (a movie exists, a product launched, a person is in office), TRUST THEM — don't argue or say "I'm not aware of that."
- If you don't know something recent, say so honestly and offer to send it to the remote model.
- You are great at timeless knowledge. You are bad at anything after mid-2024. Know the difference.
- CRITICAL: If the user asks about a movie, product, person, event, or anything you DON'T RECOGNIZE — assume it's something new that came out after your training data. Do NOT say "that doesn't exist" or "I'm not aware of that title." Instead, use SEND_TO_REMOTE because the user probably knows it exists and you just haven't heard of it yet.

RESPONSE QUALITY:
- Give SUBSTANTIVE answers, not one-liners. If someone asks about a topic, give a real explanation with useful detail.
- For factual questions: include key facts, context, and nuance — not just the bare minimum.
- For code: include the code AND a brief explanation of how it works.
- For health/medical: cover the main points someone would actually want to know.
- For comparisons: cover the key differences with enough detail to be useful.
- Think "helpful coworker explaining something at a whiteboard" — not "dictionary definition."
- Aim for 3-8 sentences for most answers. One-sentence answers are almost never enough unless the question is truly trivial (like "what's 2+2").

This is a 3-WAY CONVERSATION: the user, you (local model), and a remote Claude model. When you send a request to the remote model, its response will appear in the conversation history as "[Remote model (MODEL) responded]: ...". You can see and reference what Claude said. Use this context:
- If the user asks a follow-up about something Claude answered, you have that answer in your history — use it
- Don't re-send questions to the remote model if the answer is already in the conversation
- If Claude already answered well, just summarize or reference it locally instead of making another API call
- You are the user's primary interface — Claude is a resource you call on when needed, not the default

You MUST always respond with a valid JSON object — no other text, no markdown, no explanation outside the JSON. The JSON must have exactly these keys:

{
  "analysis": "one sentence — what is the user asking or doing?",
  "sensitive_data": "YES or NO",
  "next_step": "RESPOND_LOCALLY or SEND_TO_REMOTE or ASK_USER",
  "model": "HAIKU or SONNET or OPUS or NONE",
  "output": "your response, a detailed prompt for the remote model, or a clarifying question"
}

--- ROUTING RULES ---

YOUR DEFAULT IS RESPOND_LOCALLY. The user wants to work offline as much as possible. Remote calls cost money and require internet. Only use SEND_TO_REMOTE when you are CERTAIN you cannot handle it yourself. If you're unsure, TRY LOCALLY FIRST.

RESPOND_LOCALLY — THIS IS YOUR DEFAULT. Use it for everything you can:
- Greetings, casual chat, humor, thanks, goodbyes
- ANY factual question where the answer hasn't changed in years (science, math, geography, history, definitions, general knowledge)
- Health, medical, cooking, fitness, nutrition — established knowledge that doesn't change
- Conversational questions, opinions, preferences
- Questions about this system or how the relay works
- Tutorials, how-tos, step-by-step instructions for anything you know well
- Code snippets, debugging, programming concepts, algorithms, data structures
- Explanations of technologies, frameworks, languages (what they ARE, how they work)
- Recommendations, comparisons, pros/cons of tools or approaches
- Summarizing or referencing something Claude already said in the conversation
- ANYTHING you can give a complete, accurate, helpful answer to — JUST DO IT

THE KEY TEST: Can you answer this well enough? Then do it. Don't send it out just because the remote model might answer it "better." A good local answer beats a perfect remote answer that costs money.

SEND_TO_REMOTE — ONLY when you genuinely cannot do the job:
- Current events, recent news, anything after your training data cutoff — you WILL get these wrong, so don't try
- Large/complex code projects (full applications, not snippets)
- Tasks where you've tried locally and your answer is clearly inadequate
- You do NOT have internet access. If the answer depends on what's happening NOW, send it out.

CRITICAL — WHEN THE USER COMPLAINS ABOUT ROUTING:
If the user expresses frustration about unnecessary API calls — this is feedback. Do NOT send another remote call. Acknowledge it, answer locally, and adjust.

ASK_USER — you need more info before you can do anything:
- The request is too vague to act on
- Key details are missing that you need before you can help (locally OR remotely)
- PREFER this over sending a weak prompt — gathering info is free, API calls cost money

--- WHEN SENDING TO REMOTE ---

THE OUTPUT IS NOT YOUR ANSWER — it's an INSTRUCTION to the remote model.
- Write a clear instruction telling the remote model what to produce
- Include the user's constraints from THIS message only
- WRONG: "LangChain integrates AI with frameworks." (your own answer)
- RIGHT: "Explain what LangChain is in 5 sentences." (instruction for remote)

--- MODEL SELECTION (only when SEND_TO_REMOTE) ---
HAIKU ($1/$5 MTok, data: April 2024) — default for pre-2024 lookups, simple Q&A
SONNET ($3/$15 MTok, data: Jan 2026) — code, analysis, anything recent (2024-2026)
OPUS ($5/$25 MTok) — complex reasoning only, use sparingly
Set model to "NONE" when not sending to remote.

--- OTHER RULES ---
sensitive_data: "YES" if message has passwords, API keys, SSNs, personal identifiers. Strip them with [REDACTED].
Tone: warm, helpful coworker. Substantive but not robotic.

--- EXAMPLES (LOCAL vs REMOTE vs ASK) ---
"hey what's up" → RESPOND_LOCALLY
"what's the capital of Japan" → RESPOND_LOCALLY
"how do gallstones form" → RESPOND_LOCALLY (established medical knowledge)
"write a palindrome checker in python" → RESPOND_LOCALLY (you know this)
"who is the president right now" → SEND_TO_REMOTE/HAIKU (current events)
"tell me about 28 years later movie" → SEND_TO_REMOTE/SONNET (you don't recognize it = post-cutoff)
"what are the latest langchain updates" → SEND_TO_REMOTE/SONNET (recent changes)
"build me an app" → ASK_USER (too vague)
"my mother's maiden name is poop" → RESPOND_LOCALLY, sensitive_data: YES, [REDACTED]
"""

MIN_REMOTE_OUTPUT_LENGTH = 80  # minimum chars for a prompt sent to remote model

# Conversation history for the local model (gives it memory across turns)
conversation_history: list[dict[str, str]] = []


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value or ""


def call_ollama(prompt: str, show_stream: bool = False) -> str:
    """Send a message to Ollama using the chat API (supports conversation history).

    If show_stream is True, tokens are printed to stdout as they arrive so the
    user can see the model "thinking".
    """
    base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
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
                "num_predict": 2048,
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

    # Keep history manageable (last 20 turns = 10 exchanges)
    if len(conversation_history) > 20:
        conversation_history[:] = conversation_history[-20:]

    return assistant_content


_anthropic_client: Anthropic | None = None


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = get_env("ANTHROPIC_API_KEY", required=True)
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def call_anthropic(prompt: str, model: str | None = None) -> str:
    if not model:
        model = get_env("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = _get_anthropic_client()

    result = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts: list[str] = []
    for block in result.content:
        if getattr(block, "type", "") == "text":
            text_parts.append(getattr(block, "text", ""))
    return "\n".join(part for part in text_parts if part).strip()


def call_anthropic_stream(prompt: str, model: str | None = None) -> str:
    """Stream an Anthropic response, printing tokens as they arrive in green."""
    if not model:
        model = get_env("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = _get_anthropic_client()

    collected: list[str] = []
    print(GREEN, end="", flush=True)
    with client.messages.stream(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            collected.append(text)
    print(RESET)
    return "".join(collected).strip()


MODEL_MAP = {
    "HAIKU": "claude-haiku-4-5",
    "SONNET": "claude-sonnet-4-6",
    "OPUS": "claude-opus-4-6",
}

VALID_NEXT_STEPS = {"RESPOND_LOCALLY", "SEND_TO_REMOTE", "ASK_USER"}


def parse_local_response(raw_text: str) -> dict[str, str]:
    """Parse JSON response from local model."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from the response
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Local model did not return valid JSON: {raw_text[:200]}")

    # Normalize keys to uppercase
    parsed: dict[str, str] = {}
    key_map = {
        "analysis": "ANALYSIS",
        "sensitive_data": "SENSITIVE_DATA",
        "next_step": "NEXT_STEP",
        "model": "MODEL",
        "output": "OUTPUT",
    }
    for json_key, field_key in key_map.items():
        value = data.get(json_key, data.get(field_key, ""))
        parsed[field_key] = str(value).strip()

    required = {"ANALYSIS", "SENSITIVE_DATA", "NEXT_STEP", "MODEL", "OUTPUT"}
    missing = required - {k for k, v in parsed.items() if v}
    if missing:
        raise ValueError(f"Local model response missing required fields: {', '.join(sorted(missing))}")

    # Fix invalid NEXT_STEP values
    next_step = parsed["NEXT_STEP"].upper().strip()
    if next_step not in VALID_NEXT_STEPS:
        # Default to RESPOND_LOCALLY if the model invented a value
        parsed["NEXT_STEP"] = "RESPOND_LOCALLY"

    return parsed


def print_local_summary(parsed: dict[str, str]) -> None:
    print(f"  {CYAN}Analysis:{RESET} {parsed['ANALYSIS']}")
    sensitive = parsed['SENSITIVE_DATA']
    sens_color = RED if sensitive.upper() == "YES" else CYAN
    print(f"  {CYAN}Sensitive:{RESET} {sens_color}{sensitive}{RESET}")
    route = parsed['NEXT_STEP']
    route_color = GREEN if route == "RESPOND_LOCALLY" else MAGENTA if route == "SEND_TO_REMOTE" else YELLOW
    print(f"  {CYAN}Route:{RESET} {route_color}{route}{RESET}", end="")
    if parsed.get("MODEL", "NONE").upper() != "NONE":
        print(f" {MAGENTA}→ {parsed['MODEL']}{RESET}")
    else:
        print()


def relay_once(user_input: str) -> None:
    print(f"\n{CYAN}--- Local model ---{RESET}")
    local_raw = call_ollama(user_input, show_stream=True)

    try:
        parsed = parse_local_response(local_raw)
    except ValueError:
        # JSON was likely truncated — retry with a shorter-answer instruction
        print(f"  {YELLOW}[TRUNCATED] Response too long for JSON — retrying with shorter format...{RESET}")
        retry_prompt = (f"The user asked: {user_input}\n\n"
                        f"Your previous answer was too long and got cut off. "
                        f"Give a SHORTER but still complete answer. "
                        f"Hit the key points without being exhaustive. "
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
    print_local_summary(parsed)

    next_step = parsed["NEXT_STEP"].upper().strip()

    # --- Python-side guardrail: catch thin or bad prompts before they cost money ---
    if next_step == "SEND_TO_REMOTE":
        output = parsed["OUTPUT"]

        # Check if output looks like the model's own answer instead of instructions
        # (poems, declarative statements about the topic, not telling the remote what to do)
        bad_prompt_hints = False
        instruction_verbs = ["explain", "describe", "provide", "give", "list", "write",
                             "summarize", "tell", "what is", "what are", "how does",
                             "compare", "analyze", "create", "generate", "the user"]
        if (len(output) > 0
                and not any(verb in output.lower() for verb in instruction_verbs)
                and not output.rstrip().endswith("?")):
            bad_prompt_hints = True

        if len(output) < MIN_REMOTE_OUTPUT_LENGTH or bad_prompt_hints:
            # Try to auto-fix: ask the local model to rewrite as a proper instruction
            print(f"\n{YELLOW}[GUARDRAIL] Prompt looks weak ({len(output)} chars, instruction={not bad_prompt_hints}). Auto-fixing...{RESET}")
            fix_prompt = (f"The user's original request was: \"{user_input}\"\n"
                          f"You need to write a CLEAR INSTRUCTION for the remote model. "
                          f"Do NOT answer the question yourself. "
                          f"Write a prompt that tells the remote model exactly what to produce. "
                          f"Include any constraints the user specified (length, format, etc). "
                          f"Set next_step to SEND_TO_REMOTE.")
            fix_raw = call_ollama(fix_prompt)
            try:
                fix_parsed = parse_local_response(fix_raw)
                fixed_output = fix_parsed["OUTPUT"]
                if len(fixed_output) >= MIN_REMOTE_OUTPUT_LENGTH:
                    parsed["OUTPUT"] = fixed_output
                    parsed["MODEL"] = fix_parsed.get("MODEL", parsed.get("MODEL", "HAIKU"))
                    print(f"{YELLOW}[GUARDRAIL] Rewrote prompt: {fixed_output[:100]}...{RESET}")
                else:
                    print(f"\n{YELLOW}[GUARDRAIL] Still too short after rewrite. Can you be more specific about what you need?{RESET}")
                    return
            except (ValueError, json.JSONDecodeError):
                print(f"\n{YELLOW}[GUARDRAIL] Couldn't auto-fix. Can you rephrase with more detail?{RESET}")
                return

    if next_step == "RESPOND_LOCALLY":
        output = parsed["OUTPUT"]
        # Detect truncated responses
        truncation_hints = ["here's how", "let's break", "step by step", "here are the steps",
                           "follow these", "here's a"]
        looks_truncated = False

        # Check 1: instructional opener that never delivered (original check)
        if (any(hint in output.lower() for hint in truncation_hints)
                and len(output) < 200
                and output.rstrip().endswith((".", ":", "!"))):
            looks_truncated = True

        # Check 2: response ends mid-sentence (no terminal punctuation)
        # A strong signal the model hit its token limit mid-thought
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
        print(f"\n{BOLD}{output}{RESET}")
        return

    if next_step == "ASK_USER":
        print(f"\n{YELLOW}{parsed['OUTPUT']}{RESET}")
        return

    if next_step == "SEND_TO_REMOTE":
        if parsed.get("SENSITIVE_DATA", "").upper() == "YES":
            print(f"\n{RED}[SAFETY] Sensitive data detected — sanitized prompt being sent.{RESET}")

        model_choice = parsed.get("MODEL", "HAIKU").upper().strip()
        model_id = MODEL_MAP.get(model_choice, MODEL_MAP["HAIKU"])
        remote_prompt = parsed["OUTPUT"]

        print(f"\n{MAGENTA}--- Remote model ({model_choice}) ---{RESET}")
        print(f"{DIM}[Sending {len(remote_prompt)} chars to {model_id}]{RESET}")
        print(f"{DIM}[PROMPT]: {remote_prompt}{RESET}")
        print()
        remote_response = call_anthropic_stream(remote_prompt, model=model_id)

        # Auto-escalate: if Haiku says it doesn't know, retry with Sonnet
        dont_know_phrases = [
            "don't have information",
            "don't have reliable information",
            "my training data",
            "my knowledge cutoff",
            "not aware of",
            "knowledge was last updated",
            "i'm not sure about",
            "i cannot confirm",
            "as of my",
            "after my knowledge",
            "cannot provide",
            "i don't have",
        ]
        # Normalize smart quotes to ASCII before matching
        normalized_response = remote_response.lower().replace("\u2019", "'").replace("\u2018", "'")
        if model_choice == "HAIKU" and any(phrase in normalized_response for phrase in dont_know_phrases):
            print(f"\n{YELLOW}[AUTO-ESCALATE] Haiku doesn't know this — retrying with Sonnet...{RESET}")
            sonnet_id = MODEL_MAP["SONNET"]
            remote_response = call_anthropic_stream(remote_prompt, model=sonnet_id)
            print(f"\n{MAGENTA}--- Escalated to SONNET ({sonnet_id}) ---{RESET}")

        # Inject Claude's response into Ollama's conversation history
        # so the local model knows what was said and can reference it
        conversation_history.append({
            "role": "assistant",
            "content": f"[Remote model ({model_choice}) responded]: {remote_response}"
        })
        # Keep history manageable
        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]

        return

    print(f"\n{RED}[ERROR] Unexpected routing: {next_step}{RESET}")


def main() -> None:
    print(f"{BOLD}{CYAN}Relay v2: Local (Ollama) → Remote (Claude){RESET}")
    print(f"{DIM}Commands: 'exit' to quit, 'clear' to reset conversation{RESET}\n")

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
            conversation_history.clear()
            print(f"{CYAN}[Conversation cleared]{RESET}")
            continue

        try:
            relay_once(user_input)
        except (requests.RequestException, ValueError) as err:
            print(f"{RED}[ERROR] {err}{RESET}")
        except Exception as err:  # noqa: BLE001
            print(f"{RED}[UNEXPECTED ERROR] {err}{RESET}")


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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Unloading model...")
        unload_model()
