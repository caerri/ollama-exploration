import json
import os
import re
import sys
import termios
from dotenv import load_dotenv
import requests
from anthropic import Anthropic
from openai import OpenAI

load_dotenv()

# --- ANSI color codes (UI layer — swap these out when moving to a GUI) ---
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[38;5;115m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[38;5;111m"


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

This is a MULTI-MODEL CONVERSATION: the user, you (local model), and remote models (Claude from Anthropic, GPT from OpenAI). When you send a request to a remote model, its response will appear in the conversation history as "[Remote model (MODEL) responded]: ...". You can see and reference what any remote model said. Use this context:
- If the user asks a follow-up about something a remote model answered, you have that answer in your history — use it
- Don't re-send questions to a remote model if the answer is already in the conversation
- If a remote model already answered well, just summarize or reference it locally instead of making another API call
- You are the user's primary interface — remote models are resources you call on when needed, not the default

You MUST always respond with a valid JSON object — no other text, no markdown, no explanation outside the JSON. The JSON must have exactly these keys:

{
  "analysis": "one sentence — what is the user asking or doing?",
  "sensitive_data": "YES or NO",
  "next_step": "RESPOND_LOCALLY or SEND_TO_REMOTE or ASK_USER",
  "model": "HAIKU or SONNET or OPUS or GPT_MINI or GPT or GPT_PRO or NONE",
  "output": "your response, a detailed prompt for the remote model, or a clarifying question",
  "context_summary": "brief summary of relevant conversation context for the remote model, or empty string if not needed"
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
- NOTE: SEND_TO_REMOTE is a SUGGESTION. The user will be asked to confirm before the call is made ("Phone a Friend"). They can say no. So make sure your output field contains a good prompt, but also be prepared to answer locally if the user declines.
- When suggesting SEND_TO_REMOTE with conversation history that matters, fill in context_summary with a brief recap so the remote model has enough context. Keep it under 200 words. Leave it as "" for simple standalone questions.

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
You have TWO providers: Anthropic (Claude) and OpenAI (GPT). Pick the right model for the task.

ANTHROPIC (Claude) — best for: technical tasks, code, analysis, structured reasoning
  HAIKU (cheap, fast) — YOUR DEFAULT for most remote calls. Factual Q&A, summaries, explanations, lookups, light code.
  SONNET (moderate) — Complex code generation, multi-step reasoning, detailed technical analysis. Don't pick just because it feels "important."
  OPUS (expensive) — NEVER select this unless the user explicitly says "use opus". No exceptions. If you think a task needs Opus, use SONNET instead.

OPENAI (GPT) — best for: creative writing, brainstorming, ideation, open-ended exploration, storytelling
  GPT_MINI (cheap, fast) — DEFAULT when using GPT. Good at: brainstorming, creative writing, generating ideas, casual writing tasks, summarizing with flair.
  GPT (moderate) — Stronger coding and agentic tasks, longer and more polished writing, when GPT_MINI isn't cutting it.
  GPT_PRO (very expensive) — NEVER select this unless the user explicitly says "use gpt pro". No exceptions. If you think a task needs GPT_PRO, use GPT instead.

WHEN TO PICK CLAUDE vs GPT:
- Code, debugging, technical docs → Claude (HAIKU or SONNET)
- Creative writing, brainstorming, ideation, storytelling → GPT (GPT_MINI or GPT)
- Factual Q&A, lookups → either, prefer HAIKU (cheapest)
- If the user asks for a specific provider, use it

Set model to "NONE" when not sending to remote.
RULE: When in doubt, ALWAYS pick HAIKU or GPT_MINI (whichever fits the task type).

--- OTHER RULES ---
sensitive_data: "YES" if message has passwords, API keys, SSNs, personal identifiers. Strip them with [REDACTED].
Tone: warm, helpful coworker. Substantive but not robotic.

--- EXAMPLES (LOCAL vs REMOTE vs ASK) ---
"hey what's up" → RESPOND_LOCALLY
"what's the capital of Japan" → RESPOND_LOCALLY
"how do gallstones form" → RESPOND_LOCALLY (established medical knowledge)
"write a palindrome checker in python" → RESPOND_LOCALLY (you know this)
"who is the president right now" → SEND_TO_REMOTE/HAIKU (current events)
"tell me about 28 years later movie" → SEND_TO_REMOTE/HAIKU (you don't recognize it = post-cutoff, just a lookup)
"what are the latest langchain updates" → SEND_TO_REMOTE/SONNET (recent + technical)
"brainstorm 10 names for my startup" → SEND_TO_REMOTE/GPT_MINI (creative brainstorming)
"write me a short story about a robot" → SEND_TO_REMOTE/GPT_MINI (creative writing)
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
                    "num_predict": 2048,
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
        conversation_history.append({
            "role": "assistant",
            "content": f"[Local escalation ({model}) responded]: {result}"
        })
        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]

    return result


_anthropic_client: Anthropic | None = None


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = get_env("ANTHROPIC_API_KEY", required=True)
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def call_anthropic(prompt: str, model: str | None = None) -> str:
    if not model:
        model = "claude-haiku-4-5"
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
        model = "claude-haiku-4-5"
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


_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = get_env("OPENAI_API_KEY", required=True)
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_openai_stream(prompt: str, model: str | None = None) -> str:
    """Stream an OpenAI response, printing tokens as they arrive in blue."""
    if not model:
        model = "gpt-5-mini"
    client = _get_openai_client()

    collected: list[str] = []
    print(BLUE, end="", flush=True)
    stream = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            collected.append(delta.content)
    print(RESET)
    return "".join(collected).strip()


MODEL_MAP = {
    "HAIKU": "claude-haiku-4-5",
    "SONNET": "claude-sonnet-4-6",
    "OPUS": "claude-opus-4-6",
    "GPT_MINI": "gpt-5-mini",
    "GPT": "gpt-5.2",
    "GPT_PRO": "gpt-5.2-pro",
}

# Which provider handles each model
MODEL_PROVIDER = {
    "HAIKU": "anthropic",
    "SONNET": "anthropic",
    "OPUS": "anthropic",
    "GPT_MINI": "openai",
    "GPT": "openai",
    "GPT_PRO": "openai",
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
        "context_summary": "CONTEXT_SUMMARY",
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


COST_INFO = {
    "HAIKU": "cheap",
    "SONNET": "moderate",
    "OPUS": "expensive",
    "GPT_MINI": "cheap",
    "GPT": "moderate",
    "GPT_PRO": "very expensive",
}


def build_conversation_digest(max_turns: int = 10) -> str:
    """Build a plain-text digest of the recent conversation for the remote model.

    The local model's context_summary field is often too vague ("User wants
    advice...") so Python constructs a real digest from conversation_history.
    This way the remote model sees what the user actually said.
    """
    if not conversation_history:
        return ""

    # Grab the last N entries (user + assistant pairs)
    recent = conversation_history[-max_turns * 2:]
    lines: list[str] = []
    for msg in recent:
        role = msg["role"]
        content = msg["content"]
        # Skip the local model's raw JSON — it's not useful context for the remote model
        # Instead, look for remote responses injected into history
        if role == "assistant":
            if content.startswith("[Remote model"):
                lines.append(f"Remote model: {content}")
            else:
                # Try to extract just the output field from local JSON
                try:
                    parsed = json.loads(content)
                    local_answer = parsed.get("output", "")
                    if local_answer and parsed.get("next_step", "").upper() == "RESPOND_LOCALLY":
                        lines.append(f"Local model: {local_answer}")
                except (json.JSONDecodeError, AttributeError):
                    pass  # skip unparseable assistant turns
        elif role == "user":
            lines.append(f"User: {content}")

    if not lines:
        return ""
    return "--- Conversation so far ---\n" + "\n".join(lines) + "\n--- End of conversation ---"


def relay_once(user_input: str, force_remote: bool = False) -> None:
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

    # If user explicitly requested remote, override local routing
    if force_remote and parsed["NEXT_STEP"].upper() != "SEND_TO_REMOTE":
        parsed["NEXT_STEP"] = "SEND_TO_REMOTE"
        if not parsed.get("MODEL") or parsed["MODEL"].upper() == "NONE":
            parsed["MODEL"] = "HAIKU"
        # If the local model answered instead of writing a prompt, ask it to rewrite
        rewrite_prompt = (f"The user explicitly asked to phone a friend (use the remote model). "
                          f"Their original message was: \"{user_input}\"\n"
                          f"Write a CLEAR INSTRUCTION for the remote model. "
                          f"Do NOT answer the question yourself. "
                          f"Include relevant conversation context if needed. "
                          f"Set next_step to SEND_TO_REMOTE.")
        rewrite_raw = call_ollama(rewrite_prompt)
        try:
            rewrite_parsed = parse_local_response(rewrite_raw)
            parsed["OUTPUT"] = rewrite_parsed["OUTPUT"]
            parsed["CONTEXT_SUMMARY"] = rewrite_parsed.get("CONTEXT_SUMMARY", "")
            if rewrite_parsed.get("MODEL", "NONE").upper() != "NONE":
                parsed["MODEL"] = rewrite_parsed["MODEL"]
        except (ValueError, json.JSONDecodeError):
            pass  # keep original output as fallback

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
        local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        print(f"\n{DIM}[{local_model}  •  !llama !deepseek !remote]{RESET}")
        print(f"{BOLD}{output}{RESET}")
        return

    if next_step == "ASK_USER":
        local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        print(f"\n{DIM}[{local_model}  •  !llama !deepseek !remote]{RESET}")
        print(f"{YELLOW}{parsed['OUTPUT']}{RESET}")
        return

    if next_step == "SEND_TO_REMOTE":
        if parsed.get("SENSITIVE_DATA", "").upper() == "YES":
            print(f"\n{RED}[SAFETY] Sensitive data detected — sanitized prompt being sent.{RESET}")

        model_choice = parsed.get("MODEL", "HAIKU").upper().strip()

        # --- Hard guardrail: OPUS and GPT_PRO require explicit user request + passphrase ---
        if model_choice in ("OPUS", "GPT_PRO"):
            provider_label = "Anthropic" if model_choice == "OPUS" else "OpenAI"
            downgrade = "SONNET" if model_choice == "OPUS" else "GPT"
            # Did the user actually ask for this model?
            user_requested_expensive = (
                (model_choice == "OPUS" and "use opus" in user_input.lower())
                or (model_choice == "GPT_PRO" and "use gpt pro" in user_input.lower())
            )
            if user_requested_expensive:
                # User asked for it — confirm with passphrase
                print(f"\n{RED}⚠  {model_choice} is the most expensive {provider_label} model.{RESET}")
                print(f"{YELLOW}Type the phrase exactly: {BOLD}Money grows on trees.{RESET}")
                passphrase = input(f"   > ").strip()
                if passphrase != "Money grows on trees.":
                    print(f"{DIM}[Wrong phrase — downgrading to {downgrade}]{RESET}")
                    model_choice = downgrade
            else:
                # Local model picked it on its own — auto-downgrade, no prompt
                print(f"\n{YELLOW}[GUARDRAIL] Local model picked {model_choice} without user request — downgrading to {downgrade}{RESET}")
                model_choice = downgrade

        model_id = MODEL_MAP.get(model_choice, MODEL_MAP["HAIKU"])
        remote_prompt = parsed["OUTPUT"]
        context_summary = parsed.get("CONTEXT_SUMMARY", "")
        cost = COST_INFO.get(model_choice, "unknown cost")

        # --- Local escalation: try a bigger local model before going remote ---
        escalation_model = get_env("OLLAMA_ESCALATION_MODEL", "")
        if escalation_model and not force_remote:
            print(f"\n{CYAN}🧠 Try {escalation_model} locally before phoning a friend?{RESET}")
            escalate_local = input(f"   {BOLD}[y]{RESET} Run locally / {BOLD}[n]{RESET} Skip to remote > ").strip().lower()
            if escalate_local not in ("y", "yes"):
                print(f"{DIM}[Skipping local escalation]{RESET}")
            else:
                print(f"\n{CYAN}--- Local escalation ({escalation_model}) ---{RESET}")
                # Build the same context-rich prompt we'd send to the remote model
                digest = build_conversation_digest()
                escalation_prompt_parts: list[str] = []
                if digest:
                    escalation_prompt_parts.append(digest)
                escalation_prompt_parts.append(
                    f"The user asked: \"{user_input}\"\n\n"
                    f"Answer this question directly and thoroughly. "
                    f"If you truly cannot answer (e.g. it requires real-time data "
                    f"you don't have), say exactly: \"I cannot answer this.\""
                )
                escalation_prompt = "\n\n".join(escalation_prompt_parts)

                escalation_response = call_ollama_direct(
                    escalation_prompt, model=escalation_model, show_stream=True
                )

                # If deepseek gave a real answer (not a punt), use it and skip remote
                punt_phrases = ["i cannot answer", "i don't have", "i'm not able to",
                                "i don't know", "beyond my knowledge", "no information",
                                "i'm unable to", "i cannot provide"]
                if escalation_response and not any(
                    phrase in escalation_response.lower() for phrase in punt_phrases
                ):
                    print(f"\n{DIM}[{escalation_model}  •  !llama !deepseek !remote]{RESET}")
                    print(f"{BOLD}{escalation_response}{RESET}")
                    return
                else:
                    print(f"{DIM}[Escalation model punted — proceeding to remote]{RESET}")

        # --- Phone a Friend confirmation gate ---
        if not force_remote:
            print(f"\n{MAGENTA}📞 Phone a Friend?{RESET}  ({model_choice} — {cost})")
            preview = remote_prompt[:200] + ("..." if len(remote_prompt) > 200 else "")
            print(f'{DIM}   "{preview}"{RESET}')
            choice = input(f"   {BOLD}[y]{RESET} Send / {BOLD}[n]{RESET} Answer locally / {BOLD}[more]{RESET} Need more context > ").strip().lower()

            if choice in ("n", "no"):
                # Ask local model to answer it instead
                local_retry_prompt = (f"The user asked: \"{user_input}\"\n"
                                      f"You suggested sending this to the remote model, but the user "
                                      f"declined. Answer the question yourself to the best of your ability. "
                                      f"Set next_step to RESPOND_LOCALLY.")
                print(f"\n{CYAN}--- Local model (retry) ---{RESET}")
                retry_raw = call_ollama(local_retry_prompt, show_stream=True)
                try:
                    retry_parsed = parse_local_response(retry_raw)
                    local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
                    print(f"\n{DIM}[{local_model}  •  !llama !deepseek !remote]{RESET}")
                    print(f"{BOLD}{retry_parsed['OUTPUT']}{RESET}")
                except (ValueError, json.JSONDecodeError):
                    print(f"\n{YELLOW}[NOTE] Couldn't parse retry — here's the raw response{RESET}")
                    print(retry_raw)
                return

            if choice in ("more", "m"):
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

            if choice not in ("y", "yes"):
                # Empty Enter or unrecognized input — don't send, treat as cancel
                print(f"{DIM}[Cancelled — press y to send]{RESET}")
                return

        # Build the full remote prompt with real conversation context
        # Python builds this from actual history — not relying on the local
        # model's often-vague context_summary field
        digest = build_conversation_digest()
        parts: list[str] = []
        if digest:
            parts.append(digest)
        if context_summary:
            parts.append(f"Local model's note: {context_summary}")
        parts.append(remote_prompt)
        full_remote_prompt = "\n\n".join(parts)

        provider = MODEL_PROVIDER.get(model_choice, "anthropic")
        print(f"\n{MAGENTA}--- Remote model ({model_choice} via {provider}) ---{RESET}")
        print(f"{DIM}[Sending {len(full_remote_prompt)} chars to {model_id}]{RESET}")
        print(f"{DIM}[PROMPT]: {remote_prompt}{RESET}")
        print()
        if provider == "openai":
            remote_response = call_openai_stream(full_remote_prompt, model=model_id)
        else:
            remote_response = call_anthropic_stream(full_remote_prompt, model=model_id)

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
        normalized_response = remote_response.lower().replace("\u2019", "'").replace("\u2018", "'")
        is_cheap_model = model_choice in ("HAIKU", "GPT_MINI")
        if is_cheap_model and any(phrase in normalized_response for phrase in dont_know_phrases):
            # Escalate within the same provider
            if provider == "openai":
                upgrade_model = "GPT"
                upgrade_id = MODEL_MAP["GPT"]
                upgrade_cost = COST_INFO["GPT"]
            else:
                upgrade_model = "SONNET"
                upgrade_id = MODEL_MAP["SONNET"]
                upgrade_cost = COST_INFO["SONNET"]
            escalate = input(f"\n{YELLOW}{model_choice} couldn't answer. Escalate to {upgrade_model} ({upgrade_cost} cost)? [y/n] > {RESET}").strip().lower()
            if escalate in ("y", "yes"):
                print(f"\n{MAGENTA}--- Escalating to {upgrade_model} ({upgrade_id}) ---{RESET}")
                if provider == "openai":
                    remote_response = call_openai_stream(full_remote_prompt, model=upgrade_id)
                else:
                    remote_response = call_anthropic_stream(full_remote_prompt, model=upgrade_id)
            else:
                print(f"{DIM}[Keeping Haiku response]{RESET}")

        # Inject Claude's response into Ollama's conversation history
        conversation_history.append({
            "role": "assistant",
            "content": f"[Remote model ({model_choice}) responded]: {remote_response}"
        })
        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]

        return

    print(f"\n{RED}[ERROR] Unexpected routing: {next_step}{RESET}")


def main() -> None:
    local_model = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    escalation_model = get_env("OLLAMA_ESCALATION_MODEL", "")
    print(f"{BOLD}{CYAN}Relay v2: Local (Ollama) → Remote (Claude / GPT){RESET}")
    print(f"{DIM}Shortcuts:  !llama = {local_model}  |  !deepseek = {escalation_model or 'not configured'}  |  !remote / phone a friend = API{RESET}")
    print(f"{DIM}Commands:   exit, clear{RESET}\n")

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

        # Detect explicit model triggers
        force_remote = False
        force_deepseek = False
        force_llama = False
        lower_input = user_input.lower()

        # !deepseek — skip llama, go straight to deepseek
        if "!deepseek" in lower_input:
            force_deepseek = True
            user_input = user_input[:lower_input.index("!deepseek")] + user_input[lower_input.index("!deepseek") + len("!deepseek"):]
            user_input = user_input.strip()
            if not user_input:
                user_input = "Say hello and ask what the user would like to discuss."

        # !llama — force local model (no escalation, no remote)
        elif "!llama" in lower_input:
            force_llama = True
            user_input = user_input[:lower_input.index("!llama")] + user_input[lower_input.index("!llama") + len("!llama"):]
            user_input = user_input.strip()
            if not user_input:
                user_input = "Say hello and ask what the user would like to discuss."

        else:
            for trigger in ["phone a friend", "!remote"]:
                if trigger in lower_input:
                    force_remote = True
                    user_input = user_input[:lower_input.index(trigger)] + user_input[lower_input.index(trigger) + len(trigger):]
                    user_input = user_input.strip()
                    if not user_input:
                        user_input = "The user wants to phone a friend but didn't specify a question. Ask them what they'd like to send to the remote model."
                    break

        # !deepseek — send directly to escalation model, skip everything else
        if force_deepseek:
            escalation_model = get_env("OLLAMA_ESCALATION_MODEL", "")
            if not escalation_model:
                print(f"{YELLOW}No escalation model configured (OLLAMA_ESCALATION_MODEL is empty){RESET}")
            else:
                digest = build_conversation_digest()
                parts: list[str] = []
                if digest:
                    parts.append(digest)
                parts.append(user_input)
                full_prompt = "\n\n".join(parts)
                print(f"\n{CYAN}--- {escalation_model} ---{RESET}")
                response = call_ollama_direct(full_prompt, model=escalation_model, show_stream=True)
                if response:
                    print(f"\n{DIM}[{escalation_model}  •  !llama !deepseek !remote]{RESET}")
                    print(f"{BOLD}{response}{RESET}")
                else:
                    print(f"{YELLOW}[No response from {escalation_model}]{RESET}")
            continue

        # !llama — send directly to local model, force RESPOND_LOCALLY
        if force_llama:
            try:
                print(f"\n{CYAN}--- Local model ---{RESET}")
                local_raw = call_ollama(user_input, show_stream=True)
                parsed = parse_local_response(local_raw)
                local_model_name = get_env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
                print(f"\n{DIM}[{local_model_name}  •  !llama !deepseek !remote]{RESET}")
                print(f"{BOLD}{parsed['OUTPUT']}{RESET}")
            except (ValueError, json.JSONDecodeError):
                print(f"{YELLOW}[Couldn't parse response]{RESET}")
            continue

        try:
            relay_once(user_input, force_remote=force_remote)
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
