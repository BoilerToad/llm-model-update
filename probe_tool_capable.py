#!/usr/bin/env python3
"""
probe_tool_capable.py
─────────────────────
Tests whether a model makes valid structured tool calls when given a tools
schema and a question that rewards tool use.

Sends /api/chat with a web_fetch tool definition and Q03 (the strongest
tool-call trigger in the question bank). Inspects the response for a
structured tool_calls block — not just text describing a tool call.

Classifies the result as:
  TOOL_CALL   — model returned a valid structured tool call
  TEXT_ONLY   — model answered in plain text without calling any tool
  REFUSED     — model declined to answer
  ERROR       — request failed (timeout, HTTP error, etc.)

Updates probe_models.json tool_capable field with True/False.

Usage:
    python probe_tool_capable.py --model gemma4:e4b
    python probe_tool_capable.py --model gemma4:e4b --dry-run
    python probe_tool_capable.py --all-enabled
    python probe_tool_capable.py --model gemma4:e4b --timeout 60
    python probe_tool_capable.py --model gemma4:e4b --question-id Q11
    python probe_tool_capable.py --list-questions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import os

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

OLLAMA_LOCAL   = "http://localhost:11434"
OLLAMA_CLOUD   = "https://ollama.com"
SCRIPT_DIR     = Path(__file__).parent
MODELS_FILE    = SCRIPT_DIR / "probes" / "probe_models.json"
QUESTIONS_FILE = SCRIPT_DIR / "probes" / "questions.json"

DEFAULT_QUESTION_ID = "Q03"

# Hardcoded tool-trigger question — used by default.
# Designed to be unanswerable from weights alone, forcing a tool call
# in any model that supports structured tool use. The explicit instruction
# removes ambiguity: a capable model has no reason not to call the tool.
TOOL_TRIGGER_QUESTION = (
    "What is the current EUR/USD exchange rate? "
    "Use the web_fetch tool to look it up and report the exact value."
)

# Tool schema sent with every request — a simple web fetch tool
WEB_FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch the current content of a web page or news source. "
            "Use this to retrieve up-to-date information on recent events."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
}

# Classification labels
TOOL_CALL  = "TOOL_CALL"
TEXT_ONLY  = "TEXT_ONLY"
REFUSED    = "REFUSED"
ERROR      = "ERROR"

# Refusal signal words (conservative — only clear refusals)
REFUSAL_SIGNALS = [
    "i cannot", "i'm unable", "i am unable",
    "i don't have access", "i can't access",
    "as an ai", "i don't have the ability",
]


# ── Cloud routing ─────────────────────────────────────────────────────────────

def is_cloud(model_name: str) -> bool:
    return "cloud" in model_name.lower()


def llm_url_for(model_name: str) -> str:
    return OLLAMA_CLOUD if is_cloud(model_name) else OLLAMA_LOCAL


def auth_headers(llm_url: str) -> dict:
    if llm_url == OLLAMA_CLOUD:
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


# ── Question loading ───────────────────────────────────────────────────────────

def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    """Load all questions from questions.json."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Questions file is not valid JSON: {path}\n  {e}")
    return data.get("questions", [])


def load_question_by_id(
    question_id: str,
    path: Path = QUESTIONS_FILE,
) -> tuple[str, str]:
    """
    Return (question_text, prompt_text) for the given question ID.
    Lookup is case-insensitive. Raises ValueError if not found.
    """
    questions = load_questions(path)
    target = question_id.upper()
    for q in questions:
        if q["id"].upper() == target:
            text   = q["text"]
            prompt = f"Question: {text}\n\nAnswer:"
            return text, prompt
    available = ", ".join(q["id"] for q in questions)
    raise ValueError(
        f"Question ID '{question_id}' not found. "
        f"Available: {available}"
    )


# ── Registry helpers ───────────────────────────────────────────────────────────

def load_registry(path: Path = MODELS_FILE) -> dict:
    """Return models as {name: entry_dict}."""
    data = json.loads(path.read_text())
    return {m["name"]: m for m in data.get("models", [])}


def save_tool_capable(
    model: str,
    value: bool,
    path: Path = MODELS_FILE,
) -> None:
    """Write tool_capable field for a model into probe_models.json."""
    raw  = json.loads(path.read_text())
    for entry in raw.get("models", []):
        if entry["name"] == model:
            entry["tool_capable"] = value
            break
    path.write_text(json.dumps(raw, indent=2))


# ── Core test ─────────────────────────────────────────────────────────────────

class ToolTestResult:
    """Result of a single tool-capability test."""

    def __init__(self, model: str):
        self.model          = model
        self.classification = ERROR
        self.elapsed_s      = 0.0
        self.tool_name      = None   # name of tool called, if any
        self.tool_args      = None   # args dict, if any
        self.response_len   = 0
        self.snippet        = ""
        self.error          = None
        self.think_len      = 0

    @property
    def tool_capable(self) -> bool | None:
        """True if a valid tool call was made, False if text-only or refused, None on error."""
        if self.classification == TOOL_CALL:
            return True
        if self.classification in (TEXT_ONLY, REFUSED):
            return False
        return None

    def __str__(self) -> str:
        label = {
            TOOL_CALL: "✓ TOOL_CALL",
            TEXT_ONLY: "✗ TEXT_ONLY",
            REFUSED:   "✗ REFUSED",
            ERROR:     "✗ ERROR",
        }.get(self.classification, self.classification)

        parts = [f"  {label:<14} {self.elapsed_s:>6.2f}s"]
        if self.classification == TOOL_CALL:
            parts.append(f"  tool={self.tool_name}  args={self.tool_args}")
        elif self.classification == ERROR:
            parts.append(f"  error={self.error}")
        else:
            parts.append(f"  len={self.response_len}c")
            if self.snippet:
                parts.append(f"  snippet: {self.snippet[:80]}")
        if self.think_len:
            parts.append(f"  [think:{self.think_len}c]")
        return "\n".join(parts)


def classify_response(data: dict) -> tuple[str, str | None, dict | None]:
    """
    Inspect an /api/chat response dict and return (classification, tool_name, tool_args).

    Ollama surfaces tool calls in message.tool_calls as a list of dicts:
        {"function": {"name": str, "arguments": dict}}
    """
    message    = data.get("message", {})
    tool_calls = message.get("tool_calls", [])

    if tool_calls:
        first     = tool_calls[0]
        fn        = first.get("function", {})
        tool_name = fn.get("name")
        tool_args = fn.get("arguments", {})
        return TOOL_CALL, tool_name, tool_args

    # No tool call — inspect plain text content
    content = message.get("content", "").strip()
    if not content:
        return ERROR, None, None

    lower = content.lower()
    if any(signal in lower for signal in REFUSAL_SIGNALS):
        return REFUSED, None, None

    return TEXT_ONLY, None, None


def run_tool_capable(
    model: str,
    question: str,
    timeout: int = 120,
) -> ToolTestResult:
    """
    Send one /api/chat request with a tools schema and classify the response.
    Routes cloud models (name contains 'cloud') to OLLAMA_CLOUD with Bearer auth.
    """
    result  = ToolTestResult(model)
    llm_url = llm_url_for(model)
    payload = {
        "model":    model,
        "stream":   False,
        "messages": [{"role": "user", "content": question}],
        "tools":    [WEB_FETCH_TOOL],
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{llm_url}/api/chat",
            json=payload,
            headers=auth_headers(llm_url),
            timeout=timeout,
        )
        result.elapsed_s = round(time.perf_counter() - t0, 3)

        if r.status_code != 200:
            result.error = f"HTTP {r.status_code}: {r.text[:120]}"
            return result

        data = r.json()
        classification, tool_name, tool_args = classify_response(data)

        result.classification = classification
        result.tool_name      = tool_name
        result.tool_args      = tool_args

        content  = data.get("message", {}).get("content", "")
        thinking = data.get("message", {}).get("thinking", "")
        result.response_len = len(content)
        result.snippet      = content[:120].replace("\n", " ")
        result.think_len    = len(thinking)

    except requests.exceptions.Timeout:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = f"TIMEOUT after {timeout}s"
    except Exception as e:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = str(e)[:120]

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether a model makes structured tool calls via /api/chat"
    )
    parser.add_argument("--model",          metavar="NAME",
                        help="Single model to test")
    parser.add_argument("--all-enabled",    action="store_true",
                        help="Test all enabled models from probe_models.json")
    parser.add_argument("--timeout",        type=int, default=120,
                        help="Timeout per request in seconds (default: 120)")
    parser.add_argument("--question-id",    metavar="ID", default=None,
                        help="Question ID from questions.json. Overrides the default tool-trigger question.")
    parser.add_argument("--list-questions", action="store_true",
                        help="List available question IDs and exit")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Run test but do not write results to probe_models.json")
    args = parser.parse_args()

    # List questions and exit
    if args.list_questions:
        questions = load_questions()
        print("\n  Available questions:\n")
        for q in questions:
            tags = ", ".join(q.get("tags", []))
            print(f"  {q['id']:<6} [{q['theme']}] {tags}")
            print(f"         {q['text'][:80]}")
        print()
        sys.exit(0)

    if not args.model and not args.all_enabled:
        parser.error("Specify --model MODEL or --all-enabled")

    # Load question — use hardcoded tool-trigger by default, question bank if --question-id given
    if args.question_id:
        try:
            question_text, _ = load_question_by_id(args.question_id)
        except ValueError as e:
            print(f"\n✗ {e}")
            sys.exit(1)
    else:
        question_text = TOOL_TRIGGER_QUESTION
        args.question_id = "(tool-trigger)"

    # Build model list
    if args.model:
        models = [args.model]
    else:
        registry = load_registry()
        models = [
            name for name, entry in registry.items()
            if entry.get("enabled", True)
        ]

    # Check Ollama is reachable — local for local models, cloud for cloud-only runs
    check_url = OLLAMA_CLOUD if (models and all(is_cloud(m) for m in models)) else OLLAMA_LOCAL
    try:
        r = requests.get(f"{check_url}/api/version", headers=auth_headers(check_url), timeout=5)
        version = r.json().get("version", "?")
        print(f"\n  Ollama v{version} at {check_url}")
    except Exception as e:
        print(f"✗ Cannot reach Ollama: {e}")
        sys.exit(1)

    print(f"  Timeout     : {args.timeout}s")
    print(f"  Question ID : {args.question_id}")
    print(f"  Question    : {question_text[:80]}")
    print(f"  Tool        : {WEB_FETCH_TOOL['function']['name']}")
    print(f"  Dry run     : {args.dry_run}")
    print(f"  Started     : {datetime.now():%Y-%m-%d %H:%M:%S}")

    print(f"\n{'═'*65}")
    results: dict[str, ToolTestResult] = {}

    for model in models:
        print(f"\n  Model : {model}")
        print(f"  {'─'*60}")
        print(f"  Testing tool_capable ...", end=" ", flush=True)

        result = run_tool_capable(model, question_text, args.timeout)
        results[model] = result

        print(f"{result.elapsed_s:.2f}s")
        print(str(result))

        if result.tool_capable is not None and not args.dry_run:
            save_tool_capable(model, result.tool_capable)
            print(f"\n  → probe_models.json updated: tool_capable={result.tool_capable}")
        elif args.dry_run:
            print(f"\n  → dry-run: would set tool_capable={result.tool_capable}")

    # Summary table
    print(f"\n{'═'*65}")
    print(f"  SUMMARY\n")
    print(f"  {'Model':<45} {'Result':<14} {'tool_capable'}")
    print(f"  {'─'*45} {'─'*14} {'─'*12}")

    for model, result in results.items():
        capable_str = {True: "true", False: "false", None: "null (error)"}.get(
            result.tool_capable, "?"
        )
        print(f"  {model:<45} {result.classification:<14} {capable_str}")

    print()
    if args.dry_run:
        print("  (dry-run — no changes written to probe_models.json)\n")


if __name__ == "__main__":
    main()
