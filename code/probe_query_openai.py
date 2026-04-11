"""
probe_query_openai.py
─────────────────────────────────────────────────────────────────────────────
OpenAI-compatible query layer for providers that use /v1/chat/completions
rather than the Ollama /api/chat wire format.

Currently supported backends:
  xai   — xAI Grok API (api.x.ai)
          Key : XAI_API_KEY in ~/.env

Adding a new provider:
  1. Add an entry to PROVIDER_CONFIG with base_url and env_key
  2. Add the backend string to probe_models.json entries
  3. No other code changes required

Design notes:
  - generate endpoint does not exist for OpenAI-compatible APIs.
    Coverage is chat-only for all backends handled here.
    This is documented in probe_models.json notes and is consistent with
    how Falcon3 (chat-only due to template bug) is handled in the corpus.
  - Think blocks: xAI's reasoning_effort parameter is not used here.
    We probe base chat behaviour, consistent with how Ollama cloud models
    are probed (no special reasoning flags).
  - Tool calls: probe_tool_capable.py routes xAI models here via
    query_chat_openai() with tools= kwarg.

Usage (from probe_static.py or probe_endpoint_sweep.py):
  from probe_query_openai import is_openai_backend, query_chat_openai, openai_headers

  if is_openai_backend(model_entry):
      result = query_chat_openai(model_entry, question, timeout)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path.home() / ".env")

# ── Provider registry ──────────────────────────────────────────────────────────

PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "env_key":  "XAI_API_KEY",
        "label":    "xAI",
    },
}

# Backend strings that are handled by this module
OPENAI_BACKENDS: set[str] = set(PROVIDER_CONFIG.keys())

# Default model names per backend — used when probe_models.json omits api_model_id
DEFAULT_MODEL_IDS: dict[str, str] = {
    "xai": "grok-4.20-0309-reasoning",
}


# ── Routing helpers ────────────────────────────────────────────────────────────

def is_openai_backend(model_entry: dict) -> bool:
    """Return True if this registry entry should route through this module."""
    return model_entry.get("backend", "") in OPENAI_BACKENDS


def backend_for(model_entry: dict) -> str:
    return model_entry.get("backend", "")


def base_url_for(model_entry: dict) -> str:
    backend = backend_for(model_entry)
    cfg = PROVIDER_CONFIG.get(backend, {})
    return cfg.get("base_url", "")


def api_key_for(model_entry: dict) -> str:
    backend = backend_for(model_entry)
    cfg = PROVIDER_CONFIG.get(backend, {})
    env_key = cfg.get("env_key", "")
    return os.environ.get(env_key, "")


def api_model_id_for(model_entry: dict) -> str:
    """
    Return the model ID string to send in the API request.
    Uses api_model_id from the registry entry if present,
    otherwise falls back to DEFAULT_MODEL_IDS.
    """
    explicit = model_entry.get("api_model_id", "")
    if explicit:
        return explicit
    backend = backend_for(model_entry)
    return DEFAULT_MODEL_IDS.get(backend, model_entry.get("name", ""))


def openai_headers(model_entry: dict) -> dict[str, str]:
    key = api_key_for(model_entry)
    if not key:
        return {}
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
    }


def has_api_key(model_entry: dict) -> bool:
    return bool(api_key_for(model_entry))


def provider_label(model_entry: dict) -> str:
    backend = backend_for(model_entry)
    return PROVIDER_CONFIG.get(backend, {}).get("label", backend)


# ── Think block extraction ─────────────────────────────────────────────────────

def extract_think(text: str) -> tuple[str, str]:
    """Extract <think>...</think> tags. Returns (think, clean_answer)."""
    think_blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return "\n\n".join(think_blocks).strip(), clean


# ── Core query ─────────────────────────────────────────────────────────────────

def query_chat_openai(
    model_entry: dict,
    question: str,
    timeout: int,
    tools: list[dict] | None = None,
    system: str | None = None,
) -> dict:
    """
    POST to /v1/chat/completions for OpenAI-compatible providers.

    Returns a result dict that matches the shape of probe_static.query_chat()
    so callers can treat both interchangeably:
      {
        "endpoint":     "chat",
        "success":      bool,
        "elapsed_s":    float,
        "raw_response": str,       # full content string
        "think_block":  str,       # extracted think content or ""
        "answer":       str,       # content with think tags stripped
        "think_format": str,       # "tag" | "none"
        "eval_count":   int|None,
        "prompt_eval_count": int|None,
        "backend":      str,       # e.g. "xai"
        "api_model_id": str,       # model ID sent to API
        "error":        str,       # present only on failure
      }

    On failure, "success" is False and "error" describes the problem.
    Tool call results (for probe_tool_capable.py) are included in
    "tool_calls" key when present.
    """
    base_url    = base_url_for(model_entry)
    headers     = openai_headers(model_entry)
    api_model   = api_model_id_for(model_entry)
    backend     = backend_for(model_entry)

    if not base_url:
        return _error_result(backend, api_model, f"No base_url configured for backend '{backend}'")

    if not headers:
        env_key = PROVIDER_CONFIG.get(backend, {}).get("env_key", "???")
        return _error_result(backend, api_model, f"API key missing — set {env_key} in ~/.env")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    payload: dict[str, Any] = {
        "model":    api_model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    url = f"{base_url}/chat/completions"
    t0  = time.perf_counter()

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = round(time.perf_counter() - t0, 2)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        return _error_result(backend, api_model, f"timeout after {timeout}s",
                             elapsed=round(time.perf_counter() - t0, 2))
    except requests.exceptions.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:200]
        except Exception:
            pass
        return _error_result(backend, api_model,
                             f"HTTP {exc.response.status_code}: {body}",
                             elapsed=round(time.perf_counter() - t0, 2))
    except Exception as exc:
        return _error_result(backend, api_model, str(exc)[:200],
                             elapsed=round(time.perf_counter() - t0, 2))

    # Parse OpenAI response shape
    choice  = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""

    # Tool calls (for probe_tool_capable.py)
    tool_calls_raw = message.get("tool_calls") or []

    # Usage
    usage = data.get("usage", {})

    think, answer = extract_think(content)

    result: dict[str, Any] = {
        "endpoint":          "chat",
        "success":           True,
        "elapsed_s":         elapsed,
        "raw_response":      content,
        "think_block":       think,
        "answer":            answer,
        "think_format":      "tag" if think else "none",
        "eval_count":        usage.get("completion_tokens"),
        "prompt_eval_count": usage.get("prompt_tokens"),
        "backend":           backend,
        "api_model_id":      api_model,
    }

    if tool_calls_raw:
        result["tool_calls"] = tool_calls_raw

    return result


# ── Generate stub ──────────────────────────────────────────────────────────────

def query_generate_openai(model_entry: dict, question: str, timeout: int) -> dict:
    """
    OpenAI-compatible providers do not expose a generate (base-weight) endpoint.
    Returns a structured failure result so callers can record chat-only coverage.
    """
    backend   = backend_for(model_entry)
    api_model = api_model_id_for(model_entry)
    return {
        "endpoint":     "generate",
        "success":      False,
        "elapsed_s":    0.0,
        "error":        f"generate endpoint not available for backend '{backend}' — chat-only coverage",
        "backend":      backend,
        "api_model_id": api_model,
        "cloud":        True,
    }


# ── Availability check ─────────────────────────────────────────────────────────

def check_api_key(model_entry: dict) -> tuple[bool, str]:
    """
    Verify the API key is present and the endpoint is reachable.
    Returns (ok, detail_message).
    """
    backend = backend_for(model_entry)
    cfg     = PROVIDER_CONFIG.get(backend, {})
    env_key = cfg.get("env_key", "")
    key     = api_key_for(model_entry)

    if not key:
        return False, f"{env_key} not set in ~/.env"

    base_url = base_url_for(model_entry)
    if not base_url:
        return False, f"No base_url configured for backend '{backend}'"

    # Lightweight reachability check — list models endpoint
    try:
        r = requests.get(
            f"{base_url}/models",
            headers=openai_headers(model_entry),
            timeout=10,
        )
        if r.status_code == 200:
            model_ids = [m.get("id", "") for m in r.json().get("data", [])]
            return True, f"key present, {len(model_ids)} model(s) available"
        return False, f"HTTP {r.status_code} from {base_url}/models"
    except requests.exceptions.Timeout:
        return False, f"timeout reaching {base_url}"
    except Exception as exc:
        return False, str(exc)[:120]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _error_result(
    backend: str,
    api_model: str,
    error: str,
    elapsed: float = 0.0,
) -> dict:
    return {
        "endpoint":          "chat",
        "success":           False,
        "elapsed_s":         elapsed,
        "error":             error,
        "raw_response":      "",
        "think_block":       "",
        "answer":            "",
        "think_format":      "none",
        "eval_count":        None,
        "prompt_eval_count": None,
        "backend":           backend,
        "api_model_id":      api_model,
    }
