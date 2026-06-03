"""
probe_query_llamacpp.py
─────────────────────────────────────────────────────────────────────────────
Query layer for a locally-running llama.cpp server (llama-server).

Supported endpoints:
  /v1/chat/completions  — OpenAI-compatible chat (query_chat_llamacpp)
  /completion           — Native llama.cpp generate (query_generate_llamacpp)

Default base URL: http://127.0.0.1:8080
Override via model_entry["llamacpp_url"] or LLAMACPP_URL env variable.

No API key is required — the server is local and unauthenticated by default.

Adding a new llama.cpp instance:
  1. Set llamacpp_url in the probe_models.json entry if not on default port.
  2. Set backend: "llamacpp" in the registry entry.
  3. No other code changes required.

Usage (from probe_static.py or other callers):
  from probe_query_llamacpp import (
      is_llamacpp_backend,
      query_chat_llamacpp,
      query_generate_llamacpp,
      check_server as check_llamacpp_server,
  )

  if is_llamacpp_backend(model_entry):
      chat_result = query_chat_llamacpp(model_entry, question, timeout)
      gen_result  = query_generate_llamacpp(model_entry, question, timeout)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests

# ── Constants ──────────────────────────────────────────────────────────────────

BACKEND_NAME    = "llamacpp"
DEFAULT_BASE_URL = os.environ.get("LLAMACPP_URL", "http://127.0.0.1:8080")


# ── Routing helpers ────────────────────────────────────────────────────────────

def is_llamacpp_backend(model_entry: dict) -> bool:
    """Return True if this registry entry should route through this module."""
    return model_entry.get("backend", "") == BACKEND_NAME


def base_url_for(model_entry: dict) -> str:
    """
    Resolve the llama.cpp server base URL for this entry.
    Priority: model_entry["llamacpp_url"] > LLAMACPP_URL env var > default.
    """
    return model_entry.get("llamacpp_url", DEFAULT_BASE_URL).rstrip("/")


def api_model_id_for(model_entry: dict) -> str:
    """
    llama.cpp /v1/chat/completions accepts any non-empty model string;
    the server ignores it (only one model is loaded at a time).
    Use api_model_id from the registry if present, otherwise the entry name.
    """
    return model_entry.get("api_model_id", model_entry.get("name", "local"))


# ── Think block extraction ─────────────────────────────────────────────────────

def _extract_think(text: str) -> tuple[str, str]:
    """Extract <think>...</think> tags. Returns (think_content, clean_answer)."""
    think_blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return "\n\n".join(think_blocks).strip(), clean


# ── Error helper ───────────────────────────────────────────────────────────────

def _error_result(endpoint: str, api_model: str, msg: str, elapsed: float = 0.0) -> dict:
    return {
        "endpoint":          endpoint,
        "success":           False,
        "elapsed_s":         elapsed,
        "error":             msg,
        "backend":           BACKEND_NAME,
        "api_model_id":      api_model,
    }


# ── Chat: /v1/chat/completions ────────────────────────────────────────────────

def query_chat_llamacpp(
    model_entry: dict,
    question: str,
    timeout: int,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """
    POST to /v1/chat/completions on the llama.cpp server.

    Returns a result dict matching the shape used by probe_static.query_chat()
    and probe_query_openai.query_chat_openai():
      {
        "endpoint":          "chat",
        "success":           bool,
        "elapsed_s":         float,
        "raw_response":      str,
        "think_block":       str,
        "answer":            str,
        "think_format":      "tag" | "none",
        "eval_count":        int | None,
        "prompt_eval_count": int | None,
        "backend":           "llamacpp",
        "api_model_id":      str,
        "error":             str,   # present only on failure
      }
    """
    base_url  = base_url_for(model_entry)
    api_model = api_model_id_for(model_entry)
    url       = f"{base_url}/v1/chat/completions"

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    payload: dict[str, Any] = {
        "model":       api_model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        elapsed = round(time.perf_counter() - t0, 2)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        return _error_result("chat", api_model,
                             f"timeout after {timeout}s",
                             elapsed=round(time.perf_counter() - t0, 2))
    except requests.exceptions.ConnectionError as exc:
        return _error_result("chat", api_model,
                             f"connection refused — is llama-server running at {base_url}? ({exc})",
                             elapsed=round(time.perf_counter() - t0, 2))
    except requests.exceptions.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:200]
        except Exception:
            pass
        return _error_result("chat", api_model,
                             f"HTTP {exc.response.status_code}: {body}",
                             elapsed=round(time.perf_counter() - t0, 2))
    except Exception as exc:
        return _error_result("chat", api_model,
                             str(exc)[:200],
                             elapsed=round(time.perf_counter() - t0, 2))

    choice  = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    usage   = data.get("usage", {})

    think, answer = _extract_think(content)

    return {
        "endpoint":          "chat",
        "success":           True,
        "elapsed_s":         elapsed,
        "raw_response":      content,
        "think_block":       think,
        "answer":            answer,
        "think_format":      "tag" if think else "none",
        "eval_count":        usage.get("completion_tokens"),
        "prompt_eval_count": usage.get("prompt_tokens"),
        "backend":           BACKEND_NAME,
        "api_model_id":      api_model,
    }


# ── Generate: /completion ──────────────────────────────────────────────────────

def query_generate_llamacpp(
    model_entry: dict,
    question: str,
    timeout: int,
    temperature: float = 0.7,
    n_predict: int = 4096,
) -> dict:
    """
    POST to /completion on the llama.cpp server (native generate endpoint).

    Uses the same Question/Answer prompt frame as probe_static.query_generate()
    for consistency across backends.

    Returns a result dict matching the shape used by probe_static.query_generate():
      {
        "endpoint":          "generate",
        "success":           bool,
        "elapsed_s":         float,
        "prompt_frame":      str,
        "raw_response":      str,
        "think_block":       str,
        "answer":            str,
        "think_format":      "tag" | "none",
        "eval_count":        int | None,
        "prompt_eval_count": int | None,
        "backend":           "llamacpp",
        "api_model_id":      str,
        "error":             str,   # present only on failure
      }
    """
    base_url  = base_url_for(model_entry)
    api_model = api_model_id_for(model_entry)
    url       = f"{base_url}/completion"

    framed = f"Question: {question}\n\nAnswer:"
    frame  = "Question/Answer (llama.cpp /completion)"

    payload: dict[str, Any] = {
        "prompt":      framed,
        "temperature": temperature,
        "n_predict":   n_predict,
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        elapsed = round(time.perf_counter() - t0, 2)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        return _error_result("generate", api_model,
                             f"timeout after {timeout}s",
                             elapsed=round(time.perf_counter() - t0, 2))
    except requests.exceptions.ConnectionError as exc:
        return _error_result("generate", api_model,
                             f"connection refused — is llama-server running at {base_url}? ({exc})",
                             elapsed=round(time.perf_counter() - t0, 2))
    except requests.exceptions.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:200]
        except Exception:
            pass
        return _error_result("generate", api_model,
                             f"HTTP {exc.response.status_code}: {body}",
                             elapsed=round(time.perf_counter() - t0, 2))
    except Exception as exc:
        return _error_result("generate", api_model,
                             str(exc)[:200],
                             elapsed=round(time.perf_counter() - t0, 2))

    # llama.cpp /completion response shape:
    #   { "content": "...", "tokens_predicted": N, "tokens_evaluated": N, ... }
    content = data.get("content", "")
    think, answer = _extract_think(content)

    return {
        "endpoint":          "generate",
        "success":           True,
        "elapsed_s":         elapsed,
        "prompt_frame":      frame,
        "raw_response":      content,
        "think_block":       think,
        "answer":            answer,
        "think_format":      "tag" if think else "none",
        "eval_count":        data.get("tokens_predicted"),
        "prompt_eval_count": data.get("tokens_evaluated"),
        "backend":           BACKEND_NAME,
        "api_model_id":      api_model,
    }


# ── Availability check ─────────────────────────────────────────────────────────

def check_server(model_entry: dict, timeout: int = 10) -> tuple[bool, str]:
    """
    Verify the llama.cpp server is reachable and healthy.

    Tries GET /health first (llama-server v0.0.2228+), then falls back to
    GET / (older builds). Returns (ok, detail_message).
    """
    base_url = base_url_for(model_entry)

    for path in ("/health", "/"):
        try:
            r = requests.get(f"{base_url}{path}", timeout=timeout)
            if r.status_code == 200:
                # /health returns {"status": "ok"} when the model is loaded
                data   = {}
                try:
                    data = r.json()
                except Exception:
                    pass
                status = data.get("status", "unknown")
                if path == "/health":
                    if status == "ok":
                        return True, f"server healthy at {base_url}"
                    if status == "loading model":
                        return False, f"server at {base_url} is still loading the model"
                    return True, f"server reachable at {base_url} (status: {status})"
                return True, f"server reachable at {base_url}"
            if r.status_code == 503:
                return False, f"server at {base_url} returned 503 — model may still be loading"
        except requests.exceptions.ConnectionError:
            return False, f"connection refused — is llama-server running at {base_url}?"
        except requests.exceptions.Timeout:
            return False, f"timeout reaching {base_url}"
        except Exception as exc:
            return False, str(exc)[:120]

    return False, f"server unreachable at {base_url}"
