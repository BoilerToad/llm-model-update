"""
probe_query_omlx.py
─────────────────────────────────────────────────────────────────────────────
Query layer for a locally-running oMLX server (Apple Silicon optimized).

oMLX is an OpenAI-compatible inference server for Apple Silicon (M1/M2/M3/M4).
Built on Apple's MLX framework, it provides high-performance LLM inference
with automatic model discovery.

Supported endpoints:
  /v1/chat/completions  — OpenAI-compatible chat (query_chat_omlx)
  /completion           — Base-weight generation (query_generate_omlx)

Default base URL: http://localhost:8000/v1
Override via model_entry["omlx_url"] or OMLX_URL env variable.

No API key is required — the server is local and unauthenticated by default.
Optional: Set OMLX_API_KEY env variable if running with authentication.

Auto-discovery: oMLX discovers models from subdirectories automatically.
Chat UI available at: http://localhost:8000/admin/chat

Adding a new oMLX instance:
  1. Set omlx_url in the probe_models.json entry if not on default port.
  2. Set backend: "omlx" in the registry entry.
  3. No other code changes required.

Usage (from probe_static.py or other callers):
  from probe_query_omlx import (
      is_omlx_backend,
      query_chat_omlx,
      query_generate_omlx,
      check_server as check_omlx_server,
  )

  if is_omlx_backend(model_entry):
      chat_result = query_chat_omlx(model_entry, question, timeout)
      gen_result  = query_generate_omlx(model_entry, question, timeout)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

# ── Constants ──────────────────────────────────────────────────────────────────

BACKEND_NAME     = "omlx"
DEFAULT_BASE_URL = os.environ.get("OMLX_URL", "http://localhost:8000/v1")
OMLX_API_KEY     = os.environ.get("OMLX_API_KEY", "")


# ── Routing helpers ────────────────────────────────────────────────────────────

def is_omlx_backend(model_entry: dict) -> bool:
    """Return True if this registry entry should route through this module."""
    return model_entry.get("backend", "") == BACKEND_NAME


def base_url_for(model_entry: dict) -> str:
    """
    Resolve the oMLX server base URL for this entry.
    Priority: model_entry["omlx_url"] > OMLX_URL env var > default.
    """
    return model_entry.get("omlx_url", DEFAULT_BASE_URL).rstrip("/")


def api_model_id_for(model_entry: dict) -> str:
    """
    oMLX auto-discovers models from subdirectories.
    Use api_model_id from the registry if present, otherwise the entry name.
    """
    return model_entry.get("api_model_id", model_entry.get("name", ""))


def headers_for(model_entry: dict) -> dict[str, str] | None:
    """
    Build headers for oMLX request.
    Returns None if no API key is configured (default for local server).
    """
    api_key = model_entry.get("omlx_api_key", OMLX_API_KEY)
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return None


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
        "raw_response":      "",
        "think_block":       "",
        "answer":            "",
        "think_format":      "none",
        "eval_count":        None,
        "prompt_eval_count": None,
        "backend":           BACKEND_NAME,
        "api_model_id":      api_model,
    }


# ── Chat: /v1/chat/completions ────────────────────────────────────────────────

def query_chat_omlx(
    model_entry: dict,
    question: str,
    timeout: int,
    raw: bool = False,
    tools: list | None = None,
) -> dict:
    """
    Query oMLX chat endpoint: /v1/chat/completions
    
    Args:
        model_entry: Registry entry with omlx_url, api_model_id fields
        question:    User prompt
        timeout:     Request timeout in seconds
        raw:         Ignored for oMLX (chat endpoint only, no raw mode)
        tools:       Optional list of OpenAI tool definitions
    
    Returns:
        Result dict with success, content, elapsed_s, endpoint, etc.
    """
    base     = base_url_for(model_entry)
    api_model = api_model_id_for(model_entry)
    url      = f"{base}/chat/completions"
    headers  = headers_for(model_entry)
    
    payload: dict[str, Any] = {
        "model":    api_model,
        "stream":   False,
        "messages": [{"role": "user", "content": question}],
    }
    
    if tools:
        payload["tools"] = tools
    
    t0 = time.perf_counter()
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.perf_counter() - t0
        
        if r.status_code != 200:
            return _error_result(
                "chat",
                api_model,
                f"HTTP {r.status_code}: {r.text[:200]}",
                elapsed,
            )
        
        data = r.json()
        
        # OpenAI-compatible response format
        choices = data.get("choices", [])
        if not choices:
            return _error_result("chat", api_model, "Empty choices array", elapsed)
        
        message = choices[0].get("message", {})
        
        # Check for tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return {
                "endpoint":          "chat",
                "success":           True,
                "elapsed_s":         elapsed,
                "raw_response":      "",
                "think_block":       "",
                "answer":            "",
                "think_format":      "none",
                "eval_count":        None,
                "prompt_eval_count": None,
                "tool_calls":        tool_calls,
                "backend":           BACKEND_NAME,
                "api_model_id":      api_model,
            }
        
        # Extract text content
        content = message.get("content", "")
        
        # Check for separate thinking field (some models return this)
        thinking = data.get("thinking", "")
        if thinking:
            return {
                "endpoint":          "chat",
                "success":           True,
                "elapsed_s":         elapsed,
                "raw_response":      content,
                "think_block":       thinking,
                "answer":            content,
                "think_format":      "field",
                "eval_count":        None,
                "prompt_eval_count": None,
                "backend":           BACKEND_NAME,
                "api_model_id":      api_model,
            }
        
        # Check for inline <think> tags
        think_content, clean_answer = _extract_think(content)
        if think_content:
            return {
                "endpoint":          "chat",
                "success":           True,
                "elapsed_s":         elapsed,
                "raw_response":      content,
                "think_block":       think_content,
                "answer":            clean_answer,
                "think_format":      "tag",
                "eval_count":        None,
                "prompt_eval_count": None,
                "backend":           BACKEND_NAME,
                "api_model_id":      api_model,
            }
        
        # Standard response
        return {
            "endpoint":          "chat",
            "success":           True,
            "elapsed_s":         elapsed,
            "raw_response":      content,
            "think_block":       "",
            "answer":            content,
            "think_format":      "none",
            "eval_count":        None,
            "prompt_eval_count": None,
            "backend":           BACKEND_NAME,
            "api_model_id":      api_model,
        }
    
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t0
        return _error_result("chat", api_model, f"Timeout after {timeout}s", elapsed)
    
    except requests.exceptions.ConnectionError as exc:
        elapsed = time.perf_counter() - t0
        msg = f"Connection failed: {exc}"
        return _error_result("chat", api_model, msg[:200], elapsed)
    
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return _error_result("chat", api_model, str(exc)[:200], elapsed)


# ── Generate: /completion ─────────────────────────────────────────────────────

def query_generate_omlx(
    model_entry: dict,
    question: str,
    timeout: int,
    raw: bool = False,
) -> dict:
    """
    Query oMLX generate endpoint: /completion
    
    Unlike pure OpenAI API, oMLX supports base-weight generation
    via the /completion endpoint (similar to llama.cpp).
    
    Args:
        model_entry: Registry entry with omlx_url, api_model_id fields
        question:    User prompt
        timeout:     Request timeout in seconds
        raw:         If True, attempt raw completion (implementation-dependent)
    
    Returns:
        Result dict with success, content, elapsed_s, endpoint, etc.
    """
    base      = base_url_for(model_entry)
    api_model = api_model_id_for(model_entry)
    url       = f"{base}/completion"
    headers   = headers_for(model_entry)
    
    payload: dict[str, Any] = {
        "model":  api_model,
        "prompt": question,
        "stream": False,
    }
    
    # Note: raw mode handling depends on oMLX implementation
    # Some models may need additional parameters here
    
    t0 = time.perf_counter()
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.perf_counter() - t0
        
        if r.status_code != 200:
            return _error_result(
                "gen_plain",
                api_model,
                f"HTTP {r.status_code}: {r.text[:200]}",
                elapsed,
            )
        
        data = r.json()
        
        # oMLX returns content in "content" key (like llama.cpp)
        content = data.get("content", "")
        
        if not content:
            # Fallback: try "text" key
            content = data.get("text", "")
        
        if not content:
            return _error_result("gen_plain", api_model, "Empty response", elapsed)
        
        # Check for thinking field
        thinking = data.get("thinking", "")
        if thinking:
            return {
                "endpoint":          "gen_plain",
                "success":           True,
                "elapsed_s":         elapsed,
                "raw_response":      content,
                "think_block":       thinking,
                "answer":            content,
                "think_format":      "field",
                "eval_count":        None,
                "prompt_eval_count": None,
                "backend":           BACKEND_NAME,
                "api_model_id":      api_model,
            }
        
        # Check for inline <think> tags
        think_content, clean_answer = _extract_think(content)
        if think_content:
            return {
                "endpoint":          "gen_plain",
                "success":           True,
                "elapsed_s":         elapsed,
                "raw_response":      content,
                "think_block":       think_content,
                "answer":            clean_answer,
                "think_format":      "tag",
                "eval_count":        None,
                "prompt_eval_count": None,
                "backend":           BACKEND_NAME,
                "api_model_id":      api_model,
            }
        
        # Standard response
        return {
            "endpoint":          "gen_plain",
            "success":           True,
            "elapsed_s":         elapsed,
            "raw_response":      content,
            "think_block":       "",
            "answer":            content,
            "think_format":      "none",
            "eval_count":        None,
            "prompt_eval_count": None,
            "backend":           BACKEND_NAME,
            "api_model_id":      api_model,
        }
    
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t0
        return _error_result("gen_plain", api_model, f"Timeout after {timeout}s", elapsed)
    
    except requests.exceptions.ConnectionError as exc:
        elapsed = time.perf_counter() - t0
        msg = f"Connection failed: {exc}"
        return _error_result("gen_plain", api_model, msg[:200], elapsed)
    
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return _error_result("gen_plain", api_model, str(exc)[:200], elapsed)


# ── Health check ───────────────────────────────────────────────────────────────

def check_server(model_entry: dict | None = None) -> tuple[bool, str]:
    """
    Verify oMLX server is reachable and models are available.
    
    Args:
        model_entry: Optional registry entry. If None, uses defaults.
    
    Returns:
        (ok: bool, detail: str)
    """
    if model_entry:
        base = base_url_for(model_entry)
        headers = headers_for(model_entry)
    else:
        base = DEFAULT_BASE_URL
        headers = {"Authorization": f"Bearer {OMLX_API_KEY}"} if OMLX_API_KEY else None
    
    try:
        url = f"{base}/models"
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            if model_ids:
                return True, f"oMLX server reachable, {len(model_ids)} model(s): {', '.join(model_ids[:3])}"
            return True, "oMLX server reachable, no models discovered"
        
        return False, f"HTTP {r.status_code} from {url}"
    
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to oMLX server at {base} — is it running?"
    
    except requests.exceptions.Timeout:
        return False, f"Timeout reaching oMLX server at {base}"
    
    except Exception as exc:
        return False, str(exc)[:120]
