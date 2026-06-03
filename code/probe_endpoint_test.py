#!/usr/bin/env python3
"""
probe_endpoint_test.py
──────────────────────
Tests /api/chat and /api/generate endpoints for a given model with accurate
wall-clock timing. Designed to surface endpoint-level issues like:
  - generate timing out while chat completes
  - raw:true being silently ignored or unsupported (MLX backend)
  - asymmetric response behaviour between endpoints

Usage:
    python probe_endpoint_test.py --model gemma4:26b
    python probe_endpoint_test.py --model qwen3.5:35b-a3b-coding-nvfp4
    python probe_endpoint_test.py --model deepseek-r1:7b --timeout 60
    python probe_endpoint_test.py --all-enabled
    python probe_endpoint_test.py --model gemma4:26b --no-generate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_LOCAL   = "http://localhost:11434"
SCRIPT_DIR     = Path(__file__).parent
ROOT_DIR       = SCRIPT_DIR.parent
MODELS_FILE    = ROOT_DIR / "probes" / "probe_models.json"
TEST_QUESTION  = "What is democracy? Answer in two sentences."
TEST_PROMPT    = f"Question: {TEST_QUESTION}\n\nAnswer:"

# Import omlx backend functions
sys.path.insert(0, str(SCRIPT_DIR))
from probe_query_omlx import (
    is_omlx_backend,
    query_chat_omlx,
    query_generate_omlx,
    check_server as check_omlx_server,
)


# ── Result dataclass ───────────────────────────────────────────────────────────

class EndpointResult:
    def __init__(self, endpoint: str):
        self.endpoint    = endpoint
        self.success     = False
        self.elapsed_s   = 0.0
        self.status_code = None
        self.response_len = 0
        self.eval_count  = None
        self.done_reason = None
        self.error       = None
        self.snippet     = ""

    def __str__(self):
        if self.error:
            return (f"  /{self.endpoint:<10} FAIL  {self.elapsed_s:>7.2f}s  "
                    f"error={self.error}")
        return (f"  /{self.endpoint:<10} OK    {self.elapsed_s:>7.2f}s  "
                f"len={self.response_len:>6}c  "
                f"eval={str(self.eval_count):>5}  "
                f"done={self.done_reason}")


# ── Endpoint testers ───────────────────────────────────────────────────────────

def test_chat(model: str, timeout: int) -> EndpointResult:
    result = EndpointResult("api/chat")
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": TEST_QUESTION}],
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{OLLAMA_LOCAL}/api/chat",
            json=payload,
            timeout=timeout,
        )
        result.elapsed_s   = round(time.perf_counter() - t0, 3)
        result.status_code = r.status_code

        if r.status_code != 200:
            result.error = f"HTTP {r.status_code}: {r.text[:120]}"
            return result

        data = r.json()
        content = data.get("message", {}).get("content", "")
        thinking = data.get("message", {}).get("thinking", "")

        result.success      = True
        result.response_len = len(content)
        result.eval_count   = data.get("eval_count")
        result.done_reason  = data.get("done_reason")
        result.snippet      = content[:120].replace("\n", " ")

        # Note if think field was present
        if thinking:
            result.done_reason = f"{result.done_reason} [think:{len(thinking)}c]"

    except requests.exceptions.Timeout:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = f"TIMEOUT after {timeout}s"
    except Exception as e:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = str(e)[:120]

    return result


def test_generate(model: str, timeout: int, raw: bool = True) -> EndpointResult:
    label   = f"api/generate" + (" raw=T" if raw else " raw=F")
    result  = EndpointResult(label)
    payload = {
        "model": model,
        "prompt": TEST_PROMPT,
        "stream": False,
    }
    if raw:
        payload["raw"] = True

    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{OLLAMA_LOCAL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        result.elapsed_s   = round(time.perf_counter() - t0, 3)
        result.status_code = r.status_code

        if r.status_code != 200:
            result.error = f"HTTP {r.status_code}: {r.text[:120]}"
            return result

        data = r.json()
        content = data.get("response", "")

        result.success      = True
        result.response_len = len(content)
        result.eval_count   = data.get("eval_count")
        result.done_reason  = data.get("done_reason")
        result.snippet      = content[:120].replace("\n", " ")

    except requests.exceptions.Timeout:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = f"TIMEOUT after {timeout}s"
    except Exception as e:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = str(e)[:120]

    return result


# ── oMLX endpoint testers ──────────────────────────────────────────────────────

def test_chat_omlx(model_entry: dict, timeout: int) -> EndpointResult:
    """Test oMLX chat endpoint using probe_query_omlx."""
    result = EndpointResult("omlx/chat")
    t0 = time.perf_counter()
    try:
        response = query_chat_omlx(model_entry, TEST_QUESTION, timeout)
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        
        if "error" in response:
            result.error = response["error"][:120]
            return result
        
        result.success = True
        result.response_len = len(response.get("answer", ""))
        result.eval_count = response.get("eval_count")
        result.snippet = response.get("answer", "")[:120].replace("\n", " ")
        
        # Note if think block was present
        if response.get("think_block"):
            result.done_reason = f"complete [think:{len(response['think_block'])}c]"
        else:
            result.done_reason = "complete"
            
    except Exception as e:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = str(e)[:120]
    
    return result


def test_generate_omlx(model_entry: dict, timeout: int) -> EndpointResult:
    """Test oMLX generate endpoint using probe_query_omlx."""
    result = EndpointResult("omlx/generate")
    t0 = time.perf_counter()
    try:
        response = query_generate_omlx(model_entry, TEST_PROMPT, timeout)
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        
        if "error" in response:
            result.error = response["error"][:120]
            return result
        
        result.success = True
        result.response_len = len(response.get("answer", ""))
        result.eval_count = response.get("eval_count")
        result.snippet = response.get("answer", "")[:120].replace("\n", " ")
        result.done_reason = "complete"
        
    except Exception as e:
        result.elapsed_s = round(time.perf_counter() - t0, 3)
        result.error = str(e)[:120]
    
    return result


# ── Per-model test ─────────────────────────────────────────────────────────────

def test_model(
    model: str,
    timeout: int,
    no_generate: bool = False,
    model_entry: dict | None = None,
) -> dict:
    """Run all endpoint tests for a single model. Returns result summary dict."""
    print(f"\n  Model : {model}")
    print(f"  {'─'*60}")

    results = {}
    
    # Check if this is an omlx backend model
    is_omlx = model_entry and is_omlx_backend(model_entry)
    
    if is_omlx:
        # oMLX backend tests
        print(f"  Testing oMLX chat endpoint ...", end=" ", flush=True)
        chat = test_chat_omlx(model_entry, timeout)
        results["chat"] = chat
        print(f"{chat.elapsed_s:.2f}s  {'✓' if chat.success else '✗'}")
        
        if not no_generate:
            print(f"  Testing oMLX generate endpoint ...", end=" ", flush=True)
            gen = test_generate_omlx(model_entry, timeout)
            results["generate_plain"] = gen
            print(f"{gen.elapsed_s:.2f}s  {'✓' if gen.success else '✗'}")
    else:
        # Ollama backend tests
        # Chat
        print(f"  Testing /api/chat ...", end=" ", flush=True)
        chat = test_chat(model, timeout)
        results["chat"] = chat
        print(f"{chat.elapsed_s:.2f}s  {'✓' if chat.success else '✗'}")

        if not no_generate:
            # Generate raw:true
            print(f"  Testing /api/generate (raw:true) ...", end=" ", flush=True)
            gen_raw = test_generate(model, timeout, raw=True)
            results["generate_raw"] = gen_raw
            print(f"{gen_raw.elapsed_s:.2f}s  {'✓' if gen_raw.success else '✗'}")

            # Generate raw:false — baseline comparison
            print(f"  Testing /api/generate (raw:false) ...", end=" ", flush=True)
            gen_plain = test_generate(model, timeout, raw=False)
            results["generate_plain"] = gen_plain
            print(f"{gen_plain.elapsed_s:.2f}s  {'✓' if gen_plain.success else '✗'}")

    # Print detailed results
    print()
    for key, res in results.items():
        print(str(res))

    # Print response snippets for successful calls
    print()
    for key, res in results.items():
        if res.success and res.snippet:
            print(f"  [{key}] snippet: {res.snippet[:100]}")

# Diagnosis
    print()
    if not no_generate:
        chat_ok      = results["chat"].success
        gen_raw_ok   = results.get("generate_raw", EndpointResult("none")).success
        gen_plain_ok = results["generate_plain"].success

        if not chat_ok:
            print(f"  ✗ DIAGNOSIS: Chat endpoint failed — check model availability.")

        elif chat_ok and not gen_raw_ok and not gen_plain_ok:
            print(f"  ⚠ DIAGNOSIS: Chat works, both generate modes fail.")
            print(f"    Likely: MLX backend / safetensors — generate endpoint unsupported.")

        elif chat_ok and not gen_raw_ok and gen_plain_ok:
            print(f"  ⚠ DIAGNOSIS: raw:true hangs, raw:false works.")
            print(f"    Confirmed: MLX/safetensors model — use raw:false for generate.")
            print(f"    probe_static.py v2.6 handles this automatically via is_mlx_model().")

        elif chat_ok and gen_raw_ok and gen_plain_ok:
            chat_t     = results["chat"].elapsed_s
            gen_raw_t  = results["generate_raw"].elapsed_s
            gen_raw_ev = results["generate_raw"].eval_count or 0
            gen_pln_ev = results["generate_plain"].eval_count or 0

            issues = []

            # Suspiciously fast with very few tokens — stopped prematurely
            if gen_raw_t < 2.0 and gen_raw_ev < 60:
                issues.append(
                    f"raw:true completed in {gen_raw_t:.2f}s with only {gen_raw_ev} tokens — "
                    f"likely stopped prematurely (stop token or few-shot artifact)"
                )

            # raw:false generated dramatically more tokens than raw:true
            if gen_pln_ev > 0 and gen_raw_ev > 0:
                ratio = gen_pln_ev / gen_raw_ev
                if ratio > 5:
                    issues.append(
                        f"raw:false generated {gen_pln_ev} tokens vs raw:true {gen_raw_ev} "
                        f"({ratio:.0f}x more) — raw:true output is likely shallow/spurious"
                    )

            # Spurious Q&A continuation in raw:true snippet
            snippet = results["generate_raw"].snippet.lower()
            if any(kw in snippet for kw in ("question:", "answer:", "q:", "a:")):
                issues.append(
                    "raw:true snippet contains 'Question:/Answer:' markers — "
                    "model is continuing the few-shot prompt, not answering"
                )

            if issues:
                print(f"  ⚠ DIAGNOSIS: raw:true technically succeeds but output is suspect:")
                for issue in issues:
                    print(f"    • {issue}")
                print(f"    Recommendation: use raw:false (MLX safetensors).")
                print(f"    probe_static.py v2.6 handles this automatically via is_mlx_model().")
            else:
                speed_ratio = gen_raw_t / max(chat_t, 0.01)
                print(f"  ✓ Both endpoints working correctly.")
                print(f"    Generate/chat time ratio: {speed_ratio:.1f}x  "
                      f"(raw:{gen_raw_ev} tok  plain:{gen_pln_ev} tok)")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test Ollama /api/chat and /api/generate with accurate timing"
    )
    parser.add_argument("--model",       metavar="NAME",
                        help="Single model to test")
    parser.add_argument("--all-enabled", action="store_true",
                        help="Test all enabled models from probe_models.json")
    parser.add_argument("--timeout",     type=int, default=120,
                        help="Timeout per request in seconds (default: 120)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip generate endpoint tests")
    parser.add_argument("--models-file", default=str(MODELS_FILE),
                        help="Path to probe_models.json (default: probes/probe_models.json)")
    args = parser.parse_args()

    models_file = Path(args.models_file)

    if not args.model and not args.all_enabled:
        parser.error("Specify --model MODEL or --all-enabled")

    # Check Ollama is up
    try:
        r = requests.get(f"{OLLAMA_LOCAL}/api/version", timeout=5)
        version = r.json().get("version", "?")
        print(f"\n  Ollama v{version} at {OLLAMA_LOCAL}")
    except Exception as e:
        print(f"✗ Cannot reach Ollama: {e}")
        sys.exit(1)

    print(f"  Timeout   : {args.timeout}s per request")
    print(f"  Question  : {TEST_QUESTION}")
    print(f"  Started   : {datetime.now():%Y-%m-%d %H:%M:%S}")

    # Load registry data
    data = json.loads(models_file.read_text())
    registry = {m["name"]: m for m in data["models"]}
    
    # Build model list
    models = []
    if args.model:
        models = [args.model]
    else:
        models = [
            m["name"] for m in data["models"]
            if m.get("enabled", True)
            and "cloud" not in m["name"].lower()
        ]

    print(f"\n{'═'*65}")
    all_results = {}

    for model in models:
        model_entry = registry.get(model)
        
        # Check if this is an omlx backend model
        if model_entry and is_omlx_backend(model_entry):
            # oMLX backend — check server connectivity
            ok, detail = check_omlx_server(model_entry)
            if not ok:
                print(f"\n  Model : {model}")
                print(f"  ✗ oMLX server not available — {detail}")
                continue
        else:
            # Ollama backend — quick availability check
            try:
                tags = requests.get(f"{OLLAMA_LOCAL}/api/tags", timeout=5).json()
                available = [m["name"] for m in tags.get("models", [])]
                if model not in available:
                    print(f"\n  Model : {model}")
                    print(f"  ✗ Not available in Ollama — skipping")
                    continue
            except Exception:
                pass

        all_results[model] = test_model(model, args.timeout, args.no_generate, model_entry)

    # Summary table
    print(f"\n{'═'*65}")
    print(f"  SUMMARY\n")
    print(f"  {'Model':<45} {'Chat':>6} {'Gen/raw':>8} {'Gen/plain':>10}")
    print(f"  {'─'*45} {'─'*6} {'─'*8} {'─'*10}")

    for model, results in all_results.items():
        chat_s = (f"{results['chat'].elapsed_s:.1f}s"
                  if results['chat'].success
                  else f"FAIL")
        gen_r  = ("—" if args.no_generate or "generate_raw" not in results else
                  f"{results['generate_raw'].elapsed_s:.1f}s"
                  if results['generate_raw'].success
                  else "FAIL/TO")
        gen_p  = ("—" if args.no_generate else
                  f"{results['generate_plain'].elapsed_s:.1f}s"
                  if results['generate_plain'].success
                  else "FAIL/TO")
        print(f"  {model:<45} {chat_s:>6} {gen_r:>8} {gen_p:>10}")

    print()


if __name__ == "__main__":
    main()
