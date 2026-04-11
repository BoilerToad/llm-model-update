#!/usr/bin/env python3
"""
test_ollama_cloud.py
────────────────────
Quick test of the Ollama hosted cloud API using OLLAMA_API_KEY from .env.

Usage:
    python test_ollama_cloud.py                        # test default model
    python test_ollama_cloud.py --model llama3.2       # test specific model
    python test_ollama_cloud.py --endpoint generate    # use /api/generate
    python test_ollama_cloud.py --prompt "What is 2+2?" --model gemma3:4b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from home directory
load_dotenv(Path.home() / ".env")

OLLAMA_CLOUD_BASE = "https://ollama.com"
DEFAULT_MODEL     = "gemma3:12b-cloud"
DEFAULT_PROMPT    = "Respond in one sentence: what is the capital of France?"


def get_api_key() -> str:
    key = os.environ.get("OLLAMA_API_KEY", "")
    if not key:
        print("✗  OLLAMA_API_KEY not found. Add it to your .env file:")
        print("     OLLAMA_API_KEY=your_key_here")
        sys.exit(1)
    return key


def test_chat(model: str, prompt: str, api_key: str, timeout: int = 30) -> dict:
    url = f"{OLLAMA_CLOUD_BASE}/api/chat"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "stream": False,
        "think": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.time()
    try:
        print( " Target URL -> " + str(url))
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        return {"success": True, "elapsed_s": elapsed, "content": content, "raw": data}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP {r.status_code}: {r.text[:200]}",
                "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2)}


def test_generate(model: str, prompt: str, api_key: str, timeout: int = 30) -> dict:
    url = f"{OLLAMA_CLOUD_BASE}/api/generate"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": True,
    }
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data = r.json()
        content = data.get("response", "")
        return {"success": True, "elapsed_s": elapsed, "content": content, "raw": data}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP {r.status_code}: {r.text[:200]}",
                "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2)}


def main():
    parser = argparse.ArgumentParser(
        description="Test Ollama hosted cloud API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",    default=DEFAULT_MODEL,   help=f"Model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt",   default=DEFAULT_PROMPT,  help="Prompt to send")
    parser.add_argument("--endpoint", choices=["chat", "generate", "both"], default="chat",
                        help="API endpoint to test (default: chat)")
    parser.add_argument("--timeout",  type=int, default=30,    help="Request timeout in seconds")
    parser.add_argument("--raw",      action="store_true",     help="Print full raw JSON response")
    args = parser.parse_args()

    api_key = get_api_key()

    print(f"\n{'─'*60}")
    print(f"  Ollama Cloud API Test")
    print(f"  Base  : {OLLAMA_CLOUD_BASE}")
    print(f"  Model : {args.model}")
    print(f"  Prompt: {args.prompt[:60]}")
    print(f"{'─'*60}\n")

    endpoints = ["chat", "generate"] if args.endpoint == "both" else [args.endpoint]

    for ep in endpoints:
        print(f"  Testing /api/{ep} ...", end=" ", flush=True)

        if ep == "chat":
            result = test_chat(args.model, args.prompt, api_key, args.timeout)
        else:
            result = test_generate(args.model, args.prompt, api_key, args.timeout)

        if result["success"]:
            print(f"✓  ({result['elapsed_s']}s)")
            print(f"\n  Response:\n  {result['content'].strip()}\n")
            if args.raw:
                print("  Raw JSON:")
                print(json.dumps(result["raw"], indent=2))
        else:
            print(f"✗")
            print(f"\n  Error: {result['error']}\n")

    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
