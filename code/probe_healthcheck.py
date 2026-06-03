"""
probe_healthcheck.py
─────────────────────────────────────────────────────────────────────────────
Pre-run health check. Verifies every enabled model in probe_models.json is:
  1. Registered in local Ollama /api/tags
  2. Responds to a minimal chat prompt (local) or tags check (cloud)
  3. Generates a non-empty response (local models only — quick smoke test)

Does NOT run full probe questions — this is fast, typically < 2 mins.

Usage:
    python probe_healthcheck.py              # check all enabled models
    python probe_healthcheck.py --chat-only  # skip generate smoke test
    python probe_healthcheck.py --local      # check local models only
    python probe_healthcheck.py --cloud      # check cloud models only
    python probe_healthcheck.py --chat-timeout 60 --gen-timeout 120  # custom timeouts
    python probe_healthcheck.py --family deepseek gemma  # filter by family

Output: table showing PASS / FAIL / SKIP per model with reason.
Exit code: 0 if all pass, 1 if any fail.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

SCRIPT_DIR    = Path(__file__).parent
PROJECT_ROOT  = SCRIPT_DIR.parent
MODELS_FILE   = PROJECT_ROOT / "probes" / "probe_models.json"
OLLAMA_LOCAL  = "http://localhost:11434"
OLLAMA_CLOUD  = "https://ollama.com"
CHAT_TIMEOUT  = 30
GEN_TIMEOUT   = 60
SMOKE_PROMPT  = "Reply with exactly one word: Hello."

SKIP_FAMILIES = {"bert", "clip"}
SKIP_NAMES    = {"mxbai-embed-large:latest"}

# Columns
W_MODEL  = 42
W_TYPE   = 7
W_STATUS = 8
W_DETAIL = 36


def auth_headers(url: str) -> dict:
    if "ollama.com" in url:
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


def get_local_tags() -> set[str]:
    try:
        r = requests.get(f"{OLLAMA_LOCAL}/api/tags", timeout=10)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:  # pylint: disable=broad-exception-caught
        return set()


def chat_smoke(model: str, url: str, timeout: int = CHAT_TIMEOUT) -> tuple[bool, str]:
    """Send a minimal chat message. Returns (ok, detail)."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
    }
    t0 = time.time()
    try:
        r = requests.post(f"{url}/api/chat", json=payload,
                          headers=auth_headers(url), timeout=timeout)
        elapsed = round(time.time() - t0, 1)
        if not r.ok:
            try:
                detail = r.json().get("error", r.text)
            except Exception:  # pylint: disable=broad-exception-caught
                detail = r.text or r.reason
            return False, f"HTTP {r.status_code}: {detail}"
        msg     = r.json().get("message", {})
        content = msg.get("content", "") or msg.get("thinking", "")
        if not content.strip():
            return False, f"empty response ({elapsed}s)"
        return True, f"{len(content)}c in {elapsed}s"
    except requests.exceptions.Timeout:
        return False, f"timeout after {timeout}s"
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, str(e)


def gen_smoke(model: str, url: str = OLLAMA_LOCAL,
              timeout: int = GEN_TIMEOUT) -> tuple[bool, str]:
    """Quick generate smoke test."""
    payload = {
        "model": model,
        "prompt": "Say: OK",
        "stream": False,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{url}/api/generate", json=payload,
                          headers=auth_headers(url), timeout=timeout)
        elapsed = round(time.time() - t0, 1)
        if not r.ok:
            try:
                detail = r.json().get("error", r.text)
            except Exception:  # pylint: disable=broad-exception-caught
                detail = r.text or r.reason
            return False, f"HTTP {r.status_code}: {detail}"
        content = r.json().get("response", "")
        # Handle non-string responses (some models return integers or other types)
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        if not content.strip():
            return False, f"empty response ({elapsed}s)"
        return True, f"{len(content)}c in {elapsed}s"
    except requests.exceptions.Timeout:
        return False, f"timeout after {timeout}s"
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, str(e)


def check_cloud_key() -> tuple[bool, str]:
    key = os.environ.get("OLLAMA_API_KEY", "")
    if not key:
        return False, "OLLAMA_API_KEY not set in ~/.env"
    # Probe cloud tags
    try:
        r = requests.get(f"{OLLAMA_LOCAL}/api/tags", timeout=10)
        r.raise_for_status()
        cloud_models = [m["name"] for m in r.json().get("models", [])
                        if "cloud" in m["name"].lower()]
        return True, f"key present, {len(cloud_models)} cloud proxies registered"
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Probe model health check")
    parser.add_argument("--chat-only", action="store_true",
                        help="Skip generate smoke test (faster)")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--local", action="store_true",
                       help="Check local models only")
    scope.add_argument("--cloud", action="store_true",
                       help="Check cloud models only")
    parser.add_argument("--chat-timeout", type=int, default=CHAT_TIMEOUT, metavar="SEC",
                        help=f"Chat smoke test timeout in seconds (default: {CHAT_TIMEOUT})")
    parser.add_argument("--gen-timeout",  type=int, default=GEN_TIMEOUT,  metavar="SEC",
                        help=f"Generate smoke test timeout in seconds (default: {GEN_TIMEOUT})")
    parser.add_argument("--family",    nargs="+", metavar="FAM",
                        help="Filter by family e.g. deepseek gemma")
    parser.add_argument("--models-file", default=str(MODELS_FILE))
    args = parser.parse_args()

    chat_timeout = args.chat_timeout
    gen_timeout  = args.gen_timeout

    # ── Load registry ─────────────────────────────────────────────────────────
    data     = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
    registry = [
        m for m in data["models"]
        if m.get("enabled", True)
        and m.get("family", "") not in SKIP_FAMILIES
        and m["name"] not in SKIP_NAMES
    ]
    if args.family:
        registry = [m for m in registry if m.get("family","") in args.family]

    local_models = [m for m in registry if "cloud" not in m["name"].lower()]
    cloud_models = [m for m in registry if "cloud"     in m["name"].lower()]

    if args.local:
        cloud_models = []
    elif args.cloud:
        local_models = []

    print(f"\n{'─'*95}")
    print("  Probe Health Check")
    print(f"  Local models : {len(local_models)}")
    print(f"  Cloud models : {len(cloud_models)}")
    print(f"  Chat timeout : {chat_timeout}s   Generate timeout: {gen_timeout}s")
    print(f"{'─'*95}\n")

    # ── Check local Ollama is up ───────────────────────────────────────────────
    print("Checking Ollama local instance...")
    local_tags = get_local_tags()
    if not local_tags and not args.cloud:
        print("  ✗ Cannot reach http://localhost:11434/api/tags — is Ollama running?")
        sys.exit(1)
    if local_tags:
        print(f"  ✓ Ollama running — {len(local_tags)} models registered\n")
    else:
        print("  ─ Skipped (cloud-only mode)\n")

    # ── Check cloud API key ────────────────────────────────────────────────────
    if cloud_models:
        print("Checking Ollama cloud API key...")
        key_ok, key_detail = check_cloud_key()
        symbol = "✓" if key_ok else "✗"
        print(f"  {symbol} {key_detail}\n")
    else:
        key_ok = True

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"  {'Model':<{W_MODEL}} {'Type':<{W_TYPE}} {'Tags':<{W_STATUS}} "
          f"{'Chat':<{W_STATUS}} {'Gen':<{W_STATUS}} Detail")
    print("  " + "─" * 92)

    results = []

    # ── Local models ──────────────────────────────────────────────────────────
    for m in local_models:
        name   = m["name"]
        _family = m.get("family", "?")

        # 1. In tags?
        in_tags = name in local_tags
        tags_sym = "✓" if in_tags else "✗"

        if not in_tags:
            print(f"  {name[:W_MODEL]:<{W_MODEL}} {'local':<{W_TYPE}} "
                  f"{tags_sym:<{W_STATUS}} {'─':<{W_STATUS}} {'─':<{W_STATUS}} "
                  f"NOT in /api/tags — ollama pull needed?")
            results.append((name, "FAIL", "not in tags"))
            continue

        # 2. Chat smoke
        chat_ok, chat_detail = chat_smoke(name, OLLAMA_LOCAL, chat_timeout)
        chat_sym = "✓" if chat_ok else "✗"

        # 3. Generate smoke (optional)
        if args.chat_only:
            gen_sym = "─"
            gen_detail = "skipped"
            gen_ok = True
        else:
            gen_ok, gen_detail = gen_smoke(name, gen_timeout)
            gen_sym = "✓" if gen_ok else "✗"

        overall = "PASS" if (chat_ok and gen_ok) else "FAIL"
        detail  = chat_detail if not chat_ok else (gen_detail if not gen_ok else chat_detail)

        print(f"  {name[:W_MODEL]:<{W_MODEL}} {'local':<{W_TYPE}} "
              f"{tags_sym:<{W_STATUS}} {chat_sym:<{W_STATUS}} {gen_sym:<{W_STATUS}} {detail}")
        results.append((name, overall, detail))

    # ── Cloud models ──────────────────────────────────────────────────────────
    if cloud_models:
        print()
        if not key_ok:
            for m in cloud_models:
                print(f"  {m['name'][:W_MODEL]:<{W_MODEL}} {'cloud':<{W_TYPE}} "
                      f"{'─':<{W_STATUS}} {'─':<{W_STATUS}} {'─':<{W_STATUS}} SKIP — no API key")
                results.append((m["name"], "SKIP", "no API key"))
        else:
            for m in cloud_models:
                name = m["name"]

                # Cloud models are registered locally as proxy entries
                in_tags = name in local_tags
                tags_sym = "✓" if in_tags else "✗"

                if not in_tags:
                    print(f"  {name[:W_MODEL]:<{W_MODEL}} {'cloud':<{W_TYPE}} "
                          f"{tags_sym:<{W_STATUS}} {'─':<{W_STATUS}} {'─':<{W_STATUS}} "
                          f"NOT in local tags — ollama pull needed?")
                    results.append((name, "FAIL", "not in local tags"))
                    continue

                # Chat + generate smoke via cloud
                chat_ok, chat_detail = chat_smoke(name, OLLAMA_CLOUD, chat_timeout)
                chat_sym = "✓" if chat_ok else "✗"

                gen_ok, gen_detail = gen_smoke(name, OLLAMA_CLOUD, gen_timeout)
                gen_sym = "✓" if gen_ok else "✗"

                overall = "PASS" if (chat_ok and gen_ok) else "FAIL"
                detail  = chat_detail if not chat_ok else (gen_detail if not gen_ok else chat_detail)
                print(f"  {name[:W_MODEL]:<{W_MODEL}} {'cloud':<{W_TYPE}} "
                      f"{tags_sym:<{W_STATUS}} {chat_sym:<{W_STATUS}} {gen_sym:<{W_STATUS}} {detail}")
                results.append((name, overall, detail))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")

    print(f"\n{'─'*95}")
    print(f"  Results: {passed} PASS  {failed} FAIL  {skipped} SKIP  "
          f"({len(results)} models checked)")

    if failed:
        print("\n  Failed models:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"    ✗ {name}  —  {detail}")

    print(f"{'─'*95}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
