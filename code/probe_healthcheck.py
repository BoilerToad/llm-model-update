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
MODELS_FILE   = SCRIPT_DIR / "probes" / "probe_models.json"
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
    except Exception as e:
        return set()


def chat_smoke(model: str, url: str) -> tuple[bool, str]:
    """Send a minimal chat message. Returns (ok, detail)."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
    }
    t0 = time.time()
    try:
        r = requests.post(f"{url}/api/chat", json=payload,
                          headers=auth_headers(url), timeout=CHAT_TIMEOUT)
        elapsed = round(time.time() - t0, 1)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        if not content.strip():
            return False, f"empty response ({elapsed}s)"
        return True, f"{len(content)}c in {elapsed}s"
    except requests.exceptions.Timeout:
        return False, f"timeout after {CHAT_TIMEOUT}s"
    except Exception as e:
        return False, str(e)[:50]


def gen_smoke(model: str) -> tuple[bool, str]:
    """Quick generate smoke test — local only."""
    payload = {
        "model": model,
        "prompt": "Say: OK",
        "stream": False,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_LOCAL}/api/generate", json=payload, timeout=GEN_TIMEOUT)
        elapsed = round(time.time() - t0, 1)
        r.raise_for_status()
        content = r.json().get("response", "")
        if not content.strip():
            return False, f"empty response ({elapsed}s)"
        return True, f"{len(content)}c in {elapsed}s"
    except requests.exceptions.Timeout:
        return False, f"timeout after {GEN_TIMEOUT}s"
    except Exception as e:
        return False, str(e)[:50]


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
    except Exception as e:
        return False, str(e)[:50]


def main():
    parser = argparse.ArgumentParser(description="Probe model health check")
    parser.add_argument("--chat-only", action="store_true",
                        help="Skip generate smoke test (faster)")
    parser.add_argument("--family",    nargs="+", metavar="FAM",
                        help="Filter by family e.g. deepseek gemma")
    parser.add_argument("--models-file", default=str(MODELS_FILE))
    args = parser.parse_args()

    # ── Load registry ─────────────────────────────────────────────────────────
    data     = json.loads(Path(args.models_file).read_text())
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

    print(f"\n{'─'*95}")
    print(f"  Probe Health Check")
    print(f"  Local models : {len(local_models)}")
    print(f"  Cloud models : {len(cloud_models)}")
    print(f"  Chat timeout : {CHAT_TIMEOUT}s   Generate timeout: {GEN_TIMEOUT}s")
    print(f"{'─'*95}\n")

    # ── Check local Ollama is up ───────────────────────────────────────────────
    print("Checking Ollama local instance...")
    local_tags = get_local_tags()
    if not local_tags:
        print("  ✗ Cannot reach http://localhost:11434/api/tags — is Ollama running?")
        sys.exit(1)
    print(f"  ✓ Ollama running — {len(local_tags)} models registered\n")

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
        family = m.get("family", "?")

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
        chat_ok, chat_detail = chat_smoke(name, OLLAMA_LOCAL)
        chat_sym = "✓" if chat_ok else "✗"

        # 3. Generate smoke (optional)
        if args.chat_only:
            gen_sym = "─"
            gen_detail = "skipped"
            gen_ok = True
        else:
            gen_ok, gen_detail = gen_smoke(name)
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

                # Chat smoke via cloud
                chat_ok, chat_detail = chat_smoke(name, OLLAMA_CLOUD)
                chat_sym = "✓" if chat_ok else "✗"
                gen_sym  = "─"   # generate not smoke-tested for cloud (slow/costly)

                overall = "PASS" if chat_ok else "FAIL"
                print(f"  {name[:W_MODEL]:<{W_MODEL}} {'cloud':<{W_TYPE}} "
                      f"{tags_sym:<{W_STATUS}} {chat_sym:<{W_STATUS}} {gen_sym:<{W_STATUS}} {chat_detail}")
                results.append((name, overall, chat_detail))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")

    print(f"\n{'─'*95}")
    print(f"  Results: {passed} PASS  {failed} FAIL  {skipped} SKIP  "
          f"({len(results)} models checked)")

    if failed:
        print(f"\n  Failed models:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"    ✗ {name}  —  {detail}")

    print(f"{'─'*95}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
