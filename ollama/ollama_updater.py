#!/usr/bin/env python3
"""
ollama/ollama_updater.py  v3.0
────────────────────────────────────────────────────────────────────────────
Selectively pull/check Ollama models.

Single source of truth: probes/probe_models.json
  - Reads models where backend == "ollama"
  - Uses the "update" field (true / false / check) as pull policy
  - Runtime config (delays, log path) read from ollama/settings.yaml

Usage
─────
  python ollama_updater.py              # auto — respect each model's update field
  python ollama_updater.py --mode all  # pull every ollama model regardless
  python ollama_updater.py --mode check # dry-run for all
  python ollama_updater.py --model deepseek-r1:7b  # single model
  python ollama_updater.py --list      # show registry vs installed status
  python ollama_updater.py --sync      # add installed models missing from probe_models.json
  python ollama_updater.py --prune     # disable probe entries no longer installed

Requirements
────────────
  pip install pyyaml
  Ollama must be running: ollama serve
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT          = Path(__file__).parent.parent
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"
PROBE_FILE    = ROOT / "probes" / "probe_models.json"
OLLAMA_BIN    = "ollama"
OLLAMA_API    = "http://localhost:11434/api"

sys.path.insert(0, str(ROOT))
from shared.log_utils import setup_logging, C
from shared.config_utils import load_yaml_config

# Models registered in Ollama for operational reasons but excluded from probes
NON_PROBE_NAMES = {
    "llama3.2-vision:11b",
    "mxbai-embed-large:latest",
    "llava:13b",
}
SKIP_FAMILIES = {"bert", "clip", "mllama"}
SKIP_NAME_SUBSTRINGS = {"embed", "vision", "llava"}

FAMILY_ORIGINS = {
    "llama":   "US/Meta",    "gemma":  "US/Google",  "gemma2": "US/Google",
    "gemma3":  "US/Google",  "gemini": "US/Google",  "openai": "US/OpenAI",
    "gptoss":  "US/OpenAI",  "nvidia": "US/NVIDIA",  "nemotron_h_moe": "US/NVIDIA",
    "deepseek":"China/DeepSeek", "qwen2":"China/Alibaba", "qwen35":"China/Alibaba",
    "qwen":    "China/Alibaba", "glm":  "China/ZHIPU", "glm4moelite":"China/ZHIPU",
    "mistral": "France/Mistral", "mistral3":"France/Mistral",
    "falcon":  "UAE/TII",    "allam": "Saudi/SDAIA",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_API}/tags", timeout=3)
        return True
    except Exception:
        return False


def get_installed_models() -> dict[str, str]:
    """Return {model_name: digest_prefix}."""
    try:
        r = subprocess.run([OLLAMA_BIN, "list", "--json"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout)
            models = data.get("models", data) if isinstance(data, dict) else data
            return {m["name"]: m.get("digest", "")[:12] for m in models}
    except Exception:
        pass
    r = subprocess.run([OLLAMA_BIN, "list"], capture_output=True, text=True, timeout=15)
    out = {}
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            out[parts[0]] = parts[1][:12] if len(parts) > 1 else ""
    return out


def get_ollama_tags() -> list[dict]:
    import urllib.request
    resp = urllib.request.urlopen(f"{OLLAMA_API}/tags", timeout=5)
    return json.loads(resp.read()).get("models", [])


def load_probe_models() -> list[dict]:
    """Return all ollama-backend models from probe_models.json."""
    data = json.loads(PROBE_FILE.read_text(encoding="utf-8"))
    return [m for m in data["models"] if m.get("backend") == "ollama"]


def load_probe_data() -> dict:
    return json.loads(PROBE_FILE.read_text(encoding="utf-8"))


def save_probe_data(data: dict):
    PROBE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        return load_yaml_config(SETTINGS_FILE)
    # Fallback defaults if settings.yaml not yet created
    return {"pull_delay_seconds": 2,
            "log_file": "~/Library/Logs/llm-model-update/ollama.log",
            "log_retention_days": 30}


# ── Core operations ───────────────────────────────────────────────────────────

def pull_model(name: str, dry_run: bool, delay: float, logger) -> str:
    if dry_run:
        logger.info(f"  {C['cyan']}~ DRY-RUN{C['reset']}   would pull: {name}")
        return "dry_run"
    logger.info(f"  {C['bold']}↓ Pulling{C['reset']}    {name} …")
    try:
        r = subprocess.run([OLLAMA_BIN, "pull", name],
                           capture_output=True, text=True, timeout=600)
        time.sleep(delay)
        if r.returncode != 0:
            logger.error(f"  {C['red']}✗ FAILED{C['reset']}     {name}\n    {r.stderr.strip()}")
            return "failed"
        combined = (r.stdout + r.stderr).lower()
        if "already up to date" in combined or "up to date" in combined:
            logger.info(f"  {C['green']}✓ CURRENT{C['reset']}    {name}")
            return "current"
        logger.info(f"  {C['green']}✓ UPDATED{C['reset']}    {name}")
        return "updated"
    except subprocess.TimeoutExpired:
        logger.error(f"  {C['red']}✗ TIMEOUT{C['reset']}    {name} (>600s)")
        return "failed"


def run_update(mode: str, single: Optional[str], logger, settings: dict):
    delay    = float(settings.get("pull_delay_seconds", 2))
    dry_run  = (mode == "check")
    models   = load_probe_models()
    installed = get_installed_models()

    if single:
        work = [m for m in models if m["name"] == single]
        if not work:
            logger.warning(f"'{single}' not found in probe_models.json with backend=ollama")
            return
    elif mode == "all":
        work = [m for m in models if m["name"] in installed]
    else:
        work = [m for m in models
                if m.get("update", "check") is not False
                and m["name"] in installed]

    logger.info(f"\n{C['bold']}Ollama Updater [{mode.upper()}] {datetime.now():%Y-%m-%d %H:%M}{C['reset']}")
    logger.info(f"  Models: {len(work)}  Dry-run: {dry_run}\n")

    counts = {"updated": 0, "current": 0, "failed": 0, "skipped": 0, "dry_run": 0}

    for m in work:
        name   = m["name"]
        update = m.get("update", "check")
        if update is False and mode != "all":
            logger.info(f"  {C['yellow']}⊘ SKIP{C['reset']}       {name}  (pinned)")
            counts["skipped"] += 1
            continue
        effective_dry = dry_run or (str(update) == "check" and mode == "auto")
        status = pull_model(name, dry_run=effective_dry, delay=delay, logger=logger)
        counts[status] = counts.get(status, 0) + 1

    not_installed = [m["name"] for m in models if m["name"] not in installed]
    if not_installed:
        logger.info(f"\n{C['yellow']}  ⚠ In registry but not installed:{C['reset']}")
        for n in not_installed:
            logger.info(f"    • {n}")

    logger.info(f"\n{C['bold']}Summary:{C['reset']} Updated={counts['updated']} "
                f"Current={counts['current']} Skipped={counts['skipped']} "
                f"Failed={counts['failed']}"
                + (f" DryRun={counts['dry_run']}" if counts["dry_run"] else ""))


def list_models(logger):
    models    = load_probe_models()
    installed = get_installed_models()
    registry  = {m["name"]: m for m in models}

    logger.info(f"\n{'MODEL':<45} {'UPDATE':<8} {'INSTALLED':<12}  ORIGIN")
    logger.info("─" * 90)
    for name, digest in sorted(installed.items()):
        if name not in registry:
            logger.info(f"{name:<45} {'—':<8} {'yes':>12}  (not in probe_models.json)")
            continue
        m = registry[name]
        update = str(m.get("update", "check"))
        origin = m.get("geopolitical_origin", "—")
        logger.info(f"{name:<45} {update:<8} {digest:>12}  {origin}")

    untracked = [n for n in installed if n not in registry]
    if untracked:
        logger.info(f"\n{C['yellow']}  ⚠ Installed but not in probe_models.json:{C['reset']}")
        for n in untracked:
            logger.info(f"    • {n}")


def sync_models(logger):
    """Add installed models missing from probe_models.json."""
    installed = get_installed_models()
    data      = load_probe_data()
    probe_names = {m["name"] for m in data["models"]}

    try:
        tags_by_name = {t["name"]: t for t in get_ollama_tags()}
    except Exception:
        tags_by_name = {}

    to_add = []
    for name in sorted(installed):
        if name in probe_names or name in NON_PROBE_NAMES:
            continue
        tag = tags_by_name.get(name, {})
        family_raw = (tag.get("details", {}).get("family") or "").lower()
        families   = [f.lower() for f in (tag.get("details", {}).get("families") or [])]
        if family_raw in SKIP_FAMILIES or any(f in SKIP_FAMILIES for f in families):
            continue
        if any(s in name.lower() for s in SKIP_NAME_SUBSTRINGS):
            continue
        to_add.append((name, tag))

    if not to_add:
        logger.info(f"{C['green']}✓ probe_models.json already up to date{C['reset']}")
        return

    insert_at = len(data["models"])
    for i, m in enumerate(data["models"]):
        if not m.get("enabled", True):
            insert_at = i
            break

    added = []
    for name, tag in to_add:
        details  = tag.get("details", {})
        family   = (details.get("family") or "unknown").lower()
        size_gb  = round(tag.get("size", 0) / 1e9, 1)
        geo      = FAMILY_ORIGINS.get(family)
        if not geo:
            for key, origin in FAMILY_ORIGINS.items():
                if key in name.lower():
                    geo = origin
                    break
        entry = {
            "name": name, "backend": "ollama", "size_gb": size_gb,
            "family": family, "geopolitical_origin": geo,
            "tool_capable": None, "think_blocks": None, "chat_alignment_strong": None,
            "update": "check", "enabled": True,
            "notes": (f"Added by --sync {datetime.now():%Y-%m-%d}. "
                      "Fill in: geopolitical_origin, tool_capable, think_blocks, "
                      "chat_alignment_strong."),
        }
        data["models"].insert(insert_at, entry)
        insert_at += 1
        added.append(name)

    old_ver = data["_meta"]["version"]
    parts   = old_ver.split(".")
    new_ver = f"{parts[0]}.{int(parts[1]) + 1}"
    data["_meta"]["version"] = new_ver
    data["_meta"]["changelog"][new_ver] = (
        f"--sync {datetime.now():%Y-%m-%d}: added {', '.join(added)}"
    )
    save_probe_data(data)

    logger.info(f"{C['bold']}Synced {len(added)} model(s) into probe_models.json:{C['reset']}")
    for name in added:
        logger.info(f"  {C['cyan']}+{C['reset']} {name}")
    logger.info(f"\n  {C['yellow']}⚠ Fill in null research fields for added entries{C['reset']}")


def prune_models(logger):
    """Set enabled=false for probe entries no longer installed. Never deletes."""
    installed = get_installed_models()
    data      = load_probe_data()
    pruned    = []

    for m in data["models"]:
        if (m.get("backend") == "ollama"
                and m.get("enabled", True)
                and not m.get("cloud", False)
                and "cloud" not in m["name"].lower()
                and m["name"] not in installed):
            m["enabled"] = False
            pruned.append(m["name"])

    if not pruned:
        logger.info(f"{C['green']}✓ Nothing to prune{C['reset']}")
        return

    save_probe_data(data)
    logger.info(f"{C['bold']}Disabled {len(pruned)} model(s) in probe_models.json:{C['reset']}")
    for name in pruned:
        logger.info(f"  {C['red']}-{C['reset']} {name}  (enabled → false)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ollama model updater v3.0")
    parser.add_argument("--mode",  choices=["auto", "all", "check"],
                        default=os.environ.get("MODEL_UPDATE_MODE", "auto"))
    parser.add_argument("--model", metavar="NAME")
    parser.add_argument("--list",  action="store_true")
    parser.add_argument("--sync",  action="store_true",
                        help="Add installed models missing from probe_models.json")
    parser.add_argument("--prune", action="store_true",
                        help="Disable probe entries no longer installed")
    args = parser.parse_args()

    settings = load_settings()
    logger   = setup_logging(
        settings.get("log_file", "~/Library/Logs/llm-model-update/ollama.log"),
        name="ollama_updater",
    )

    if not ollama_running():
        logger.error(f"{C['red']}✗ Ollama not running — start with: ollama serve{C['reset']}")
        sys.exit(1)

    if args.list:
        list_models(logger)
    elif args.sync:
        sync_models(logger)
    elif args.prune:
        prune_models(logger)
    else:
        run_update(mode=args.mode, single=args.model, logger=logger, settings=settings)


if __name__ == "__main__":
    main()
