#!/usr/bin/env python3
"""
mlx/mlx_updater.py
──────────────────
Selectively update mlx-lm / HuggingFace models based on mlx/models.yaml.
Uses the HF Hub API to compare the remote HEAD commit hash against the locally
cached revision — no data is downloaded unless the repo has actually changed.

Usage
─────
  python3 mlx_updater.py                   # auto mode — follow models.yaml
  python3 mlx_updater.py --mode all        # refresh every model if stale
  python3 mlx_updater.py --mode check      # dry-run: show what's stale
  python3 mlx_updater.py --model mlx-community/Qwen3-4B-4bit
  python3 mlx_updater.py --list            # inventory with stale status

Requirements
────────────
  pip install huggingface_hub pyyaml
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path setup so shared/ is importable when run directly ────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.log_utils import setup_logging, C
from shared.config_utils import load_yaml_config

# ── HuggingFace dependency check ─────────────────────────────────────────────
try:
    from huggingface_hub import snapshot_download, model_info
    from huggingface_hub.utils import RepositoryNotFoundError
    HAS_HF = True
except ImportError:
    HAS_HF = False

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "models.yaml"


# ── HuggingFace cache helpers ─────────────────────────────────────────────────
def hf_cache_root(override: str = "") -> Path:
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface"),
    )) / "hub"


def repo_cache_dir(cache_root: Path, repo_id: str) -> Path:
    safe = repo_id.replace("/", "--")
    return cache_root / f"models--{safe}"


def get_local_commit(cache_root: Path, repo_id: str, revision: str = "main") -> Optional[str]:
    """Read the commit hash last downloaded for this repo+revision."""
    refs_file = repo_cache_dir(cache_root, repo_id) / "refs" / revision
    if refs_file.exists():
        return refs_file.read_text().strip()
    # Fallback: most-recently-modified snapshot
    snapshots = repo_cache_dir(cache_root, repo_id) / "snapshots"
    if snapshots.exists():
        hashes = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)
        if hashes:
            return hashes[-1].name
    return None


def get_remote_commit(
    repo_id: str, revision: str = "main", token: Optional[str] = None
) -> Optional[str]:
    """Query the HF Hub for the current HEAD commit hash. Returns None on failure."""
    try:
        info = model_info(repo_id, revision=revision, token=token or None)
        return info.sha
    except RepositoryNotFoundError:
        return None
    except Exception:
        return None


def is_stale(
    cache_root: Path, repo_id: str, revision: str, token: Optional[str]
) -> tuple[bool, str, str]:
    """Return (stale, local_hash_prefix, remote_hash_prefix)."""
    local  = get_local_commit(cache_root, repo_id, revision) or "—not cached—"
    remote = get_remote_commit(repo_id, revision, token)

    if remote is None:
        return False, local, "—unreachable—"

    return (local[:12] != remote[:12]), local[:12], remote[:12]


# ── Download ──────────────────────────────────────────────────────────────────
def pull_model(
    repo_id:    str,
    revision:   str,
    cache_root: Path,
    token:      Optional[str],
    dry_run:    bool,
    delay:      float,
    logger,
) -> str:
    stale, local_h, remote_h = is_stale(cache_root, repo_id, revision, token)

    if not stale and local_h != "—not cached—":
        logger.info(f"  {C['green']}✓ CURRENT{C['reset']}    {repo_id}  [{local_h}]")
        return "current"

    if dry_run:
        if local_h == "—not cached—":
            logger.info(
                f"  {C['cyan']}~ DRY-RUN{C['reset']}    {repo_id}  "
                f"[not cached → would download]"
            )
        else:
            logger.info(
                f"  {C['cyan']}~ DRY-RUN{C['reset']}    {repo_id}  "
                f"[{local_h} → {remote_h}]"
            )
        return "dry_run"

    action = "Downloading" if local_h == "—not cached—" else "Updating"
    logger.info(
        f"  {C['bold']}↓ {action}{C['reset']}  {repo_id}  [{local_h} → {remote_h}]"
    )
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_root),
            token=token or None,
        )
        time.sleep(delay)
        new_local = get_local_commit(cache_root, repo_id, revision) or "?"
        logger.info(f"  {C['green']}✓ UPDATED{C['reset']}    {repo_id}  → [{new_local[:12]}]")
        return "updated"
    except Exception as exc:
        logger.error(f"  {C['red']}✗ FAILED{C['reset']}     {repo_id}\n    {exc}")
        return "failed"


# ── Core logic ────────────────────────────────────────────────────────────────
def run_update(
    mode:   str,
    single: Optional[str],
    logger,
    config: dict,
) -> None:
    settings   = config.get("settings", {})
    delay      = float(settings.get("download_delay_seconds", 3))
    token      = settings.get("hf_token") or os.environ.get("HF_TOKEN") or None
    cache_root = hf_cache_root(settings.get("hf_cache_dir", ""))
    dry_run    = (mode == "check")

    model_entries: list[dict] = config.get("models", [])

    if single:
        work_set = [e for e in model_entries if e["repo_id"] == single]
        if not work_set:
            work_set = [{"repo_id": single, "update": True, "revision": "main", "notes": ""}]
    elif mode == "all":
        work_set = model_entries
    else:
        work_set = [e for e in model_entries if e.get("update", True) is not False]

    logger.info("")
    logger.info(
        f"{C['bold']}MLX Updater  [{mode.upper()} mode]  "
        f"{datetime.now():%Y-%m-%d %H:%M}{C['reset']}"
    )
    logger.info(f"  Cache dir : {cache_root}")
    logger.info(f"  Models    : {len(work_set)}")
    logger.info(f"  Dry run   : {dry_run}")
    logger.info("")

    counts: dict[str, int] = {
        "updated": 0, "current": 0, "failed": 0, "skipped": 0, "dry_run": 0
    }

    for entry in work_set:
        repo_id  = entry["repo_id"]
        revision = entry.get("revision", "main")
        flag     = entry.get("update", True)

        if flag is False and mode != "all":
            logger.info(f"  {C['yellow']}⊘ SKIP{C['reset']}       {repo_id}  (pinned)")
            counts["skipped"] += 1
            continue

        effective_dry = dry_run or (flag == "check" and mode == "auto")
        status = pull_model(
            repo_id=repo_id,
            revision=revision,
            cache_root=cache_root,
            token=token,
            dry_run=effective_dry,
            delay=delay,
            logger=logger,
        )
        counts[status] = counts.get(status, 0) + 1

    logger.info("")
    logger.info(
        f"{C['bold']}Summary:{C['reset']}  "
        f"Updated={counts['updated']}  "
        f"Current={counts['current']}  "
        f"Skipped={counts['skipped']}  "
        f"Failed={counts['failed']}"
        + (f"  DryRun={counts['dry_run']}" if counts["dry_run"] else "")
    )
    logger.info("")


def list_models(logger, config: dict) -> None:
    settings   = config.get("settings", {})
    token      = settings.get("hf_token") or os.environ.get("HF_TOKEN") or None
    cache_root = hf_cache_root(settings.get("hf_cache_dir", ""))

    logger.info(f"\n{'REPO_ID':<55} {'LOCAL':>12}  {'REMOTE':>12}  STATUS        UPDATE")
    logger.info("─" * 115)

    for entry in config.get("models", []):
        repo_id  = entry["repo_id"]
        revision = entry.get("revision", "main")
        flag     = entry.get("update", True)

        local  = (get_local_commit(cache_root, repo_id, revision) or "—")[:12]
        remote = (get_remote_commit(repo_id, revision, token) or "—")[:12]

        if local == "—":
            status = f"{C['yellow']}NOT CACHED  {C['reset']}"
        elif local == remote:
            status = f"{C['green']}CURRENT     {C['reset']}"
        else:
            status = f"{C['cyan']}STALE       {C['reset']}"

        logger.info(f"{repo_id:<55} {local:>12}  {remote:>12}  {status}  {flag}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not HAS_HF:
        print("✗ huggingface_hub not installed.  Run: pip install huggingface_hub")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Selective mlx-lm / HuggingFace model updater — driven by mlx/models.yaml"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "all", "check"],
        default=os.environ.get("MLX_UPDATE_MODE", "auto"),
        help="auto (default) | all | check (dry-run)",
    )
    parser.add_argument("--model", metavar="REPO_ID", help="Update a single model by repo ID")
    parser.add_argument("--list",  action="store_true", help="Show inventory with stale status")
    args = parser.parse_args()

    config = load_yaml_config(CONFIG_FILE)
    logger = setup_logging(
        config.get("settings", {}).get("log_file", "~/Library/Logs/llm-model-update/mlx.log"),
        name="mlx_updater",
    )

    if args.list:
        list_models(logger, config)
        return

    run_update(mode=args.mode, single=args.model, logger=logger, config=config)


if __name__ == "__main__":
    main()
