"""
relay_monitor.py
─────────────────────────────────────────────────────────────────────────────
Live terminal monitor for mcp-relay activity.

Polls the relay SQLite DB every second and prints new events as they arrive.
Run this in a second terminal while authoritarianism_probe_relay.py runs in
the first. All events are already being saved to the DB by the relay — this
script just presents them in real time.

Usage:
    python relay_monitor.py
    python relay_monitor.py --db ~/.mcp-relay/authoritarianism_probe.db
    python relay_monitor.py --db ~/.mcp-relay/authoritarianism_probe.db --tail 50
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

# ── ANSI colors ───────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"

def c(text, color): return f"{color}{text}{RESET}"

# ── Event formatting ──────────────────────────────────────────────────────────

EVENT_STYLES = {
    "call_start":   ("→ START  ", CYAN),
    "call_end":     ("✓ DONE   ", GREEN),
    "call_error":   ("✗ ERROR  ", RED),
    "call_blocked": ("⊘ BLOCKED", YELLOW),
    "mode_change":  ("~ MODE   ", DIM),
}

def format_event(row: sqlite3.Row, session_model_map: dict) -> str:
    etype      = row["event_type"]
    tool       = row["tool_name"]
    ts         = row["timestamp"][:19].replace("T", " ")
    latency    = row["latency_ms"]
    error      = row["error"] or ""
    session_id = row["session_id"]
    model      = session_model_map.get(session_id, "unknown")

    # Extract URL or query from payload
    payload = {}
    try:
        payload = json.loads(row["payload"]) if row["payload"] else {}
    except Exception:
        pass
    url = payload.get("url") or payload.get("query") or payload.get("uri") or ""
    if len(url) > 80:
        url = url[:77] + "..."

    label, color = EVENT_STYLES.get(etype, ("? UNKNOWN", DIM))

    parts = [
        c(ts, DIM),
        c(label, color),
        c(f"{model:<28}", BOLD),
        c(f"{tool:<10}", WHITE),
    ]
    if url:
        parts.append(c(url, CYAN))
    if latency is not None:
        parts.append(c(f"{latency:.0f}ms", DIM))
    if error:
        parts.append(c(f"  [{error[:80]}]", RED))

    return "  ".join(parts)


def print_header(db_path: str):
    print(f"\n{c('─'*72, DIM)}")
    print(f"  {c('mcp-relay monitor', BOLD)}  {c(db_path, DIM)}")
    print(f"  {c('Watching for new events. Ctrl+C to stop.', DIM)}")
    print(f"{c('─'*72, DIM)}\n")
    print(f"  {c('TIME             EVENT     MODEL                        TOOL', DIM)}")
    print(f"  {c('─'*68, DIM)}")


# ── Main poll loop ────────────────────────────────────────────────────────────

def monitor(db_path: Path, poll_interval: float, tail: int):
    print_header(str(db_path))

    conn = None
    last_row_id = 0
    session_model_map: dict[str, str] = {}
    stats = {"allowed": 0, "blocked": 0, "errors": 0}

    try:
        while True:
            # Reconnect if DB doesn't exist yet (probe may not have started)
            if not db_path.exists():
                print(f"\r  {c('Waiting for DB...', DIM)}", end="", flush=True)
                time.sleep(poll_interval)
                continue

            if conn is None:
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                conn.row_factory = sqlite3.Row

                # On first connect — show tail of existing events
                if tail > 0:
                    existing = conn.execute(
                        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (tail,)
                    ).fetchall()[::-1]
                    if existing:
                        last_row_id = existing[-1]["id"]
                        # Refresh session map
                        for s in conn.execute("SELECT session_id, model_name FROM sessions"):
                            session_model_map[s["session_id"]] = s["model_name"] or "unknown"
                        print(f"  {c(f'── last {len(existing)} events ──', DIM)}")
                        for row in existing:
                            print(format_event(row, session_model_map))
                        print(f"  {c('── live ──', DIM)}")

            # Refresh session map
            for s in conn.execute(
                "SELECT session_id, model_name FROM sessions"
            ):
                session_model_map[s["session_id"]] = s["model_name"] or "unknown"

            # Fetch new events since last seen
            new_rows = conn.execute(
                "SELECT * FROM events WHERE id > ? ORDER BY id",
                (last_row_id,),
            ).fetchall()

            for row in new_rows:
                last_row_id = row["id"]
                print(format_event(row, session_model_map))

                etype = row["event_type"]
                if etype == "call_end":
                    stats["allowed"] += 1
                elif etype == "call_blocked":
                    stats["blocked"] += 1
                elif etype == "call_error":
                    stats["errors"] += 1

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\n{c('─'*72, DIM)}")
        print(f"  {c('Session summary', BOLD)}")
        print(f"  Allowed : {c(str(stats['allowed']), GREEN)}")
        print(f"  Blocked : {c(str(stats['blocked']), YELLOW)}")
        print(f"  Errors  : {c(str(stats['errors']), RED)}")
        print(f"{c('─'*72, DIM)}\n")
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Live mcp-relay event monitor")
    parser.add_argument(
        "--db", default="~/.mcp-relay/authoritarianism_probe.db",
        help="Path to relay SQLite DB"
    )
    parser.add_argument(
        "--poll", default=1.0, type=float,
        help="Poll interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--tail", default=20, type=int,
        help="Show last N events on startup (default: 20, 0 = live only)"
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    monitor(db_path, args.poll, args.tail)


if __name__ == "__main__":
    main()
