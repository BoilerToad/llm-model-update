#!/usr/bin/env python3
"""
ollama_log_analyzer.py — Analyze ~/.ollama/logs/server.log

Understands three Ollama log formats:
  print_info: general.name = Qwen2.5 7B Instruct   (llama.cpp model metadata)
  Structured: time=2026-03-18T... level=INFO source=server.go msg="..."
  GIN HTTP:   [GIN] 2026/03/18 - 08:28:21 | 200 | 33.494s | 127.0.0.1 | POST "/api/chat"

Usage:
    python ollama_log_analyzer.py                   # full report
    python ollama_log_analyzer.py --tail 500        # last N lines only
    python ollama_log_analyzer.py --since 2026-03-18
    python ollama_log_analyzer.py --errors-only
    python ollama_log_analyzer.py --slow 30
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

LOG_PATH = Path("~/.ollama/logs/server.log").expanduser()

# ── Regex patterns ────────────────────────────────────────────────────────────

RE_STRUCT_TS  = re.compile(r'^time=(\S+)')
RE_STRUCT_LVL = re.compile(r'level=(\w+)')
RE_STRUCT_MSG = re.compile(r'msg="([^"]*)"')

RE_GIN = re.compile(
    r'\[GIN\] (\d{4}/\d{2}/\d{2}) - (\d{2}:\d{2}:\d{2}) \| '
    r'(\d{3}) \| +([^\|]+?) +\| +([^\|]+?) +\| '
    r'(\w+) ++"?(/[^"\s]*)"?'
)

# print_info lines — no timestamp, raw llama.cpp output
# e.g. "print_info: general.name     = Qwen2.5 7B Instruct"
RE_PRINT_INFO = re.compile(r'^print_info:\s+([\w.]+)\s+=\s+(.+)$')

RE_DUR_MS  = re.compile(r'^([\d.]+)ms$')
RE_DUR_S   = re.compile(r'^([\d.]+)s$')
RE_DUR_MNS = re.compile(r'^(\d+)m(\d+(?:\.\d+)?)s$')
RE_DUR_US  = re.compile(r'^([\d.]+)[µu]s$')

RE_GPU_AVAIL = re.compile(r'available="([\d.]+ \w+)"')
RE_LAYERS    = re.compile(r'offloaded (\d+)/(\d+) layers')
RE_LOAD_SEC  = re.compile(r'runner started in ([\d.]+) seconds')
RE_MODEL_REQ = re.compile(r'model requires more gpu memory')
RE_EVICT     = re.compile(r'evict', re.I)

# print_info fields we care about
PRINT_INFO_FIELDS = {
    "general.name",
    "arch",
    "model type",
    "model params",
    "file type",
    "file size",
}


def parse_gin_ts(date_str, time_str):
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S")
    except ValueError:
        return None


def parse_struct_ts(ts_str):
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_duration_s(dur):
    dur = dur.strip()
    m = RE_DUR_MNS.match(dur)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = RE_DUR_S.match(dur)
    if m:
        return float(m.group(1))
    m = RE_DUR_MS.match(dur)
    if m:
        return float(m.group(1)) / 1000
    m = RE_DUR_US.match(dur)
    if m:
        return float(m.group(1)) / 1_000_000
    return None


# ── Data structures ───────────────────────────────────────────────────────────

class ModelInfo:
    """Metadata extracted from print_info lines before a session starts."""
    def __init__(self):
        self.name:       Optional[str] = None   # general.name
        self.arch:       Optional[str] = None
        self.model_type: Optional[str] = None
        self.params:     Optional[str] = None
        self.file_type:  Optional[str] = None
        self.file_size:  Optional[str] = None

    def absorb(self, key: str, value: str):
        value = value.strip()
        if key == "general.name":   self.name       = value
        elif key == "arch":         self.arch       = value
        elif key == "model type":   self.model_type = value
        elif key == "model params": self.params     = value
        elif key == "file type":    self.file_type  = value
        elif key == "file size":    self.file_size  = value

    def is_populated(self) -> bool:
        return bool(self.name or self.arch)

    def summary(self) -> str:
        parts = []
        if self.name:       parts.append(self.name)
        if self.arch:       parts.append(f"arch={self.arch}")
        if self.params:     parts.append(f"params={self.params}")
        if self.file_size:  parts.append(f"size={self.file_size}")
        if self.file_type:  parts.append(f"quant={self.file_type}")
        return "  |  ".join(parts) if parts else "unknown"

    def clear(self):
        self.__init__()


class RequestRecord:
    def __init__(self, ts, method, path, status, duration_s):
        self.ts         = ts
        self.method     = method
        self.path       = path
        self.status     = status
        self.duration_s = duration_s

    def endpoint(self):
        if "/api/chat"     in self.path: return "chat"
        if "/api/generate" in self.path: return "generate"
        if "/api/tags"     in self.path: return "tags"
        if "/api/ps"       in self.path: return "ps"
        return self.path


class LogSession:
    def __init__(self, ts, model_info: Optional[ModelInfo] = None):
        self.start_ts       = ts
        self.model_info     = model_info   # populated from preceding print_info block
        self.requests       = []
        self.events         = []           # (ts, level, msg) — always 3-tuple
        self.evictions      = 0
        self.gpu_available  = []
        self.load_time_s    = None
        self.layers_offloaded = None

    @property
    def model_name(self) -> str:
        if self.model_info and self.model_info.name:
            return self.model_info.name
        return "unknown"

    def add_request(self, rec):
        self.requests.append(rec)

    def add_event(self, ts, level, msg):
        self.events.append((ts, level, msg))

    @property
    def errors(self):
        return [(ts, level, msg)
                for ts, level, msg in self.events
                if level in ("ERROR", "WARN", "WARNING")]

    def duration_stats(self, endpoint):
        recs = [r for r in self.requests
                if r.endpoint() == endpoint and r.duration_s is not None]
        if not recs:
            return None
        d = [r.duration_s for r in recs]
        return {"count": len(d), "avg": sum(d)/len(d), "min": min(d), "max": max(d)}


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_log(log_path, tail_lines=None, since_date=None):
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if tail_lines:
        lines = lines[-tail_lines:]

    sessions     = []
    current      = None
    all_requests = []

    # Accumulate print_info fields until a session start is seen
    pending_info = ModelInfo()

    def ensure_session(ts=None):
        nonlocal current
        if current is None:
            current = LogSession(ts)
            sessions.append(current)
        return current

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # ── print_info line (no timestamp) ────────────────────────────────────
        m = RE_PRINT_INFO.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            if key in PRINT_INFO_FIELDS:
                pending_info.absorb(key, value)
            continue

        # ── GIN HTTP line ─────────────────────────────────────────────────────
        m = RE_GIN.match(line)
        if m:
            date_s, time_s, status, duration_raw, remote, method, path = m.groups()
            ts    = parse_gin_ts(date_s, time_s)
            dur_s = parse_duration_s(duration_raw.strip())
            if since_date and ts and ts.date() < since_date:
                continue
            rec = RequestRecord(ts, method, path, int(status), dur_s)
            ensure_session(ts).add_request(rec)
            all_requests.append(rec)
            continue

        # ── Structured log line ───────────────────────────────────────────────
        m_ts = RE_STRUCT_TS.match(line)
        if not m_ts:
            continue

        ts = parse_struct_ts(m_ts.group(1))
        if since_date and ts and ts.date() < since_date:
            continue

        lv_m  = RE_STRUCT_LVL.search(line)
        level = lv_m.group(1).upper() if lv_m else "INFO"
        msg_m = RE_STRUCT_MSG.search(line)
        msg   = msg_m.group(1) if msg_m else ""

        # Model load — new session, attach any accumulated print_info metadata
        if "llama runner started" in line:
            m_load = RE_LOAD_SEC.search(line)
            info   = pending_info if pending_info.is_populated() else None
            current = LogSession(ts, model_info=info)
            current.load_time_s = float(m_load.group(1)) if m_load else None
            sessions.append(current)
            pending_info = ModelInfo()   # reset for next model

        elif RE_MODEL_REQ.search(msg) or RE_EVICT.search(msg):
            sess = ensure_session(ts)
            sess.evictions += 1
            sess.add_event(ts, "WARN", f"EVICTION: {msg}")

        elif "gpu memory" in msg or "updated VRAM" in msg:
            m_avail = RE_GPU_AVAIL.search(line)
            if m_avail:
                ensure_session(ts).gpu_available.append(m_avail.group(1))

        elif "offloaded" in msg:
            m_lay = RE_LAYERS.search(line)
            if m_lay:
                ensure_session(ts).layers_offloaded = f"{m_lay.group(1)}/{m_lay.group(2)}"

        elif level in ("ERROR", "WARN", "WARNING"):
            ensure_session(ts).add_event(ts, level, msg or line[:120])

        elif any(kw in msg.lower() for kw in ("loading model", "waiting for llama", "memory")):
            ensure_session(ts).add_event(ts, "INFO", msg)

    return sessions, all_requests


# ── Report ────────────────────────────────────────────────────────────────────

def fmt_dur(s):
    if s is None: return "—"
    if s >= 60:   return f"{int(s//60)}m{s%60:.1f}s"
    if s >= 1:    return f"{s:.1f}s"
    return f"{s*1000:.0f}ms"


def fmt_ts(ts):
    if ts is None: return "??:??:??"
    return ts.strftime("%H:%M:%S")


def print_report(sessions, all_requests, slow_threshold, errors_only):
    SEP  = "─" * 72
    DSEP = "━" * 72

    print(f"\n{SEP}")
    print(f"  Ollama Log Analyzer  —  {LOG_PATH}")
    print(SEP)

    if all_requests and not errors_only:
        chat_all = [r for r in all_requests if r.endpoint() == "chat"     and r.duration_s]
        gen_all  = [r for r in all_requests if r.endpoint() == "generate" and r.duration_s]

        print(f"\n  Overall request summary ({len(all_requests)} total requests):\n")
        for label, recs in [("chat", chat_all), ("generate", gen_all)]:
            if recs:
                avg = sum(r.duration_s for r in recs) / len(recs)
                mn  = min(r.duration_s for r in recs)
                mx  = max(r.duration_s for r in recs)
                print(f"    /api/{label:<10}  n={len(recs):>3}  "
                      f"avg={fmt_dur(avg)}  min={fmt_dur(mn)}  max={fmt_dur(mx)}")

        slow_all = [r for r in all_requests
                    if r.duration_s and r.duration_s > slow_threshold
                    and r.endpoint() in ("chat", "generate")]
        if slow_all:
            print(f"\n  Slow requests (>{slow_threshold:.0f}s):  {len(slow_all)}")
            for r in slow_all[-10:]:
                print(f"    [{fmt_ts(r.ts)}] {r.endpoint():<10} {fmt_dur(r.duration_s)}")

    print(f"\n  Sessions detected: {len(sessions)}\n")

    for i, sess in enumerate(sessions, 1):
        if errors_only and not sess.errors:
            continue

        print(DSEP)

        # Session header — now includes model name from print_info
        model_str = sess.model_name
        if sess.model_info and sess.model_info.is_populated():
            model_str = sess.model_info.summary()

        print(f"  Session {i}  [{model_str}]")
        print(f"  started={fmt_ts(sess.start_ts)}  "
              f"requests={len(sess.requests)}  evictions={sess.evictions}")

        if sess.load_time_s is not None:
            print(f"  Model load time : {sess.load_time_s:.2f}s")
        if sess.layers_offloaded:
            print(f"  GPU layers      : {sess.layers_offloaded} offloaded")
        if sess.gpu_available:
            print(f"  GPU available   : {sess.gpu_available[-1]} (last seen)")

        if not errors_only:
            for ep in ("chat", "generate"):
                stats = sess.duration_stats(ep)
                if stats:
                    print(f"\n  /api/{ep}:")
                    print(f"    count={stats['count']}  avg={fmt_dur(stats['avg'])}  "
                          f"min={fmt_dur(stats['min'])}  max={fmt_dur(stats['max'])}")
                    slow = [r for r in sess.requests
                            if r.endpoint() == ep
                            and r.duration_s and r.duration_s > slow_threshold]
                    if slow:
                        print(f"    ⚠ {len(slow)} slow (>{slow_threshold:.0f}s):")
                        for r in slow:
                            print(f"      [{fmt_ts(r.ts)}] {fmt_dur(r.duration_s)}")

        if sess.errors:
            print(f"\n  ✗ Errors / Warnings ({len(sess.errors)}):")
            for ts, level, msg in sess.errors:
                print(f"    [{fmt_ts(ts)}] {level:<7} {msg[:100]}")

        notable = [(ts, msg) for ts, level, msg in sess.events if level == "INFO"]
        if notable and not errors_only:
            print(f"\n  Notable events:")
            for ts, msg in notable[-5:]:
                print(f"    [{fmt_ts(ts)}] {msg[:100]}")

        if not sess.requests and not sess.errors:
            print(f"  (no requests or errors)")

    if not errors_only:
        slow_sorted = sorted(
            [r for r in all_requests
             if r.duration_s and r.endpoint() in ("chat", "generate")],
            key=lambda r: r.duration_s, reverse=True
        )
        if slow_sorted:
            print(f"\n{SEP}")
            print(f"  Top 10 slowest requests:\n")
            print(f"  {'Time':>8}  {'Endpoint':<10}  {'Duration':>10}  Status")
            print(f"  {'─'*8}  {'─'*10}  {'─'*10}  ──────")
            for r in slow_sorted[:10]:
                flag = "  ⚠" if r.duration_s > slow_threshold else ""
                print(f"  {fmt_ts(r.ts):>8}  {r.endpoint():<10}  "
                      f"{fmt_dur(r.duration_s):>10}  {r.status}{flag}")

    # ── Model name summary table ──────────────────────────────────────────────
    named = [(i+1, s) for i, s in enumerate(sessions) if s.model_info and s.model_info.name]
    if named and not errors_only:
        print(f"\n{SEP}")
        print(f"  Model identity map (from print_info):\n")
        print(f"  {'Sess':>4}  {'Model name (self-reported)':<38}  {'Arch':<10}  {'Params':<12}  Quant")
        print(f"  {'─'*4}  {'─'*38}  {'─'*10}  {'─'*12}  ─────")
        for idx, sess in named:
            mi = sess.model_info
            print(f"  {idx:>4}  {(mi.name or '—'):<38}  "
                  f"{(mi.arch or '—'):<10}  "
                  f"{(mi.params or '—'):<12}  "
                  f"{mi.file_type or '—'}")

    print(f"\n{SEP}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Ollama server.log for performance issues and errors"
    )
    parser.add_argument("--log",         default=str(LOG_PATH))
    parser.add_argument("--tail",        type=int,   metavar="N")
    parser.add_argument("--since",       metavar="YYYY-MM-DD")
    parser.add_argument("--slow",        type=float, default=30.0, metavar="SECONDS")
    parser.add_argument("--errors-only", action="store_true")
    args = parser.parse_args()

    since_date = None
    if args.since:
        try:
            since_date = date.fromisoformat(args.since)
        except ValueError:
            print(f"Invalid date: {args.since}")
            sys.exit(1)

    log_path = Path(args.log).expanduser()
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        sys.exit(1)

    sessions, all_requests = parse_log(log_path, args.tail, since_date)
    print_report(sessions, all_requests, args.slow, args.errors_only)


if __name__ == "__main__":
    main()
