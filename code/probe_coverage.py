"""
probe_coverage.py  v1.1
─────────────────────────────────────────────────────────────────────────────
Coverage checker — reads all probe JSON results and cross-references against
probe_models.json and questions.json to produce:

  1. A coverage matrix showing what has been run vs what is still needed
  2. A prioritised rerun list with ready-to-paste shell commands
  3. A quality report flagging responses that succeeded but need a rerun

What counts as "good" coverage for a model/question pair:
  - chat endpoint: success=True AND len > MIN_CHAT_LEN (100c)
  - generate endpoint: success=True AND len > MIN_GEN_LEN (50c)

Usage:
    python probe_coverage.py                    # full report to stdout
    python probe_coverage.py --quiet            # rerun commands only
    python probe_coverage.py --out coverage.md  # save report to file
    python probe_coverage.py --model deepseek   # filter by model substring
    python probe_coverage.py --question Q03     # single question
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR          = Path(__file__).parent
ROOT_DIR            = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "data" / "probes"
REPORTS_DIR         = ROOT_DIR / "results" / "reports" / "coverage"
MODELS_FILE         = ROOT_DIR / "probes" / "probe_models.json"
QUESTIONS_FILE      = ROOT_DIR / "probes" / "questions.json"

MIN_CHAT_LEN  = 100
MIN_GEN_LEN   = 50
SKIP_FAMILIES = {"bert", "clip"}
SKIP_NAMES    = {"mxbai-embed-large:latest"}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_registry(models_file: Path = MODELS_FILE) -> list[dict]:
    data = json.loads(models_file.read_text())
    return [
        m for m in data["models"]
        if m.get("enabled", True)
        and m.get("family", "") not in SKIP_FAMILIES
        and m["name"] not in SKIP_NAMES
    ]


def load_questions(questions_file: Path = QUESTIONS_FILE) -> list[dict]:
    return json.loads(questions_file.read_text())["questions"]


def load_all_results(results_dir: Path) -> dict:
    """
    Scan all JSON files in results_dir.
    Returns: {model: {qid: {"chat": resp, "generate": resp}}}
    Keeps the longest successful response per (model, qid, endpoint).
    """
    best: dict = defaultdict(lambda: defaultdict(dict))

    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in data:
            if entry.get("skipped"):
                continue
            model = entry["model"]
            for key, resp in entry.get("responses", {}).items():
                # key format: "{QID}_{endpoint}"  e.g. "Q03_chat", "Q10b_generate"
                # Use rsplit with maxsplit=1 to handle question IDs like "Q10b"
                parts = key.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                qid, endpoint = parts[0], parts[1]
                if endpoint not in ("chat", "generate"):
                    continue
                text    = resp.get("answer", "") or resp.get("raw_response", "") or ""
                success = resp.get("success", False)
                existing = best[model][qid].get(endpoint)
                existing_len = len(
                    (existing or {}).get("answer", "") or
                    (existing or {}).get("raw_response", "") or ""
                ) if existing else 0
                if success and len(text) > existing_len:
                    best[model][qid][endpoint] = resp

    return best


# ── Coverage logic ─────────────────────────────────────────────────────────────

def response_text(resp: dict | None) -> str:
    if not resp:
        return ""
    return resp.get("answer", "") or resp.get("raw_response", "") or ""


def is_good(resp: dict | None, is_cloud: bool = False, endpoint: str = "chat") -> bool:
    """True if the response is successful and meets minimum length."""
    if not resp or not resp.get("success"):
        return False
    min_len = MIN_GEN_LEN if (endpoint == "generate" or is_cloud) else MIN_CHAT_LEN
    return len(response_text(resp)) >= min_len


def coverage_status(model: str, qid: str, results: dict, is_cloud: bool) -> dict:
    """
    Return a status dict for a single (model, question) pair.

    Keys are consistent — always use "chat_*" and "generate_*" prefixes
    so callers can safely do st["chat_len"], st["generate_len"] etc.
    """
    q   = results.get(model, {}).get(qid, {})
    cr  = q.get("chat")
    gr  = q.get("generate")

    chat_ok  = is_good(cr, is_cloud, "chat")
    gen_ok   = is_good(gr, is_cloud, "generate")
    chat_len = len(response_text(cr))
    gen_len  = len(response_text(gr))

    return {
        # Availability flags
        "chat_ok":      chat_ok,
        "generate_ok":  gen_ok,
        # Lengths — BOTH short and full names so callers can use either
        "chat_len":     chat_len,
        "gen_len":      gen_len,          # short alias
        "generate_len": gen_len,          # full alias — used by f"{ep}_len"
        # Error strings
        "chat_err":      (cr or {}).get("error", "") if not chat_ok else "",
        "gen_err":        (gr or {}).get("error", "") if not gen_ok  else "",
        "generate_err":   (gr or {}).get("error", "") if not gen_ok  else "",
        # Convenience flags
        "needs_chat":    not chat_ok,
        "needs_gen":     not gen_ok,
        "needs_generate": not gen_ok,
        "needs_any":     not chat_ok or not gen_ok,
    }


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(
    results_dir: Path,
    filter_model: str | None    = None,
    filter_question: str | None = None,
    quiet: bool                 = False,
    models_file: Path           = MODELS_FILE,
    questions_file: Path        = QUESTIONS_FILE,
) -> str:
    registry  = load_registry(models_file)
    questions = load_questions(questions_file)
    results   = load_all_results(results_dir)

    if filter_model:
        registry  = [m for m in registry  if filter_model.lower() in m["name"].lower()]
    if filter_question:
        questions = [q for q in questions if q["id"] == filter_question.upper()]

    # Build matrix and gap list
    matrix: dict = {}
    gaps:   dict = defaultdict(list)   # model -> [(qid, [missing_eps], status)]

    for m in registry:
        model    = m["name"]
        is_cloud = "cloud" in model.lower()
        matrix[model] = {}
        for q in questions:
            qid = q["id"]
            st  = coverage_status(model, qid, results, is_cloud)
            matrix[model][qid] = st
            if st["needs_any"]:
                missing = []
                if st["needs_chat"]: missing.append("chat")
                if st["needs_gen"]:  missing.append("generate")
                gaps[model].append((qid, missing, st))

    total_cells     = len(registry) * len(questions)
    chat_done       = sum(1 for m in matrix.values() for s in m.values() if s["chat_ok"])
    gen_done        = sum(1 for m in matrix.values() for s in m.values() if s["generate_ok"])
    both_done       = sum(1 for m in matrix.values() for s in m.values() if s["chat_ok"] and s["generate_ok"])
    models_complete = sum(1 for m in registry if not gaps[m["name"]])

    lines = []

    if not quiet:
        pct = lambda n: round(100 * n / max(total_cells, 1))
        lines += [
            "# Probe Coverage Report",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            f"*Results dir: {results_dir}*",
            "",
            "## Summary",
            "",
            f"- Models in registry (enabled): **{len(registry)}**",
            f"- Questions in bank: **{len(questions)}**",
            f"- Total cells (model × question): **{total_cells}**",
            f"- Chat complete: **{chat_done}/{total_cells}** ({pct(chat_done)}%)",
            f"- Generate complete: **{gen_done}/{total_cells}** ({pct(gen_done)}%)",
            f"- Both endpoints complete: **{both_done}/{total_cells}** ({pct(both_done)}%)",
            f"- Models fully covered: **{models_complete}/{len(registry)}**",
            "",
            "## Coverage matrix",
            "",
            "Legend: `✓` both · `C` chat only · `G` generate only · `?` partial · `·` neither",
            "",
        ]

        q_ids  = [q["id"] for q in questions]
        lines.append("| Model | Geo | " + " | ".join(f"`{q}`" for q in q_ids) + " |")
        lines.append("|---|---|" + "---|" * len(q_ids))

        for m in registry:
            model    = m["name"]
            is_cloud = "cloud" in model.lower()
            geo      = (m.get("geopolitical_origin", "") or "")[:18]
            prefix   = "☁ " if is_cloud else ""
            cells    = []
            for qid in q_ids:
                st = matrix[model][qid]
                if   st["chat_ok"] and st["generate_ok"]:     cells.append("✓")
                elif st["chat_ok"]:                            cells.append("C")
                elif st["generate_ok"]:                        cells.append("G")
                elif st["chat_len"] > 0 or st["gen_len"] > 0: cells.append("?")
                else:                                          cells.append("·")
            lines.append(f"| {prefix}{model[:38]} | {geo} | " + " | ".join(cells) + " |")

        lines += ["", "## Gap detail by model", ""]

        for m in registry:
            model      = m["name"]
            model_gaps = gaps[model]
            if not model_gaps:
                lines.append(f"**{model}** — ✓ fully covered")
                continue
            is_cloud = "cloud" in model.lower()
            origin   = m.get("geopolitical_origin", "")
            lines.append(f"**{model}** [{origin}]{'  ☁' if is_cloud else ''}")
            for qid, missing_eps, st in sorted(model_gaps):
                parts = []
                for ep in missing_eps:
                    err = st[f"{ep}_err"]
                    ln  = st[f"{ep}_len"]
                    tag = f"err={err[:30]}" if err else f"len={ln}"
                    parts.append(f"{ep}({tag})")
                lines.append(f"  - {qid}: {', '.join(parts)}")
            lines.append("")

    # ── Rerun commands ─────────────────────────────────────────────────────────
    local_full: dict  = defaultdict(set)   # chat missing
    local_gen:  dict  = defaultdict(set)   # generate only missing
    cloud_full: dict  = defaultdict(set)
    cloud_gen:  dict  = defaultdict(set)

    for m in registry:
        model    = m["name"]
        is_cloud = "cloud" in model.lower()
        for qid, missing_eps, _st in gaps[model]:
            if "chat" in missing_eps:
                (cloud_full if is_cloud else local_full)[model].add(qid)
            else:
                (cloud_gen if is_cloud else local_gen)[model].add(qid)

    lines += ["" if quiet else "", "## Rerun commands", ""]

    if not quiet:
        lines += ["Copy-paste to fill gaps. Run local in parallel with cloud.", ""]

    def _commands(gaps_dict: dict, timeout: int, label: str) -> list[str]:
        out = []
        q_to_models: dict = defaultdict(list)
        for model, qids in sorted(gaps_dict.items()):
            q_to_models[tuple(sorted(qids))].append(model)
        for qids_tuple, models in sorted(q_to_models.items()):
            out += [
                "```bash",
                "python probe_static.py \\",
                f"  --questions {' '.join(sorted(qids_tuple))} \\",
                "  --models \\",
                "    " + " \\\n    ".join(models) + " \\",
                f"  --generate-timeout {timeout} \\",
                f"  --label {label}",
                "```",
                "",
            ]
        return out

    if local_full:
        lines += ["### Local — chat + generate needed", ""]
        lines += _commands(local_full, 600, "rerun_local")

    if local_gen:
        lines += ["### Local — generate only (chat already done)", ""]
        lines += _commands(local_gen, 600, "rerun_gen_only")

    combined_cloud = defaultdict(set)
    for model, qids in {**cloud_full, **cloud_gen}.items():
        combined_cloud[model] |= qids
    if combined_cloud:
        lines += ["### Cloud — reruns needed", ""]
        lines += _commands(combined_cloud, 1200, "rerun_cloud")

    if not local_full and not local_gen and not combined_cloud:
        lines += ["### ✓ All models fully covered — no reruns needed", ""]

    # ── Quality flags ──────────────────────────────────────────────────────────
    if not quiet:
        suspicious = []
        for m in registry:
            model    = m["name"]
            is_cloud = "cloud" in model.lower()
            for q in questions:
                st = matrix[m["name"]][q["id"]]
                for ep in ("chat", "generate"):
                    ln        = st[f"{ep}_len"]
                    threshold = MIN_GEN_LEN if (ep == "generate" or is_cloud) else MIN_CHAT_LEN
                    if 0 < ln < threshold:
                        suspicious.append((model, q["id"], ep, ln))
        if suspicious:
            lines += [
                "## Quality flags — present but suspiciously short", "",
                "success=True but below minimum length — may need rerun.", "",
                "| Model | QID | Endpoint | Length |", "|---|---|---|---|",
            ]
            for model, qid, ep, ln in suspicious:
                lines.append(f"| {model} | {qid} | {ep} | {ln}c |")
            lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Probe coverage checker v1.2")
    parser.add_argument("--model",       default=None)
    parser.add_argument("--question",    default=None)
    parser.add_argument("--quiet",       action="store_true")
    parser.add_argument("--out",         default=None,
                        help="Output path. Use 'auto' to write a dated file to "
                             "results/reports/coverage/. Omit to print to stdout.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"✗ Results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    report = build_report(
        results_dir=results_dir,
        filter_model=args.model,
        filter_question=args.question,
        quiet=args.quiet,
    )

    if args.out == "auto" or args.out is None and not args.quiet and not args.model and not args.question:
        # Auto-dated output when saving a full report
        out_path = REPORTS_DIR / f"coverage_{datetime.now().strftime('%Y%m%d')}.md"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to {out_path}")
        print(report)
    elif args.out and args.out != "auto":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
