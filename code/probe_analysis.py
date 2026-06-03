"""
probe_analysis.py
─────────────────────────────────────────────────────────────────────────────
Research analysis over the probe database. Produces the findings reports
that would feed into a paper.

Outputs (all written to results/analysis/):
  01_suppression_matrix.md      — CCP suppression across all models/questions
  02_divergence_report.md       — Chat vs generate divergence findings
  03_geopolitical_comparison.md — Four-pole (US/EU/China/Gulf) comparison
  04_paired_comparison.md       — Mechanism vs named-actor delta analysis
  05_scale_analysis.md          — Local vs cloud / small vs large comparisons
  06_notable_responses.md       — Full text of notable/unusual responses

Usage:
    python probe_analysis.py
    python probe_analysis.py --db results/probe_results.db
    python probe_analysis.py --report suppression  # single report
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH     = PROJECT_ROOT / "results" / "db" / "probe_results.db"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"

GEO_ORDER = ["US/Meta","US/Google","US/OpenAI","US/NVIDIA",
             "France/Mistral","France+US/Mistral+NVIDIA",
             "China/DeepSeek","China/Alibaba","China/ZHIPU",
             "UAE/TII","Saudi/SDAIA","unknown"]


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"✗ DB not found: {db_path}")
        print(f"  Run: python probe_db.py --ingest --summary")
        raise SystemExit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Report 1: Suppression matrix ──────────────────────────────────────────────

def report_suppression(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    lines = [
        "# Report 1 — CCP Suppression Matrix",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Rows: models. Columns: questions. Cell: classification on chat endpoint.",
        "ANALYST = normal response. CCP-VOICE = premise denial/institutional defense.",
        "REFUSAL = explicit refusal. EMPTY = success=True but no content.",
        "",
    ]

    # Get all models and questions
    models = [r[0] for r in cur.execute(
        "SELECT DISTINCT model FROM analysis ORDER BY geopolitical_origin, model")]
    questions = [r[0] for r in cur.execute(
        "SELECT DISTINCT question_id FROM analysis ORDER BY question_id")]

    # Build lookup
    data = {}
    for row in cur.execute("SELECT model, question_id, classification, length_chat FROM analysis"):
        data[(row["model"], row["question_id"])] = (row["classification"], row["length_chat"])

    # Header
    q_short = [q[:5] for q in questions]
    lines.append("| Model | Origin | " + " | ".join(q_short) + " |")
    lines.append("|" + "---|" * (len(questions) + 2))

    CLASS_SYMBOL = {
        "ANALYST":   "✓",
        "CCP-VOICE": "**CCP**",
        "REFUSAL":   "REF",
        "EMPTY":     "∅",
        None:        "—",
    }

    for model in models:
        origin = cur.execute(
            "SELECT geopolitical_origin FROM analysis WHERE model=? LIMIT 1", (model,)
        ).fetchone()
        origin_str = origin[0] if origin else "unknown"
        cells = []
        for qid in questions:
            cls, length = data.get((model, qid), (None, 0))
            sym = CLASS_SYMBOL.get(cls, "?")
            if cls == "ANALYST" and length:
                sym = f"✓{round(length/1000,1)}k"
            cells.append(sym)
        cloud = " ☁" if "cloud" in model else ""
        lines.append(f"| {model+cloud} | {origin_str} | " + " | ".join(cells) + " |")

    lines += ["", "## CCP-VOICE Detail", ""]
    for row in cur.execute("""
        SELECT model, geopolitical_origin, question_id, classification,
               length_chat, ccp_phrases, notes
        FROM analysis WHERE classification='CCP-VOICE'
        ORDER BY geopolitical_origin, model, question_id
    """):
        phrases = json.loads(row["ccp_phrases"] or "[]")
        lines.append(f"**{row['model']}** / {row['question_id']}")
        lines.append(f"- Origin: {row['geopolitical_origin']}")
        lines.append(f"- Length: {row['length_chat']}c")
        lines.append(f"- Phrases: {phrases[:3]}")
        lines.append("")

    return "\n".join(lines)


# ── Report 2: Chat vs generate divergence ─────────────────────────────────────

def report_divergence(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    lines = [
        "# Report 2 — Chat vs Generate Divergence",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "chat_gen_ratio = generate_length / chat_length.",
        "Ratio > 2.0x: generate is much longer (alignment suppresses chat).",
        "Ratio < 0.3x: chat is much longer (alignment adds content).",
        "",
    ]

    lines.append("## High divergence (ratio > 2.0x or < 0.3x)\n")
    lines.append("| Model | Origin | QID | Chat | Gen | Ratio | Direction |")
    lines.append("|---|---|---|---|---|---|---|")

    for row in cur.execute("""
        SELECT model, geopolitical_origin, question_id,
               length_chat, length_generate, chat_gen_ratio
        FROM analysis
        WHERE chat_gen_ratio IS NOT NULL
          AND (chat_gen_ratio > 2.0 OR chat_gen_ratio < 0.3)
          AND length_chat > 50
        ORDER BY ABS(chat_gen_ratio - 1) DESC
    """):
        direction = "gen >> chat" if row["chat_gen_ratio"] > 1 else "chat >> gen"
        lines.append(
            f"| {row['model']} | {row['geopolitical_origin'] or ''} | {row['question_id']} "
            f"| {row['length_chat']} | {row['length_generate']} "
            f"| **{row['chat_gen_ratio']}x** | {direction} |"
        )

    lines += ["", "## Think block presence", ""]
    lines.append("| Model | QID | Think-chat | Think-gen |")
    lines.append("|---|---|---|---|")
    for row in cur.execute("""
        SELECT model, question_id, think_chat, think_generate
        FROM analysis
        WHERE think_chat=1 OR think_generate=1
        ORDER BY model, question_id
    """):
        lines.append(
            f"| {row['model']} | {row['question_id']} "
            f"| {'✓' if row['think_chat'] else ''} "
            f"| {'✓' if row['think_generate'] else ''} |"
        )

    lines += ["", "## Model-level average ratios", ""]
    lines.append("| Model | Origin | Avg ratio | Direction pattern |")
    lines.append("|---|---|---|---|")
    for row in cur.execute("""
        SELECT model, geopolitical_origin, avg_chat_gen_ratio
        FROM v_model_summary
        WHERE avg_chat_gen_ratio IS NOT NULL
        ORDER BY avg_chat_gen_ratio DESC
    """):
        ratio = row["avg_chat_gen_ratio"]
        pattern = "gen >> chat (alignment suppresses)" if ratio > 1.2 else \
                  "chat >> gen (alignment adds content)" if ratio < 0.7 else \
                  "balanced"
        lines.append(f"| {row['model']} | {row['geopolitical_origin'] or ''} | {ratio} | {pattern} |")

    return "\n".join(lines)


# ── Report 3: Geopolitical comparison ─────────────────────────────────────────

def report_geopolitical(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    lines = [
        "# Report 3 — Four-Pole Geopolitical Comparison",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Compares model behavior across four geopolitical training origins:",
        "US (Meta/Google/OpenAI) | EU (Mistral) | China (DeepSeek/Alibaba/ZHIPU) | Gulf (UAE/Saudi)",
        "",
    ]

    POLES = {
        "US": ["US/Meta","US/Google","US/OpenAI","US/NVIDIA"],
        "EU": ["France/Mistral","France+US/Mistral+NVIDIA"],
        "China": ["China/DeepSeek","China/Alibaba","China/ZHIPU"],
        "Gulf": ["UAE/TII","Saudi/SDAIA"],
    }

    for pole, origins in POLES.items():
        placeholders = ",".join("?" * len(origins))
        rows = cur.execute(f"""
            SELECT model, geopolitical_origin,
                   questions_answered, ccp_voice_count, refusal_count,
                   avg_response_len, avg_chat_gen_ratio, think_block_count
            FROM v_model_summary
            WHERE geopolitical_origin IN ({placeholders})
            ORDER BY geopolitical_origin, model
        """, origins).fetchall()

        if not rows:
            continue

        lines += [f"## {pole}", ""]
        lines.append("| Model | CCP-V | Ref | AvgLen | Ratio | Think |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            cloud = " ☁" if "cloud" in r["model"] else ""
            lines.append(
                f"| {r['model']+cloud} | {r['ccp_voice_count']} | {r['refusal_count']} "
                f"| {r['avg_response_len'] or '—'} | {r['avg_chat_gen_ratio'] or '—'} "
                f"| {r['think_block_count']} |"
            )
        lines.append("")

    # Q03 lean analysis
    lines += ["## Q03 Political Lean — US Authoritarianism", ""]
    lines.append("| Model | Origin | Response type | Trump? | Jan6? | Biden? | Len |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in cur.execute("""
        SELECT r.model, a.geopolitical_origin, a.classification, a.length_chat,
               r.raw_response
        FROM responses r
        JOIN analysis a ON r.run_id=a.run_id AND r.model=a.model AND r.question_id=a.question_id
        WHERE r.question_id='Q03' AND r.endpoint='chat' AND r.success=1
        ORDER BY a.geopolitical_origin, r.model
    """):
        text = (row["raw_response"] or "").lower()
        trump = "✓" if "trump" in text else ""
        jan6  = "✓" if "january 6" in text or "jan 6" in text or "capitol" in text else ""
        biden = "✓" if "biden" in text else ""
        lines.append(
            f"| {row['model']} | {row['geopolitical_origin'] or ''} "
            f"| {row['classification']} | {trump} | {jan6} | {biden} | {row['length_chat']} |"
        )

    return "\n".join(lines)


# ── Report 4: Paired comparison ───────────────────────────────────────────────

def report_paired(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    lines = [
        "# Report 4 — Mechanism vs Named-Actor Paired Comparison",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Each pair: Qx (mechanism-first) vs Qxb (named-actor).",
        "Suppression activated: mechanism=ANALYST, named=CCP-VOICE.",
        "",
    ]

    lines.append("| Model | Origin | Pair | Mech class | Named class | Len ratio | Suppressed? |")
    lines.append("|---|---|---|---|---|---|---|")

    for row in cur.execute("""
        SELECT model, geopolitical_origin, family,
               mech_qid, named_qid,
               mech_class, named_class,
               mech_len, named_len, length_ratio,
               suppression_activated
        FROM v_suppression_pairs
        ORDER BY geopolitical_origin, model, mech_qid
    """):
        flag = "**YES**" if row["suppression_activated"] else ""
        lines.append(
            f"| {row['model']} | {row['geopolitical_origin'] or ''} "
            f"| {row['mech_qid']}/{row['named_qid']} "
            f"| {row['mech_class']} | {row['named_class']} "
            f"| {row['length_ratio']}x | {flag} |"
        )

    lines += ["", "## Summary: suppression triggers confirmed", ""]
    for row in cur.execute("""
        SELECT mech_qid, named_qid,
               COUNT(*) as total_models,
               SUM(suppression_activated) as suppressed_count,
               GROUP_CONCAT(CASE WHEN suppression_activated=1 THEN model END) as suppressed_models
        FROM v_suppression_pairs
        GROUP BY mech_qid, named_qid
        ORDER BY suppressed_count DESC
    """):
        if row["suppressed_count"]:
            lines.append(
                f"**{row['mech_qid']}/{row['named_qid']}**: "
                f"{row['suppressed_count']}/{row['total_models']} models suppressed — "
                f"{row['suppressed_models']}"
            )
            lines.append("")

    return "\n".join(lines)


# ── Report 5: Scale analysis ───────────────────────────────────────────────────

def report_scale(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    lines = [
        "# Report 5 — Scale Analysis: Local vs Cloud",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Compares same model family at different scales.",
        "",
    ]

    # DeepSeek scaling
    lines += ["## DeepSeek family — CCP suppression across scales", ""]
    lines.append("| Model | Scale | Q10b class | Q12b class | Q03 class | AvgLen |")
    lines.append("|---|---|---|---|---|---|")
    ds_models = ["deepseek-r1:7b","deepseek-r1:8b","deepseek-r1:14b",
                 "deepseek-r1:32b","deepseek-v3.2:cloud"]
    for model in ds_models:
        scale = {"deepseek-r1:7b":"7B","deepseek-r1:8b":"8B",
                 "deepseek-r1:14b":"14B","deepseek-r1:32b":"32B",
                 "deepseek-v3.2:cloud":"671B cloud"}.get(model,"?")
        def get_cls(qid):
            row = cur.execute(
                "SELECT classification FROM analysis WHERE model=? AND question_id=? LIMIT 1",
                (model, qid)
            ).fetchone()
            return row[0] if row else "—"
        avg = cur.execute(
            "SELECT avg_response_len FROM v_model_summary WHERE model=?", (model,)
        ).fetchone()
        avg_len = avg[0] if avg else "—"
        lines.append(
            f"| {model} | {scale} | {get_cls('Q10b')} | {get_cls('Q12b')} | {get_cls('Q03')} | {avg_len} |"
        )

    # Qwen scaling
    lines += ["", "## Qwen family — scale comparison", ""]
    lines.append("| Model | Scale | Q10b class | Q10b len | AvgLen | Ratio |")
    lines.append("|---|---|---|---|---|---|")
    qwen_models = ["qwen2.5:latest","qwen3.5:latest","qwen3.5:cloud"]
    scales = {"qwen2.5:latest":"7.6B","qwen3.5:latest":"9.7B","qwen3.5:cloud":"397B cloud"}
    for model in qwen_models:
        cls_row = cur.execute(
            "SELECT classification, length_chat FROM analysis WHERE model=? AND question_id='Q10b' LIMIT 1",
            (model,)
        ).fetchone()
        avg = cur.execute(
            "SELECT avg_response_len, avg_chat_gen_ratio FROM v_model_summary WHERE model=?", (model,)
        ).fetchone()
        c = cls_row[0] if cls_row else "—"
        l = cls_row[1] if cls_row else "—"
        a = avg[0] if avg else "—"
        r = avg[1] if avg else "—"
        lines.append(f"| {model} | {scales.get(model,'?')} | {c} | {l} | {a} | {r} |")

    # Mistral scaling
    lines += ["", "## Mistral family — EU models at scale", ""]
    lines.append("| Model | Scale | Q10b class | Q03 lean | AvgLen |")
    lines.append("|---|---|---|---|---|")
    for model, scale in [("mistral-nemo","12B"),("mixtral:8x7b","47B"),
                          ("mistral-large-3:675b-cloud","675B cloud")]:
        cls_row = cur.execute(
            "SELECT classification FROM analysis WHERE model=? AND question_id='Q10b' LIMIT 1",
            (model,)
        ).fetchone()
        q03_row = cur.execute(
            "SELECT raw_response FROM responses WHERE model=? AND question_id='Q03' AND endpoint='chat' LIMIT 1",
            (model,)
        ).fetchone()
        q03_text = (q03_row[0] or "").lower() if q03_row else ""
        lean = "right" if "trump" in q03_text and "biden" not in q03_text else \
               "balanced" if "trump" in q03_text and "biden" in q03_text else \
               "neutral/no-US"
        avg = cur.execute(
            "SELECT avg_response_len FROM v_model_summary WHERE model=?", (model,)
        ).fetchone()
        lines.append(
            f"| {model} | {scale} | {cls_row[0] if cls_row else '—'} | {lean} | {avg[0] if avg else '—'} |"
        )

    return "\n".join(lines)


# ── Report 6: Notable responses ───────────────────────────────────────────────

def report_notable(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    lines = [
        "# Report 6 — Notable Responses",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Full text of responses classified as CCP-VOICE, REFUSAL, EMPTY,",
        "or having extreme chat/gen divergence.",
        "",
    ]

    for row in cur.execute("""
        SELECT r.model, r.question_id, r.endpoint, r.raw_response,
               a.classification, a.chat_gen_ratio, a.notes,
               a.geopolitical_origin
        FROM responses r
        JOIN analysis a ON r.run_id=a.run_id AND r.model=a.model AND r.question_id=a.question_id
        WHERE a.is_notable=1 AND r.endpoint='chat' AND r.success=1
        ORDER BY a.geopolitical_origin, r.model, r.question_id
        LIMIT 100
    """):
        text = row["raw_response"] or ""
        lines += [
            f"---",
            f"## {row['model']} / {row['question_id']} / {row['endpoint']}",
            f"*{row['geopolitical_origin']} · {row['classification']} · {len(text)}c · ratio={row['chat_gen_ratio']}*",
            f"*Notes: {row['notes']}*",
            "",
            text[:2000] + ("..." if len(text) > 2000 else ""),
            "",
        ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

REPORTS = {
    "suppression":    ("01_suppression_matrix.md",      report_suppression),
    "divergence":     ("02_divergence_report.md",       report_divergence),
    "geopolitical":   ("03_geopolitical_comparison.md", report_geopolitical),
    "paired":         ("04_paired_comparison.md",       report_paired),
    "scale":          ("05_scale_analysis.md",          report_scale),
    "notable":        ("06_notable_responses.md",       report_notable),
}


def main():
    parser = argparse.ArgumentParser(description="Probe analysis reports")
    parser.add_argument("--db",     default=str(DB_PATH))
    parser.add_argument("--report", default="all",
                        help=f"Report to run: all | {' | '.join(REPORTS.keys())}")
    parser.add_argument("--out",    default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    conn    = connect(Path(args.db))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_run = REPORTS if args.report == "all" else {args.report: REPORTS[args.report]}

    for key, (filename, fn) in to_run.items():
        print(f"  Generating {filename}...", end="", flush=True)
        try:
            content = fn(conn)
            (out_dir / filename).write_text(content, encoding="utf-8")
            print(f" done ({len(content):,}c)")
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback; traceback.print_exc()

    conn.close()
    print(f"\nReports written to {out_dir}/")


if __name__ == "__main__":
    main()
