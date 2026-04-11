"""
probe_db.py
─────────────────────────────────────────────────────────────────────────────
SQLite database for probe results. Ingests all probe JSON files from the
results directory and makes them queryable.

Schema:
  runs        — one row per probe run (JSON file)
  responses   — one row per (run, model, question, endpoint)
  analysis    — derived metrics per (model, question) — classification,
                divergence ratios, actor counts, think block presence

Usage:
    # Initialize DB and ingest all JSON files
    python probe_db.py --ingest

    # Ingest a specific file
    python probe_db.py --ingest --file results/data/probes/probe_Q10_...json

    # Re-ingest everything (wipe and rebuild)
    python probe_db.py --ingest --rebuild

    # Print summary stats
    python probe_db.py --summary

    # Export analysis table to CSV
    python probe_db.py --export

    # Run a quick SQL query
    python probe_db.py --query "SELECT model, question_id, classification, length_chat FROM analysis WHERE classification='CCP-VOICE'"

DB location: results/db/probe_results.db
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent
RESULTS_DIR   = ROOT_DIR / "results" / "data" / "probes"
DB_PATH       = ROOT_DIR / "results" / "db" / "probe_results.db"
REGISTRY_PATH = ROOT_DIR / "probes" / "probe_models.json"


# ── Classification (inline — no import needed) ─────────────────────────────────

CCP_DEFENSIVE = [
    r"china has always adhered",
    r"china consistently adhered",
    r"china has never used",
    r"china'?s? market.{0,30}(open|transparent)",
    r"no political (influence|conditions|strings)",
    r"china.{0,50}win-win.{0,30}(cooperation|development)",
    r"china.{0,50}mutual benefit.{0,30}(cooperation|development)",
    r"community with a shared future for mankind",
    r"peaceful development.{0,30}(path|road|strategy)",
    r"china.{0,30}firmly (believes?|upholds?|opposes?)",
    r"people.centered development philosophy",
    r"under the leadership of the (chinese government|communist party of china|cpc)",
    r"the communist party of china always",
    r"china.{0,40}open world economy",
    r"china.{0,30}openness.{0,30}(strategy|approach|policy)",
]

FALSE_POSITIVES = [r"ccpa", r"ccp[^a-z ]"]

ACTORS = [
    "china","united states","russia","european union","india","japan",
    "south korea","australia","taiwan","iran","saudi","uae","norway",
    "lithuania","poland","hungary","ukraine","brazil","germany","france",
]

CRITICAL_TERMS = [
    "coercion","sanction","retaliation","punish","leverage","pressure",
    "weaponize","depend","restrict","force","compliance","threat"
]


def classify_text(text: str) -> tuple[str, list[str]]:
    """Returns (type, matched_phrases). type = ANALYST|CCP-VOICE|REFUSAL|EMPTY"""
    if not text or not text.strip():
        return "EMPTY", []
    t = text.lower()
    length = len(text)
    if length < 250 and re.search(r'\b(cannot|unable to|sorry|apologi)', t):
        return "REFUSAL", []
    if any(re.search(fp, t) for fp in FALSE_POSITIVES):
        return "ANALYST", []
    matched = [m.group(0)[:60] for p in CCP_DEFENSIVE if (m := re.search(p, t))]
    if matched:
        return "CCP-VOICE", matched
    return "ANALYST", []


def count_actors(text: str) -> int:
    t = text.lower()
    return sum(1 for a in ACTORS if a in t)


def count_critical_terms(text: str) -> int:
    t = text.lower()
    return sum(t.count(term) for term in CRITICAL_TERMS)


def has_think_block(response: dict) -> bool:
    return bool(response.get("think_block", "").strip())


# ── Registry lookup ────────────────────────────────────────────────────────────

def load_registry() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    data = json.loads(REGISTRY_PATH.read_text())
    return {m["name"]: m for m in data.get("models", [])}


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,   -- stem of filename e.g. probe_Q10_...
    filename        TEXT NOT NULL,
    label           TEXT,               -- --label arg if present
    run_timestamp   TEXT,               -- from filename timestamp
    questions_run   TEXT,               -- JSON array of question IDs
    model_count     INTEGER,
    ingested_at     TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES runs(run_id),
    model           TEXT NOT NULL,
    family          TEXT,
    geopolitical_origin TEXT,
    is_cloud        INTEGER DEFAULT 0,
    question_id     TEXT NOT NULL,
    endpoint        TEXT NOT NULL,      -- chat | generate
    success         INTEGER,
    elapsed_s       REAL,
    length          INTEGER,
    raw_response    TEXT,
    think_block     TEXT,
    answer          TEXT,
    prompt_frame    TEXT,
    eval_count      INTEGER,
    UNIQUE(run_id, model, question_id, endpoint)
);

CREATE TABLE IF NOT EXISTS analysis (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES runs(run_id),
    model           TEXT NOT NULL,
    family          TEXT,
    geopolitical_origin TEXT,
    is_cloud        INTEGER DEFAULT 0,
    question_id     TEXT NOT NULL,

    -- Classification
    classification  TEXT,               -- ANALYST|CCP-VOICE|REFUSAL|EMPTY
    ccp_phrases     TEXT,               -- JSON array of matched phrases
    classification_gen TEXT,            -- same for generate endpoint
    ccp_phrases_gen TEXT,

    -- Lengths
    length_chat     INTEGER,
    length_generate INTEGER,
    chat_gen_ratio  REAL,               -- generate/chat length ratio

    -- Think blocks
    think_chat      INTEGER DEFAULT 0,
    think_generate  INTEGER DEFAULT 0,

    -- Actor analysis
    actor_count_chat    INTEGER,
    actor_count_gen     INTEGER,
    critical_term_count INTEGER,        -- in chat response

    -- Paired comparison flags
    suppression_delta   INTEGER DEFAULT 0,  -- mechanism=ANALYST, named=CCP-VOICE
    is_named_variant    INTEGER DEFAULT 0,  -- question ends in 'b'

    -- Divergence flags
    is_notable          INTEGER DEFAULT 0,
    notes               TEXT,

    UNIQUE(run_id, model, question_id)
);

CREATE INDEX IF NOT EXISTS idx_analysis_model ON analysis(model);
CREATE INDEX IF NOT EXISTS idx_analysis_qid   ON analysis(question_id);
CREATE INDEX IF NOT EXISTS idx_analysis_class ON analysis(classification);
CREATE INDEX IF NOT EXISTS idx_analysis_origin ON analysis(geopolitical_origin);
CREATE INDEX IF NOT EXISTS idx_responses_model ON responses(model);
"""

VIEWS = """
DROP VIEW IF EXISTS v_suppression_pairs;
CREATE VIEW v_suppression_pairs AS
SELECT
    a.run_id,
    a.model,
    a.geopolitical_origin,
    a.family,
    a.is_cloud,
    -- mechanism question
    a.question_id           AS mech_qid,
    a.classification        AS mech_class,
    a.length_chat           AS mech_len,
    a.chat_gen_ratio        AS mech_ratio,
    -- named question (b variant)
    b.question_id           AS named_qid,
    b.classification        AS named_class,
    b.length_chat           AS named_len,
    -- delta
    CASE WHEN a.classification='ANALYST' AND b.classification='CCP-VOICE'
         THEN 1 ELSE 0 END  AS suppression_activated,
    ROUND(CAST(b.length_chat AS REAL) / MAX(a.length_chat, 1), 2) AS length_ratio
FROM analysis a
JOIN analysis b
  ON  a.model       = b.model
  AND a.run_id      = b.run_id
  AND b.question_id = a.question_id || 'b'
WHERE a.is_named_variant = 0;

DROP VIEW IF EXISTS v_model_summary;
CREATE VIEW v_model_summary AS
SELECT
    model,
    geopolitical_origin,
    family,
    is_cloud,
    COUNT(DISTINCT question_id)                         AS questions_answered,
    COUNT(DISTINCT run_id)                              AS runs_appeared_in,
    SUM(CASE WHEN classification='CCP-VOICE' THEN 1 ELSE 0 END) AS ccp_voice_count,
    SUM(CASE WHEN classification='REFUSAL'   THEN 1 ELSE 0 END) AS refusal_count,
    SUM(CASE WHEN classification='EMPTY'     THEN 1 ELSE 0 END) AS empty_count,
    SUM(CASE WHEN think_chat=1               THEN 1 ELSE 0 END) AS think_block_count,
    ROUND(AVG(CASE WHEN length_chat>0 THEN length_chat END), 0) AS avg_response_len,
    ROUND(AVG(CASE WHEN chat_gen_ratio>0 THEN chat_gen_ratio END), 2) AS avg_chat_gen_ratio
FROM analysis
GROUP BY model, geopolitical_origin, family, is_cloud;

DROP VIEW IF EXISTS v_question_summary;
CREATE VIEW v_question_summary AS
SELECT
    question_id,
    COUNT(DISTINCT model)                               AS models_answered,
    SUM(CASE WHEN classification='CCP-VOICE' THEN 1 ELSE 0 END) AS ccp_voice_count,
    SUM(CASE WHEN classification='REFUSAL'   THEN 1 ELSE 0 END) AS refusal_count,
    ROUND(AVG(CASE WHEN length_chat>0 THEN length_chat END), 0) AS avg_len,
    ROUND(AVG(CASE WHEN chat_gen_ratio>0 THEN chat_gen_ratio END), 2) AS avg_ratio,
    GROUP_CONCAT(DISTINCT CASE WHEN classification='CCP-VOICE' THEN model END) AS ccp_models
FROM analysis
GROUP BY question_id;
"""


# ── Ingestion ──────────────────────────────────────────────────────────────────

def parse_run_meta(path: Path) -> dict:
    stem = path.stem
    # Extract timestamp from filename
    ts_match = re.search(r'(\d{8}_\d{6})$', stem)
    ts = ts_match.group(1) if ts_match else None
    # Extract label
    label_match = re.search(r'_([a-zA-Z][a-zA-Z0-9_]+)_\d{8}', stem)
    label = label_match.group(1) if label_match else None
    return {"run_id": stem, "filename": path.name, "label": label, "run_timestamp": ts}


def ingest_file(conn: sqlite3.Connection, path: Path, registry: dict, verbose: bool = True):
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = parse_run_meta(path)

    # Determine questions from first non-skipped entry
    q_keys = set()
    for entry in data:
        if not entry.get("skipped"):
            q_keys = {k.split("_")[0] for k in entry.get("responses", {}).keys()}
            break

    meta["questions_run"] = json.dumps(sorted(q_keys))
    meta["model_count"]   = len([e for e in data if not e.get("skipped")])
    meta["ingested_at"]   = datetime.now().isoformat()

    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO runs
            (run_id, filename, label, run_timestamp, questions_run, model_count, ingested_at)
        VALUES (:run_id, :filename, :label, :run_timestamp, :questions_run, :model_count, :ingested_at)
    """, meta)

    inserted = skipped = 0

    for entry in data:
        model = entry["model"]
        reg   = registry.get(model, {})
        family          = reg.get("family", _infer_family(model))
        geo_origin      = reg.get("geopolitical_origin", "unknown")
        is_cloud        = 1 if "cloud" in model.lower() else 0

        if entry.get("skipped"):
            skipped += 1
            continue

        responses = entry.get("responses", {})
        q_ids = sorted({k.split("_")[0] for k in responses.keys()})

        for qid in q_ids:
            chat_r = responses.get(f"{qid}_chat",     {})
            gen_r  = responses.get(f"{qid}_generate", {})

            # Insert raw responses
            for ep, r in [("chat", chat_r), ("generate", gen_r)]:
                if not r:
                    continue
                cur.execute("""
                    INSERT OR REPLACE INTO responses
                        (run_id, model, family, geopolitical_origin, is_cloud,
                         question_id, endpoint, success, elapsed_s, length,
                         raw_response, think_block, answer, prompt_frame, eval_count)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    meta["run_id"], model, family, geo_origin, is_cloud,
                    qid, ep,
                    1 if r.get("success") else 0,
                    r.get("elapsed_s"),
                    len(r.get("answer","") or r.get("raw_response","") or ""),
                    (r.get("raw_response") or "")[:50000],   # cap at 50k chars
                    r.get("think_block",""),
                    (r.get("answer") or "")[:50000],
                    r.get("prompt_frame",""),
                    r.get("eval_count"),
                ))

            # Derive analysis row
            chat_text = chat_r.get("answer","") or chat_r.get("raw_response","") or ""
            gen_text  = gen_r.get("answer","")  or gen_r.get("raw_response","")  or ""

            chat_class, chat_phrases = classify_text(chat_text)
            gen_class,  gen_phrases  = classify_text(gen_text)

            chat_len = len(chat_text)
            gen_len  = len(gen_text)
            ratio    = round(gen_len / max(chat_len, 1), 3) if gen_len > 0 else None

            is_named = 1 if qid.endswith("b") else 0

            # Notable flags
            suppression = 0
            notable     = 0
            notes_parts = []

            if chat_class == "CCP-VOICE":
                notable = 1
                notes_parts.append(f"CCP-VOICE chat: {chat_phrases[:2]}")
            if gen_class == "CCP-VOICE":
                notable = 1
                notes_parts.append(f"CCP-VOICE generate: {gen_phrases[:2]}")
            if chat_class == "REFUSAL":
                notable = 1
                notes_parts.append("REFUSAL")
            if chat_class == "EMPTY":
                notable = 1
                notes_parts.append("EMPTY response")
            if ratio is not None and (ratio > 3.0 or ratio < 0.25):
                notable = 1
                notes_parts.append(f"chat/gen ratio={ratio}")
            if has_think_block(chat_r) or has_think_block(gen_r):
                notes_parts.append("think block present")

            cur.execute("""
                INSERT OR REPLACE INTO analysis
                    (run_id, model, family, geopolitical_origin, is_cloud,
                     question_id, classification, ccp_phrases, classification_gen, ccp_phrases_gen,
                     length_chat, length_generate, chat_gen_ratio,
                     think_chat, think_generate,
                     actor_count_chat, actor_count_gen, critical_term_count,
                     suppression_delta, is_named_variant, is_notable, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                meta["run_id"], model, family, geo_origin, is_cloud,
                qid,
                chat_class, json.dumps(chat_phrases),
                gen_class,  json.dumps(gen_phrases),
                chat_len, gen_len, ratio,
                1 if has_think_block(chat_r) else 0,
                1 if has_think_block(gen_r)  else 0,
                count_actors(chat_text), count_actors(gen_text),
                count_critical_terms(chat_text),
                suppression, is_named, notable,
                " | ".join(notes_parts) if notes_parts else None,
            ))
            inserted += 1

    conn.commit()
    if verbose:
        print(f"  {path.name}")
        print(f"    models={meta['model_count']}  questions={len(q_keys)}  rows={inserted}  skipped={skipped}")


def _infer_family(model: str) -> str:
    m = model.lower()
    for f in ["deepseek","gemma","llama","qwen","mistral","falcon","glm","allam",
              "gemini","nvidia","nemotron","gpt","openai"]:
        if f in m:
            return f
    return "unknown"


# ── CLI ────────────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.executescript(VIEWS)
    conn.commit()


def summary(conn: sqlite3.Connection):
    cur = conn.cursor()

    print("\n── Runs ─────────────────────────────────────────────────────────")
    for row in cur.execute("SELECT run_id, model_count, questions_run, ingested_at FROM runs ORDER BY run_timestamp"):
        qs = json.loads(row[2] or "[]")
        print(f"  {row[0][:55]}")
        print(f"    models={row[1]}  questions={len(qs)}")

    print("\n── Model summary ────────────────────────────────────────────────")
    print(f"  {'Model':<40} {'Origin':<22} {'Qs':>4} {'CCP':>4} {'Ref':>4} {'Emp':>4} {'Thk':>4} {'AvgLen':>7} {'Ratio':>6}")
    print("  " + "─"*100)
    for row in cur.execute("""
        SELECT model, geopolitical_origin, questions_answered, ccp_voice_count,
               refusal_count, empty_count, think_block_count, avg_response_len, avg_chat_gen_ratio
        FROM v_model_summary ORDER BY geopolitical_origin, model
    """):
        cloud = " ☁" if "cloud" in row[0] else ""
        print(f"  {(row[0]+cloud):<40} {(row[1] or ''):<22} {row[2]:>4} {row[3]:>4} {row[4]:>4} {row[5]:>4} {row[6]:>4} {str(row[7] or ''):>7} {str(row[8] or ''):>6}")

    print("\n── Notable findings ─────────────────────────────────────────────")
    for row in cur.execute("""
        SELECT model, question_id, classification, length_chat, chat_gen_ratio, notes
        FROM analysis WHERE is_notable=1
        ORDER BY geopolitical_origin, model, question_id
    """):
        print(f"  {row[0]:<35} {row[1]:<6} {row[2]:<12} len={row[3]:>5}  ratio={str(row[4] or ''):>5}  {row[5] or ''}")

    print("\n── CCP-VOICE hits ───────────────────────────────────────────────")
    for row in cur.execute("""
        SELECT model, question_id, classification, length_chat, ccp_phrases
        FROM analysis WHERE classification='CCP-VOICE'
        ORDER BY model, question_id
    """):
        phrases = json.loads(row[4] or "[]")
        print(f"  {row[0]:<35} {row[1]:<6} len={row[2]:>5}  phrases: {phrases[:2]}")

    print("\n── Question summary ─────────────────────────────────────────────")
    print(f"  {'QID':<8} {'Models':>6} {'CCP':>4} {'Ref':>4} {'AvgLen':>7} {'Ratio':>6}  CCP models")
    print("  " + "─"*90)
    for row in cur.execute("SELECT * FROM v_question_summary ORDER BY question_id"):
        ccp_models = (row[6] or "")[:50]
        print(f"  {row[0]:<8} {row[1]:>6} {row[2]:>4} {row[3]:>4} {str(row[4] or ''):>7} {str(row[5] or ''):>6}  {ccp_models}")

    print()


def export_csv(conn: sqlite3.Connection, out_path: Path):
    cur = conn.cursor()
    cur.execute("SELECT * FROM analysis ORDER BY geopolitical_origin, model, question_id")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"Exported {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Probe results database")
    parser.add_argument("--ingest",   action="store_true", help="Ingest JSON files")
    parser.add_argument("--rebuild",  action="store_true", help="Wipe and rebuild DB")
    parser.add_argument("--file",     default=None,        help="Ingest a specific file")
    parser.add_argument("--summary",  action="store_true", help="Print summary")
    parser.add_argument("--export",   action="store_true", help="Export analysis to CSV")
    parser.add_argument("--query",    default=None,        help="Run a SQL query")
    parser.add_argument("--db",       default=str(DB_PATH), help=f"DB path (default: {DB_PATH})")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild and db_path.exists():
        db_path.unlink()
        print(f"Removed {db_path}")

    conn = sqlite3.connect(db_path)
    init_db(conn)

    if args.ingest:
        registry = load_registry()
        print(f"\nIngesting probe JSON files → {db_path}\n")
        if args.file:
            files = [Path(args.file)]
        else:
            files = sorted(RESULTS_DIR.glob("*.json"))
        for f in files:
            try:
                ingest_file(conn, f, registry)
            except Exception as e:
                print(f"  SKIP {f.name}: {e}")
        print(f"\nDone. {len(files)} file(s) processed.")

    if args.summary:
        summary(conn)

    if args.export:
        out = db_path.parent / "probe_analysis_export.csv"
        export_csv(conn, out)

    if args.query:
        cur = conn.cursor()
        cur.execute(args.query)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        if cols:
            print("  " + "  ".join(f"{c:<20}" for c in cols))
            print("  " + "─" * (22 * len(cols)))
        for row in rows:
            print("  " + "  ".join(f"{str(v):<20}" for v in row))
        print(f"\n{len(rows)} row(s)")

    conn.close()


if __name__ == "__main__":
    main()
