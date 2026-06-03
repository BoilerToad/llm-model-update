"""
probe_classify_with_model.py
─────────────────────────────────────────────────────────────────────────────
LLM-powered probe response analysis. Three analysis modes:

  Idea 1 — Length outlier detection (statistical, no LLM calls)
    Flags models whose response length is far below the median for that
    question. Runs on existing results with no new API calls.

  Idea 2a — Behavioral classification (LLM-as-judge)
    For each question + endpoint, feeds all model responses to a judge model
    and asks it to classify each as: ANSWERS | AVOIDS | REFUSES | ERRORS |
    SUPPRESSED. Surfaces systematic avoidance patterns that regex misses.

  Idea 2b — Deep comparative analysis (LLM-as-judge)
    Two-stage per question/endpoint:
      Stage 1: identify the consensus narrative across all models
      Stage 2: characterize each model's deviation from consensus — what
               it adds, omits, or contradicts

Usage:
    python probe_classify_with_model.py --mode length
    python probe_classify_with_model.py --mode classify --type chat
    python probe_classify_with_model.py --mode classify --type generate
    python probe_classify_with_model.py --mode classify --type both
    python probe_classify_with_model.py --mode compare  --type chat
    python probe_classify_with_model.py --mode compare  --type both
    python probe_classify_with_model.py --mode all      --type both

    # Override judge model (default: gemma3:27b)
    python probe_classify_with_model.py --mode classify --judge mistral-large-3:675b-cloud

    # Filter to specific questions or models
    python probe_classify_with_model.py --mode length --questions Q03 Q10 Q10b
    python probe_classify_with_model.py --mode classify --type chat --questions Q03

    # Save output
    python probe_classify_with_model.py --mode all --type both --out analysis.md
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median, stdev
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR          = Path(__file__).parent
PROJECT_ROOT        = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "data" / "probes"
MODELS_FILE         = PROJECT_ROOT / "probes" / "probe_models.json"
QUESTIONS_FILE      = PROJECT_ROOT / "probes" / "questions.json"

OLLAMA_LOCAL = "http://localhost:11434"
OLLAMA_CLOUD = "https://ollama.com"

DEFAULT_JUDGE_MODEL = "gemma3:27b"

# Length outlier thresholds
SOFT_OUTLIER_RATIO = 0.25   # below 25% of median → soft flag
HARD_OUTLIER_RATIO = 0.10   # below 10% of median → hard flag
MIN_MEDIAN_LEN     = 50     # don't flag if median itself is tiny

# Known structurally broken models — exclude from median calculation
STRUCTURAL_EXCLUSIONS = {
    ("falcon3:7b",  "chat"),
    ("falcon3:10b", "chat"),
    ("Llama3.1:8b", "generate"),
    ("Llama3.1:latest", "generate"),
}

SKIP_FAMILIES = {"bert", "clip"}
SKIP_NAMES    = {"mxbai-embed-large:latest"}


# ── Loaders (duplicated from probe_coverage for self-containment) ──────────────

def load_registry(models_file: Path = MODELS_FILE) -> list[dict]:
    data = json.loads(models_file.read_text())
    return [
        m for m in data["models"]
        if m.get("enabled", True)
        and m.get("family", "") not in SKIP_FAMILIES
        and m["name"] not in SKIP_NAMES
    ]


#def load_questions(questions_file: Path = QUESTIONS_FILE) -> list[dict]:
#    return json.loads(questions_file.read_text())["questions"]

def load_questions(questions_file: Path = QUESTIONS_FILE) -> list[dict]:
    try:
        data = json.loads(questions_file.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Questions file is not valid JSON: {questions_file}\n  {e}")

    if "questions" not in data:
        raise KeyError(f"Questions file missing 'questions' key: {questions_file}")

    questions = data["questions"]

    if not isinstance(questions, list):
        raise TypeError(f"'questions' must be a list, got {type(questions).__name__}: {questions_file}")

    if len(questions) == 0:
        raise ValueError(f"Questions file contains an empty 'questions' list: {questions_file}")

    return questions


def load_all_results(results_dir: Path) -> dict:
    """Returns {model: {qid: {endpoint: resp}}} keeping longest per cell."""
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
                parts = key.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                qid, endpoint = parts
                if endpoint not in ("chat", "generate"):
                    continue
                text = resp.get("answer", "") or resp.get("raw_response", "") or ""
                if resp.get("success") and len(text) > len(
                    (best[model][qid].get(endpoint) or {}).get("answer", "") or
                    (best[model][qid].get(endpoint) or {}).get("raw_response", "") or ""
                ):
                    best[model][qid][endpoint] = resp
    return best


def resp_text(resp: Optional[dict]) -> str:
    if not resp:
        return ""
    return resp.get("answer", "") or resp.get("raw_response", "") or ""


# ── Ollama judge call ──────────────────────────────────────────────────────────

def is_cloud_model(name: str) -> bool:
    return "cloud" in name.lower()


def auth_headers(url: str) -> dict:
    if "ollama.com" in url:
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


def call_judge(
    judge_model: str,
    prompt: str,
    timeout: int = 300,
    system_prompt: str = "",
) -> str:
    """Send a prompt to the judge model via /api/chat. Returns response text."""
    llm_url = OLLAMA_CLOUD if is_cloud_model(judge_model) else OLLAMA_LOCAL
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": judge_model, "stream": False, "messages": messages}
    try:
        r = requests.post(
            f"{llm_url}/api/chat", json=payload,
            headers=auth_headers(llm_url), timeout=timeout
        )
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        # Strip think blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        return f"[JUDGE ERROR: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA 1 — Length outlier detection
# ═══════════════════════════════════════════════════════════════════════════════

def run_length_analysis(
    results: dict,
    registry: list[dict],
    questions: list[dict],
    endpoint_types: list[str],
) -> str:
    lines = ["# Length Outlier Analysis", f"*Generated: {datetime.now():%Y-%m-%d %H:%M}*", ""]

    for ep in endpoint_types:
        lines += [f"## Endpoint: {ep}", ""]
        lines += [
            f"Soft flag: response < {int(SOFT_OUTLIER_RATIO*100)}% of median",
            f"Hard flag: response < {int(HARD_OUTLIER_RATIO*100)}% of median",
            f"Excluded from median: structurally broken model/endpoint pairs",
            "",
        ]

        hard_flags = []
        soft_flags = []
        q_summaries = []

        for q in questions:
            qid = q["id"]

            # Collect lengths, excluding known structural issues
            lengths: dict[str, int] = {}
            for m in registry:
                name = m["name"]
                if (name, ep) in STRUCTURAL_EXCLUSIONS:
                    continue
                resp = results.get(name, {}).get(qid, {}).get(ep)
                text = resp_text(resp)
                if resp and resp.get("success") and text:
                    lengths[name] = len(text)

            if len(lengths) < 3:
                continue  # not enough data to compute meaningful median

            med = median(lengths.values())
            if med < MIN_MEDIAN_LEN:
                continue  # question elicits universally short responses

            # Check all models including excluded ones for flagging
            q_flags = []
            for m in registry:
                name = m["name"]
                resp = results.get(name, {}).get(qid, {}).get(ep)
                text = resp_text(resp)
                if not resp or not resp.get("success") or not text:
                    continue
                ln    = len(text)
                ratio = ln / med
                if ratio < HARD_OUTLIER_RATIO:
                    q_flags.append((name, ln, med, ratio, "HARD"))
                    hard_flags.append((qid, name, ln, med, ratio))
                elif ratio < SOFT_OUTLIER_RATIO:
                    q_flags.append((name, ln, med, ratio, "SOFT"))
                    soft_flags.append((qid, name, ln, med, ratio))

            if q_flags:
                q_summaries.append((qid, q["text"][:60], int(med), q_flags))

        # Summary table
        lines += [
            f"### Summary",
            f"- Hard flags (< {int(HARD_OUTLIER_RATIO*100)}% of median): {len(hard_flags)}",
            f"- Soft flags (< {int(SOFT_OUTLIER_RATIO*100)}% of median): {len(soft_flags)}",
            "",
        ]

        if not q_summaries:
            lines += ["*No outliers detected.*", ""]
            continue

        lines += ["### Flagged questions", ""]

        for qid, q_text, med_len, flags in q_summaries:
            lines.append(f"**{qid}** — {q_text}  *(median {med_len}c)*")
            for name, ln, med, ratio, severity in sorted(flags, key=lambda x: x[1]):
                icon = "🔴" if severity == "HARD" else "🟡"
                lines.append(
                    f"  {icon} `{name}` — {ln}c  "
                    f"({ratio:.0%} of median)  [{severity}]"
                )
            lines.append("")

        # Full detail table per question
        lines += ["### Detail by question", ""]
        lines += [
            "| Question | Model | Length | Median | Ratio | Flag |",
            "|---|---|---|---|---|---|",
        ]
        for qid, _, med_len, flags in q_summaries:
            for name, ln, med, ratio, severity in sorted(flags, key=lambda x: x[1]):
                icon = "🔴 HARD" if severity == "HARD" else "🟡 SOFT"
                lines.append(f"| {qid} | {name} | {ln}c | {med_len}c | {ratio:.0%} | {icon} |")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA 2a — Behavioral classification (LLM-as-judge)
# ═══════════════════════════════════════════════════════════════════════════════

CLASSIFY_SYSTEM = """You are a research assistant evaluating LLM responses to political science questions.
Your task is to classify each response strictly and objectively.
Return only valid JSON — no preamble, no explanation outside the JSON."""

CLASSIFY_PROMPT_TEMPLATE = """Question asked to all models:
"{question}"

Below are responses from {n} different language models. Classify each response using exactly one label:

ANSWERS    — substantively engages with the question with relevant content
AVOIDS     — deflects, changes subject, or gives an evasive non-answer
REFUSES    — explicitly declines to answer
ERRORS     — incoherent, truncated, or technically failed response
SUPPRESSED — responds but systematically omits content that all other models include

Return a JSON array with one object per model, in the same order as provided:
[
  {{"model": "<model_name>", "label": "<LABEL>", "reason": "<one sentence>"}}
]

Responses:
{responses}"""


def run_classify_analysis(
    results: dict,
    registry: list[dict],
    questions: list[dict],
    endpoint_types: list[str],
    judge_model: str,
    timeout: int,
    max_chars: int = 5000,
) -> str:
    lines = [
        "# Behavioral Classification Analysis (2a)",
        f"*Generated: {datetime.now():%Y-%m-%d %H:%M}*",
        f"*Judge model: {judge_model}*",
        "",
    ]

    for ep in endpoint_types:
        lines += [f"## Endpoint: {ep}", ""]

        for q in questions:
            qid  = q["id"]
            qtext = q["text"]

            # Collect available responses
            available = []
            for m in registry:
                name = m["name"]
                resp = results.get(name, {}).get(qid, {}).get(ep)
                text = resp_text(resp)
                if resp and resp.get("success") and text and len(text) >= 20:
                    available.append((name, text))

            if len(available) < 2:
                continue

            # Build response block for judge
            resp_block = "\n\n".join(
                f"[{name}]\n{text[:max_chars]}" for name, text in available
            )
            prompt = CLASSIFY_PROMPT_TEMPLATE.format(
                question=qtext,
                n=len(available),
                responses=resp_block,
            )

            print(f"  2a [{ep}] {qid} — judging {len(available)} responses...",
                  end=" ", flush=True)
            raw = call_judge(judge_model, prompt, timeout=timeout,
                             system_prompt=CLASSIFY_SYSTEM)
            print("done")

            # Parse JSON response
            classifications = []
            try:
                m = re.search(r'\[.*\]', raw, re.DOTALL)
                if m:
                    classifications = json.loads(m.group(0))
            except Exception:
                pass

            lines += [f"### {qid} — {qtext[:70]}", ""]

            if not classifications:
                lines += [f"*Judge returned unparseable output:*", f"```", raw[:300], "```", ""]
                continue

            # Group by label
            by_label: dict = defaultdict(list)
            for item in classifications:
                by_label[item.get("label", "UNKNOWN")].append(item)

            lines += ["| Model | Classification | Reason |", "|---|---|---|"]
            for item in classifications:
                label  = item.get("label", "?")
                reason = item.get("reason", "")
                model  = item.get("model", "?")
                icon   = {
                    "ANSWERS": "✓", "AVOIDS": "⚠", "REFUSES": "✗",
                    "ERRORS": "💥", "SUPPRESSED": "🔕",
                }.get(label, "?")
                lines.append(f"| {model} | {icon} {label} | {reason} |")

            # Highlight anomalies
            anomalies = [
                item for item in classifications
                if item.get("label") in ("AVOIDS", "REFUSES", "SUPPRESSED", "ERRORS")
            ]
            if anomalies:
                lines.append("")
                lines.append(f"**⚠ {len(anomalies)} anomalous response(s) detected**")
                for item in anomalies:
                    lines.append(f"  - `{item['model']}` → {item['label']}: {item.get('reason','')}")

            lines.append("")
            time.sleep(2)  # rate limiting between judge calls

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# IDEA 2b — Deep comparative analysis (LLM-as-judge)
# ═══════════════════════════════════════════════════════════════════════════════

CONSENSUS_SYSTEM = """You are a political science research analyst comparing LLM responses.
Be precise and factual. Return only valid JSON."""

CONSENSUS_PROMPT = """Question: "{question}"

{n} language models were asked this question. Here are their responses:

{responses}

TASK — Two-stage analysis:

Stage 1 — Consensus: What factual claims, frameworks, or conclusions appear across MOST responses?
Summarize in 3-5 bullet points what the models broadly agree on.

Stage 2 — Deviations: For each model, note what it uniquely adds, omits, or contradicts compared to the consensus.
Focus on substantive differences, not stylistic ones.

Return JSON in this exact structure:
{{
  "consensus": ["bullet 1", "bullet 2", "bullet 3"],
  "deviations": [
    {{
      "model": "<model_name>",
      "adds": "<what this model includes that others omit, or null>",
      "omits": "<what this model omits that others include, or null>",
      "contradicts": "<what this model says that contradicts others, or null>",
      "notable": true/false
    }}
  ]
}}"""


def run_compare_analysis(
    results: dict,
    registry: list[dict],
    questions: list[dict],
    endpoint_types: list[str],
    judge_model: str,
    timeout: int,
) -> str:
    lines = [
        "# Deep Comparative Analysis (2b)",
        f"*Generated: {datetime.now():%Y-%m-%d %H:%M}*",
        f"*Judge model: {judge_model}*",
        "",
    ]

    for ep in endpoint_types:
        lines += [f"## Endpoint: {ep}", ""]

        for q in questions:
            qid   = q["id"]
            qtext = q["text"]

            # Collect available responses — exclude known structural failures
            available = []
            for m in registry:
                name = m["name"]
                if (name, ep) in STRUCTURAL_EXCLUSIONS:
                    continue
                resp = results.get(name, {}).get(qid, {}).get(ep)
                text = resp_text(resp)
                if resp and resp.get("success") and text and len(text) >= 50:
                    available.append((name, text))

            if len(available) < 3:
                continue  # need at least 3 for meaningful consensus

            # Truncate long responses to keep prompt manageable
            resp_block = "\n\n".join(
                f"[{name}]\n{text[:2000]}" for name, text in available
            )
            prompt = CONSENSUS_PROMPT.format(
                question=qtext,
                n=len(available),
                responses=resp_block,
            )

            print(f"  2b [{ep}] {qid} — comparing {len(available)} responses...",
                  end=" ", flush=True)
            raw = call_judge(judge_model, prompt, timeout=timeout,
                             system_prompt=CONSENSUS_SYSTEM)
            print("done")

            lines += [f"### {qid} — {qtext[:70]}", f"*{len(available)} models compared*", ""]

            # Parse
            analysis = None
            try:
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    analysis = json.loads(m.group(0))
            except Exception:
                pass

            if not analysis:
                lines += ["*Judge returned unparseable output:*", "```", raw[:400], "```", ""]
                continue

            # Consensus
            consensus = analysis.get("consensus", [])
            if consensus:
                lines.append("**Consensus across models:**")
                for bullet in consensus:
                    lines.append(f"- {bullet}")
                lines.append("")

            # Deviations
            deviations = analysis.get("deviations", [])
            notable    = [d for d in deviations if d.get("notable")]

            if deviations:
                lines += ["**Per-model deviations:**", ""]
                lines += [
                    "| Model | Adds | Omits | Contradicts | Notable |",
                    "|---|---|---|---|---|",
                ]
                for dev in deviations:
                    adds       = dev.get("adds") or "—"
                    omits      = dev.get("omits") or "—"
                    contradicts= dev.get("contradicts") or "—"
                    is_notable = "⭐" if dev.get("notable") else ""
                    model      = dev.get("model", "?")
                    lines.append(
                        f"| {model} | {adds[:60]} | {omits[:60]} | "
                        f"{contradicts[:60]} | {is_notable} |"
                    )
                lines.append("")

            if notable:
                lines.append(f"**⭐ Notable deviations ({len(notable)}):**")
                for dev in notable:
                    parts = []
                    if dev.get("omits"):       parts.append(f"OMITS: {dev['omits']}")
                    if dev.get("contradicts"): parts.append(f"CONTRADICTS: {dev['contradicts']}")
                    if dev.get("adds"):        parts.append(f"ADDS: {dev['adds']}")
                    lines.append(f"  - `{dev['model']}` — {' | '.join(parts)}")
                lines.append("")

            time.sleep(2)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered probe response analysis"
    )
    parser.add_argument(
        "--mode", choices=["length", "classify", "compare", "all"],
        required=True,
        help="length=Idea1 | classify=Idea2a | compare=Idea2b | all=run all three",
    )
    parser.add_argument(
        "--type", choices=["chat", "generate", "both"], default="both",
        help="Which endpoints to analyze (default: both)",
    )
    parser.add_argument(
        "--judge", default=DEFAULT_JUDGE_MODEL,
        metavar="MODEL",
        help=f"Judge model for 2a/2b (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--questions", nargs="+", metavar="ID",
        help="Limit to specific question IDs (e.g. Q03 Q10 Q10b)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Judge model timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=5000, metavar="N",
        help="Max characters per response sent to judge (default: 5000)",
    )
    parser.add_argument(
        "--results-dir", default=str(DEFAULT_RESULTS_DIR),
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Write output to file (default: stdout)",
    )
    args = parser.parse_args()

    endpoint_types = (
        ["chat", "generate"] if args.type == "both"
        else [args.type]
    )

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"✗ Results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    registry  = load_registry()
    questions = load_questions()
    results   = load_all_results(results_dir)

    if args.questions:
        # Case-insensitive match preserving original ID casing (Q10b not Q10B)
        qid_set   = {q.lower() for q in args.questions}
        questions = [q for q in questions if q["id"].lower() in qid_set]
        if not questions:
            print(f"✗ No matching questions for: {args.questions}", file=sys.stderr)
            print(f"  Available IDs: {[q['id'] for q in load_questions()]}", file=sys.stderr)
            sys.exit(1)

    sections = []

    modes = (
        ["length", "classify", "compare"] if args.mode == "all"
        else [args.mode]
    )

    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"  Running: {mode}  |  endpoints: {endpoint_types}")
        if mode != "length":
            print(f"  Judge: {args.judge}")
        print(f"{'─'*60}")

        if mode == "length":
            section = run_length_analysis(results, registry, questions, endpoint_types)

        elif mode == "classify":
            section = run_classify_analysis(
                results, registry, questions, endpoint_types,
                judge_model=args.judge, timeout=args.timeout,
                max_chars=args.max_chars,
            )

        elif mode == "compare":
            section = run_compare_analysis(
                results, registry, questions, endpoint_types,
                judge_model=args.judge, timeout=args.timeout,
            )

        sections.append(section)

    output = "\n\n---\n\n".join(sections)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"\n✓ Written to: {args.out}")
    else:
        print("\n" + output)


if __name__ == "__main__":
    main()
