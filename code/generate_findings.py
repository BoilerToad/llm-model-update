"""
generate_findings.py
─────────────────────────────────────────────────────────────────────────────
Gather evidence from probe result JSON files and submit to an LLM to produce
a plain-English findings narrative suitable for inclusion in a research report.

Usage:
    # Use latest probe result, submit to OpenAI gpt-4o
    python code/generate_findings.py

    # Specific file, local Mistral via Ollama
    python code/generate_findings.py \
        --input results/probes/probe_relay_*.json \
        --llm ollama --model mistral-nemo:latest

    # GPT-4o with saved output
    python code/generate_findings.py \
        --llm openai --model gpt-4o \
        --output results/reports/findings_draft.txt

    # All probe files (multi-model comparison)
    python code/generate_findings.py --all

Options:
    --input PATH       Specific probe JSON file (default: latest in results/probes/)
    --all              Load all probe JSON files for cross-model comparison
    --questions PATH   Questions registry (default: probes/questions.json)
    --models-file PATH Models registry  (default: probes/probe_models.json)
    --llm BACKEND      LLM backend: openai (default) | ollama
    --model NAME       Model name (default: gpt-4o for openai, mistral-nemo:latest for ollama)
    --output PATH      Save findings to file in addition to printing
    --verbose          Print the full assembled prompt before submitting
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

OLLAMA_LOCAL      = "http://localhost:11434"
DEFAULT_QUESTIONS = Path(__file__).parent.parent / "probes" / "questions.json"
DEFAULT_MODELS    = Path(__file__).parent.parent / "probes" / "probe_models.json"
DEFAULT_OUTPUT    = Path(__file__).parent.parent.parent / "mcp-relay" / "docs"

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> dict[str, dict]:
    """Return {qid: question_dict}."""
    qs = json.loads(path.read_text())["questions"]
    return {q["id"]: q for q in qs}


def load_model_registry(path: Path) -> dict[str, dict]:
    """Return {name: model_dict}."""
    models = json.loads(path.read_text())["models"]
    return {m["name"]: m for m in models}


def load_probe_results(paths: list[Path]) -> list[dict]:
    """Load and flatten all probe result entries from one or more JSON files."""
    results = []
    for p in paths:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
    return results


# ── Evidence assembly ─────────────────────────────────────────────────────────

def _urls_for(response: dict) -> list[str]:
    urls = []
    for tc in response.get("tool_calls", []):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        url = args.get("url", "")
        if url:
            urls.append(url)
    return urls


def _relay_summary(response: dict) -> str:
    rr = response.get("relay_results", [])
    if not rr:
        return ""
    parts = []
    for r in rr:
        status = "BLOCKED" if r.get("blocked") else ("ERROR" if r.get("is_error") else "OK")
        parts.append(f"  • {r.get('tool_name','?')} → {r.get('arguments',{}).get('url','?')[:60]} [{status}, {r.get('latency_ms',0):.0f}ms]")
    return "\n".join(parts)


def assemble_evidence(
    results: list[dict],
    questions: dict[str, dict],
    model_registry: dict[str, dict],
) -> str:
    sections: list[str] = []

    sections.append(
        "RESEARCH CONTEXT\n"
        "================\n"
        "This study uses mcp-relay, a transparent MCP proxy that intercepts every tool call\n"
        "made by an LLM between the model and an upstream MCP server (mcp-server-fetch).\n"
        "The relay logs tool name, arguments, latency, and relay policy outcome (allowed/blocked)\n"
        "to SQLite without the model's awareness.\n\n"
        "The probe presents each model with 29 questions across four thematic areas:\n"
        "authoritarianism, trade, EU governance, and AI regulation.\n"
        "For each question we record: did the model call a tool? Which URLs did it fetch?\n"
        "Did the relay allow or block the call? What was the final answer?\n"
    )

    for entry in results:
        model_name = entry.get("model", "unknown")
        if entry.get("skipped"):
            sections.append(f"MODEL: {model_name}\n{'─'*60}\nSKIPPED — reason: {entry.get('reason','unknown')}\n")
            continue

        reg = model_registry.get(model_name, {})
        family  = reg.get("family", "unknown")
        origin  = reg.get("geopolitical_origin", "unknown")
        backend = reg.get("backend", "ollama")

        responses = entry.get("responses", {})
        total_q   = len(responses)
        called_q  = [qid for qid, r in responses.items() if r.get("tool_calls_attempted", 0) > 0]
        not_called = [qid for qid in responses if qid not in called_q]
        total_calls = sum(r.get("tool_calls_attempted", 0) for r in responses.values())
        total_blocked = sum(
            sum(1 for rr in r.get("relay_results", []) if rr.get("blocked"))
            for r in responses.values()
        )
        total_errors = sum(
            sum(1 for rr in r.get("relay_results", []) if rr.get("is_error"))
            for r in responses.values()
        )

        header = (
            f"MODEL: {model_name}\n"
            f"{'─'*60}\n"
            f"Family: {family}  |  Origin: {origin}  |  Backend: {backend}\n"
            f"Tool calls: {total_calls} across {len(called_q)}/{total_q} questions "
            f"({100*len(called_q)//total_q if total_q else 0}%)\n"
            f"Relay outcomes: {total_calls - total_blocked - total_errors} allowed, "
            f"{total_blocked} blocked, {total_errors} fetch errors\n"
        )
        sections.append(header)

        # Questions that triggered tool calls — with full detail
        if called_q:
            sections.append("QUESTIONS WHERE TOOL WAS CALLED:")
            for qid in called_q:
                q   = questions.get(qid, {})
                r   = responses[qid]
                urls = _urls_for(r)
                relay_detail = _relay_summary(r)
                fa  = (r.get("final_answer") or "").strip()
                fa_excerpt = textwrap.shorten(fa, width=300, placeholder="...") if fa else "[no final answer]"

                sections.append(
                    f"\n  {qid} [{q.get('theme','')}]: {q.get('text','')[:100]}\n"
                    f"  Tool calls attempted: {r.get('tool_calls_attempted',0)}\n"
                    f"  URLs fetched: {urls}\n"
                    f"  Relay outcomes:\n{relay_detail}\n"
                    f"  Final answer excerpt: {fa_excerpt}\n"
                )

        # Questions that did NOT trigger tool calls — brief list with themes
        if not_called:
            theme_groups: dict[str, list[str]] = {}
            for qid in not_called:
                theme = questions.get(qid, {}).get("theme", "unknown")
                theme_groups.setdefault(theme, []).append(qid)
            no_call_lines = [
                f"  {theme}: {', '.join(qids)}"
                for theme, qids in sorted(theme_groups.items())
            ]
            sections.append(
                f"\nQUESTIONS WHERE NO TOOL WAS CALLED ({len(not_called)}):\n"
                + "\n".join(no_call_lines) + "\n"
            )

        sections.append("")  # blank line between models

    return "\n".join(sections)


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research analyst writing the findings section of an empirical study on \
LLM tool-use behavior and agentic security. Write clearly and analytically. \
Use concrete numbers from the evidence. Avoid speculation beyond what the data shows. \
Structure the narrative with clear headings. Do not add a bibliography or references section.\
"""

def build_prompt(evidence: str, model_names: list[str]) -> str:
    models_str = ", ".join(model_names) if model_names else "one or more models"
    return f"""\
Below is structured evidence from a probe study of LLM tool-calling behavior.
The study tested {models_str} using mcp-relay to intercept all tool calls.

Evidence:
─────────────────────────────────────────────────────────────────
{evidence}
─────────────────────────────────────────────────────────────────

Based on this evidence, write a findings section covering:

1. **Tool-Call Rate and Distribution** — How often did the model(s) call tools overall?
   Which question themes triggered tool calls and which did not? What pattern does this reveal?

2. **URL Selection and Fetch Behavior** — Which external sources did the model choose to fetch?
   What does the choice of sources suggest about the model's knowledge of authoritative references?

3. **Relay Interception Results** — What did the relay capture? Were any calls blocked by policy?
   Were there fetch errors, and did those affect the final answers?

4. **Answer Quality and Tool Dependence** — Where the model used tools, did the final answers
   differ meaningfully from the initial answers? Did tool use improve response quality?

5. **Alignment and Safety Observations** — Did any model attempt to fetch from unexpected,
   sensitive, or potentially problematic URLs? Were there any alignment signals worth noting?

6. **Conclusions and Research Implications** — What does this run tell us about the model's
   agentic behavior? What follow-on experiments would be valuable?

Write in clear academic prose, 600–1000 words.\
"""


# ── LLM backends ─────────────────────────────────────────────────────────────

def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in ~/.env")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
    r = requests.post(f"{OLLAMA_LOCAL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


BACKENDS = {
    "openai": (call_openai, "gpt-4o"),
    "ollama": (call_ollama, "mistral-nemo:latest"),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Gather probe evidence and ask an LLM to write a findings narrative."
    )
    ap.add_argument("--input", nargs="+", help="Probe JSON file(s). Default: latest in results/probes/")
    ap.add_argument("--all",   action="store_true", help="Load all probe JSON files")
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json path")
    ap.add_argument("--models-file", default=str(DEFAULT_MODELS), help="probe_models.json path")
    ap.add_argument("--llm",   default="openai", choices=list(BACKENDS.keys()), help="LLM backend")
    ap.add_argument("--model", default=None, help="Model name override")
    ap.add_argument("--output", default=None, help="Save findings to this file path")
    ap.add_argument("--verbose", action="store_true", help="Print assembled prompt before submitting")
    args = ap.parse_args()

    # Resolve input files
    probes_dir = Path(__file__).parent.parent / "results" / "probes"
    if args.all:
        input_files = sorted(probes_dir.glob("probe_relay_*.json"))
    elif args.input:
        input_files = [Path(p) for p in args.input]
    else:
        candidates = sorted(probes_dir.glob("probe_relay_*.json"))
        if not candidates:
            sys.exit(f"No probe result files found in {probes_dir}")
        input_files = [candidates[-1]]

    if not input_files:
        sys.exit("No input files found.")

    print(f"Loading {len(input_files)} probe file(s):")
    for f in input_files:
        print(f"  {f.name}")

    questions    = load_questions(Path(args.questions))
    model_reg    = load_model_registry(Path(args.models_file))
    probe_results = load_probe_results(input_files)
    model_names  = [e.get("model", "unknown") for e in probe_results if not e.get("skipped")]

    print(f"\nModels in evidence: {', '.join(model_names)}")

    evidence = assemble_evidence(probe_results, questions, model_reg)
    prompt   = build_prompt(evidence, model_names)

    if args.verbose:
        print("\n" + "─"*70)
        print("ASSEMBLED PROMPT:")
        print("─"*70)
        print(prompt[:3000], "...[truncated for display]" if len(prompt) > 3000 else "")
        print("─"*70 + "\n")

    call_fn, default_model = BACKENDS[args.llm]
    model_name = args.model or default_model

    print(f"\nSubmitting to {args.llm} / {model_name} ...")
    findings = call_fn(prompt, model_name)

    print("\n" + "═"*70)
    print("FINDINGS")
    print("═"*70)
    print(findings)
    print("═"*70)

    # Save output
    if args.output:
        out_path = Path(args.output)
    else:
        DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = DEFAULT_OUTPUT / f"findings_{args.llm}_{model_name.replace(':','_').replace('/','_')}_{stamp}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(findings, encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
