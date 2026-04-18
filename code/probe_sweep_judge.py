#!/usr/bin/env python3
"""
probe_sweep_judge.py
────────────────────
Reads a sweep results file produced by probe_endpoint_sweep.py and submits
each model's collected response texts to a cloud judge model for semantic
repeatability and quality analysis.

The judge (default: mistral-large-3:675b-cloud) evaluates each set of
responses and returns a structured assessment per model per endpoint.

What the judge evaluates:
  - Semantic consistency  : do all N responses convey the same core meaning?
  - Factual accuracy      : are the responses factually correct?
  - Notable variance      : which responses deviate most from the majority?
  - Repeatability verdict : CONSISTENT / VARIABLE / INCONSISTENT
  - Quality verdict       : GOOD / ACCEPTABLE / POOR

Output: JSON file saved to results/data/judges/judge_TIMESTAMP.json

Usage:
    python probe_sweep_judge.py --sweep results/data/sweeps/endpoint_sweep_20260404_105505.json
    python probe_sweep_judge.py --sweep results/data/sweeps/endpoint_sweep_20260404_105505.json \\
        --judge mistral-large-3:675b-cloud \\
        --endpoints chat gen_plain \\
        --timeout 120 \\
        --out results/data/judges/my_judgement.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

SCRIPT_DIR   = Path(__file__).parent
ROOT_DIR     = SCRIPT_DIR.parent
OLLAMA_CLOUD = "https://ollama.com"
OLLAMA_LOCAL = "http://localhost:11434"

DEFAULT_JUDGE   = "mistral-large-3:675b-cloud"
DEFAULT_TIMEOUT = 120

# Endpoints in sweep results file that contain response text arrays
SWEEP_ENDPOINTS = {
    "chat":      "chat",
    "gen_raw":   "gen_raw",
    "gen_plain": "gen_plain",
}

JUDGE_SYSTEM_PROMPT = """\
You are a research assistant evaluating the output consistency and quality of \
language model responses. You will be given a set of responses produced by a \
single model answering the same question repeatedly. Your task is to assess \
semantic consistency and factual quality across the response set.

Respond ONLY with a valid JSON object — no preamble, no markdown fences, \
no additional text. The JSON must exactly match the schema provided."""

JUDGE_PROMPT_TEMPLATE = """\
A language model was asked the following question {n} times:

QUESTION: {question}

The {n} responses are listed below, numbered 1 to {n}:

{responses_block}

Evaluate this response set and return a JSON object with exactly these fields:

{{
  "semantic_consistency": "CONSISTENT" | "VARIABLE" | "INCONSISTENT",
  "consistency_explanation": "<one sentence explaining your verdict>",
  "factual_accuracy": "GOOD" | "ACCEPTABLE" | "POOR",
  "accuracy_explanation": "<one sentence on factual quality>",
  "most_deviant_indices": [<list of 0-based response indices that deviate most, or empty list>],
  "deviance_explanation": "<one sentence on what makes those responses deviant, or 'None'>",
  "repeatability_verdict": "CONSISTENT" | "VARIABLE" | "INCONSISTENT",
  "quality_verdict": "GOOD" | "ACCEPTABLE" | "POOR",
  "summary": "<two sentences summarising the overall assessment>"
}}

Definitions:
- CONSISTENT: all responses convey the same core meaning with only surface wording differences
- VARIABLE: responses share the main point but differ in scope, emphasis, or secondary facts
- INCONSISTENT: responses contradict each other or wander significantly in meaning
- GOOD: factually accurate and well-formed
- ACCEPTABLE: mostly accurate with minor gaps or imprecisions
- POOR: contains factual errors, hallucinations, or incoherent content"""


# ── Cloud request ──────────────────────────────────────────────────────────────

def _cloud_url_for(model: str) -> str:
    return OLLAMA_CLOUD if "cloud" in model.lower() else OLLAMA_LOCAL


def _auth_headers(url: str) -> dict:
    if url == OLLAMA_CLOUD:
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


def call_judge(
    model: str,
    system: str,
    user: str,
    timeout: int,
) -> tuple[bool, str, float]:
    """
    Call the judge model via Ollama chat API.
    Returns (success, response_text, elapsed_s).
    """
    url     = _cloud_url_for(model)
    payload = {
        "model":  model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{url}/api/chat",
            json=payload,
            headers=_auth_headers(url),
            timeout=timeout,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}", elapsed
        content = r.json().get("message", {}).get("content", "")
        return True, content, elapsed
    except requests.exceptions.Timeout:
        return False, f"TIMEOUT after {timeout}s", round(time.perf_counter() - t0, 2)
    except Exception as e:
        return False, str(e)[:200], round(time.perf_counter() - t0, 2)


def parse_judge_response(text: str) -> dict | None:
    """
    Parse the judge's JSON response. Strips markdown fences if present.
    Returns None if parsing fails.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Drop first and last fence lines
        inner = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(inner).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


# ── Sweep file loading ─────────────────────────────────────────────────────────

def load_sweep(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Sweep file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Sweep file is not valid JSON: {path}\n  {e}")


def extract_responses(
    sweep: dict,
    endpoints: list[str],
) -> dict[str, dict[str, list[str]]]:
    """
    Returns {model_name: {endpoint_key: [response_text, ...]}}
    Only includes endpoints with at least 2 non-empty responses.
    """
    result: dict[str, dict[str, list[str]]] = {}
    stats = sweep.get("stats", {})

    for model, model_stats in stats.items():
        model_data: dict[str, list[str]] = {}
        for ep_key in endpoints:
            ep_data = model_stats.get(ep_key, {})
            responses = [
                r for r in ep_data.get("responses", [])
                if r and r.strip()
            ]
            if len(responses) >= 2:
                model_data[ep_key] = responses
        if model_data:
            result[model] = model_data

    return result


# ── Judgement runner ──────────────────────────────────────────────────────────

def build_prompt(question: str, responses: list[str]) -> str:
    numbered = "\n\n".join(
        f"[{i+1}] {r.strip()}" for i, r in enumerate(responses)
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        n              = len(responses),
        question       = question,
        responses_block= numbered,
    )


def run_judgement(
    sweep_path: Path,
    judge_model: str,
    endpoints: list[str],
    timeout: int,
    out_path: Path | None,
) -> dict:
    sweep    = load_sweep(sweep_path)
    question = sweep.get("meta", {}).get("question", "Unknown question")
    run_at   = sweep.get("meta", {}).get("run_at", "unknown")
    n_sweeps = sweep.get("meta", {}).get("sweeps", "?")

    print(f"\n{'═'*70}")
    print(f"  probe_sweep_judge.py")
    print(f"  Sweep file  : {sweep_path.name}")
    print(f"  Sweep run   : {run_at}  ({n_sweeps} sweeps)")
    print(f"  Question    : {question}")
    print(f"  Judge model : {judge_model}")
    print(f"  Endpoints   : {', '.join(endpoints)}")
    print(f"  Timeout     : {timeout}s")
    print(f"{'═'*70}\n")

    # Verify judge model is reachable
    judge_url = _cloud_url_for(judge_model)
    try:
        r = requests.get(
            f"{judge_url}/api/version",
            headers=_auth_headers(judge_url),
            timeout=10,
        )
        version = r.json().get("version", "?")
        print(f"  Ollama {version} reachable at {judge_url}\n")
    except Exception as e:
        print(f"  ✗ Cannot reach judge endpoint: {e}")
        sys.exit(1)

    model_responses = extract_responses(sweep, endpoints)
    if not model_responses:
        print("  ✗ No eligible responses found in sweep file.")
        sys.exit(1)

    judgements: dict[str, dict] = {}
    total_tasks = sum(len(eps) for eps in model_responses.values())
    task_num    = 0

    for model, ep_map in model_responses.items():
        print(f"  Model: {model}")
        judgements[model] = {}

        for ep_key, responses in ep_map.items():
            task_num += 1
            n = len(responses)
            ep_label = {
                "chat":      "chat",
                "gen_raw":   "generate raw:true",
                "gen_plain": "generate raw:false",
            }.get(ep_key, ep_key)

            print(f"    [{task_num}/{total_tasks}] {ep_label}  ({n} responses) ... ",
                  end="", flush=True)

            prompt = build_prompt(question, responses)
            ok, raw_response, elapsed = call_judge(
                judge_model, JUDGE_SYSTEM_PROMPT, prompt, timeout
            )

            if not ok:
                print(f"✗ FAILED  ({elapsed:.1f}s)  {raw_response[:80]}")
                judgements[model][ep_key] = {
                    "success":  False,
                    "error":    raw_response,
                    "elapsed_s": elapsed,
                }
                continue

            parsed = parse_judge_response(raw_response)
            if parsed is None:
                print(f"✗ PARSE ERROR  ({elapsed:.1f}s)")
                judgements[model][ep_key] = {
                    "success":    False,
                    "error":      "JSON parse failed",
                    "raw":        raw_response[:500],
                    "elapsed_s":  elapsed,
                }
                continue

            verdict   = parsed.get("repeatability_verdict", "?")
            quality   = parsed.get("quality_verdict", "?")
            flag      = " ⚠" if verdict != "CONSISTENT" or quality != "GOOD" else " ✓"
            print(f"{verdict}/{quality}{flag}  ({elapsed:.1f}s)")

            judgements[model][ep_key] = {
                "success":    True,
                "elapsed_s":  elapsed,
                "n_responses": n,
                "assessment": parsed,
                "raw_response": raw_response,
            }

        print()

    # ── Build output ──────────────────────────────────────────────────────────
    output = {
        "meta": {
            "judged_at":    datetime.now().isoformat(),
            "sweep_file":   str(sweep_path),
            "sweep_run_at": run_at,
            "sweep_sweeps": n_sweeps,
            "question":     question,
            "judge_model":  judge_model,
            "endpoints":    endpoints,
            "timeout_s":    timeout,
        },
        "judgements": judgements,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"  JUDGE SUMMARY\n")
    print(f"  {'Model':<40} {'Endpoint':<16} {'Repeat':<13} {'Quality'}")
    print(f"  {'─'*40} {'─'*16} {'─'*13} {'─'*10}")

    for model, ep_map in judgements.items():
        for ep_key, result in ep_map.items():
            ep_label = {"chat": "chat", "gen_raw": "gen raw:T",
                        "gen_plain": "gen raw:F"}.get(ep_key, ep_key)
            if not result.get("success"):
                print(f"  {model:<40} {ep_label:<16} {'ERROR':<13} —")
                continue
            a       = result["assessment"]
            repeat  = a.get("repeatability_verdict", "?")
            quality = a.get("quality_verdict", "?")
            flag    = " ⚠" if repeat != "CONSISTENT" or quality != "GOOD" else " ✓"
            print(f"  {model:<40} {ep_label:<16} {repeat:<13} {quality}{flag}")

    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    if out_path is None:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT_DIR / "results" / "data" / "judges" / f"judge_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Results saved → {out_path}")
    print(f"{'─'*70}\n")

    return output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge analysis of probe_endpoint_sweep.py results"
    )
    parser.add_argument(
        "--sweep", required=True, metavar="PATH",
        help="Path to endpoint_sweep JSON file",
    )
    parser.add_argument(
        "--judge", default=DEFAULT_JUDGE, metavar="MODEL",
        help=f"Judge model name (default: {DEFAULT_JUDGE})",
    )
    parser.add_argument(
        "--endpoints", nargs="+",
        choices=list(SWEEP_ENDPOINTS.keys()),
        default=["chat", "gen_plain"],
        metavar="EP",
        help="Endpoints to judge: chat gen_raw gen_plain (default: chat gen_plain)",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Save results to this path (default: results/judge_TIMESTAMP.json)",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    out_path   = Path(args.out) if args.out else None

    run_judgement(
        sweep_path  = sweep_path,
        judge_model = args.judge,
        endpoints   = args.endpoints,
        timeout     = args.timeout,
        out_path    = out_path,
    )


if __name__ == "__main__":
    main()
