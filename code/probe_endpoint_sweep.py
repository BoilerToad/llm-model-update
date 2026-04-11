#!/usr/bin/env python3
"""
probe_endpoint_sweep.py
───────────────────────
Runs multiple sweeps of the endpoint test across all enabled local models,
collects timing and reliability data, then runs a post-evaluation that
recommends changes to probe_models.json.

What it measures per model per sweep:
  - /api/chat            : elapsed, success, eval_count, response_text
  - /api/generate raw:T  : elapsed, success, eval_count, token_ratio vs raw:F, response_text
  - /api/generate raw:F  : elapsed, success, eval_count, response_text

Repeatability analysis:
  Stores all response texts and computes pairwise Jaccard similarity on
  word sets between consecutive sweeps. High similarity = repeatable output.
  Low similarity = model is non-deterministic or KV-cache-sensitive.

Post-evaluation recommendations:
  - needs_raw_false      : flag models where raw:T consistently fails or is shallow
  - chat_only            : flag models where both generate modes consistently fail
  - unreliable           : flag models with high failure rate (>30%) on any endpoint
  - suspiciously_fast    : flag raw:T completions averaging <2s with <60 tokens
  - enabled: false       : suggest disabling models that fail chat entirely

Usage:
    python probe_endpoint_sweep.py                          # 11 sweeps, all enabled
    python probe_endpoint_sweep.py --sweeps 20             # custom sweep count
    python probe_endpoint_sweep.py --model gemma4:26b      # single model
    python probe_endpoint_sweep.py --timeout 60            # custom timeout
    python probe_endpoint_sweep.py --out results/data/sweeps/sweep.json  # custom output path
    python probe_endpoint_sweep.py --no-generate           # chat only
    python probe_endpoint_sweep.py --question-id Q10b      # use question from questions.json
    python probe_endpoint_sweep.py --list-questions        # show available question IDs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import requests

OLLAMA_LOCAL    = "http://localhost:11434"
SCRIPT_DIR      = Path(__file__).parent
ROOT_DIR        = SCRIPT_DIR.parent
MODELS_FILE     = ROOT_DIR / "probes" / "probe_models.json"
QUESTIONS_FILE  = ROOT_DIR / "probes" / "questions.json"

from probe_query_openai import (
    is_openai_backend,
    query_chat_openai,
    check_api_key as check_openai_api_key,
)

# Default question used when --question-id is not provided
DEFAULT_QUESTION = "What is democracy? Answer in two sentences."
DEFAULT_PROMPT   = f"Question: {DEFAULT_QUESTION}\n\nAnswer:"

# Thresholds for post-evaluation
FAIL_RATE_THRESHOLD     = 0.30   # >30% failures → unreliable
SHALLOW_TIME_THRESHOLD  = 2.0    # raw:T under 2s → suspicious
SHALLOW_TOKEN_THRESHOLD = 60     # raw:T under 60 tokens → suspicious
TOKEN_RATIO_THRESHOLD   = 5.0    # raw:F generates 5x+ more tokens than raw:T → shallow

# Repeatability threshold — similarity below this flags non-deterministic output
LOW_SIMILARITY_THRESHOLD = 0.50  # Jaccard similarity below 50% = low repeatability


# ── Question loading ───────────────────────────────────────────────────────────

def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    """Load all questions from questions.json."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Questions file is not valid JSON: {path}\n  {e}")
    return data.get("questions", [])


def load_question_by_id(question_id: str, path: Path = QUESTIONS_FILE) -> tuple[str, str]:
    """
    Look up a question by ID from questions.json.
    Returns (question_text, generate_prompt) where generate_prompt wraps
    the question text in the standard Question/Answer frame.
    Raises ValueError if the ID is not found.
    """
    questions = load_questions(path)
    # Case-insensitive match to be consistent with probe_static.py
    id_lower = question_id.lower()
    for q in questions:
        if q.get("id", "").lower() == id_lower:
            text   = q["text"]
            prompt = f"Question: {text}\n\nAnswer:"
            return text, prompt
    available = ", ".join(q["id"] for q in questions)
    raise ValueError(
        f"Question ID '{question_id}' not found in {path}.\n"
        f"Available IDs: {available}"
    )


# ── Similarity metric ──────────────────────────────────────────────────────────

def jaccard_similarity(a: str, b: str) -> float:
    """
    Word-level Jaccard similarity between two strings.
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    Returns 0.0 if either string is empty. Range: [0.0, 1.0].
    Robust to word reordering and minor wording changes.
    """
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    intersection = set_a & set_b
    union        = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def pairwise_similarities(texts: list[str]) -> list[float]:
    """
    Compute Jaccard similarity between each consecutive pair of texts.
    For n texts returns n-1 similarity scores.
    """
    return [
        jaccard_similarity(texts[i], texts[i + 1])
        for i in range(len(texts) - 1)
    ]


# ── Single request helpers ─────────────────────────────────────────────────────

def _post(url: str, payload: dict, timeout: int) -> tuple[bool, float, dict, str]:
    """
    Makes a single POST request. Returns (success, elapsed_s, data, error).
    Uses time.perf_counter() for accurate wall-clock measurement.
    """
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        elapsed = round(time.perf_counter() - t0, 3)
        if r.status_code != 200:
            return False, elapsed, {}, f"HTTP {r.status_code}"
        return True, elapsed, r.json(), ""
    except requests.exceptions.Timeout:
        elapsed = round(time.perf_counter() - t0, 3)
        return False, elapsed, {}, f"TIMEOUT after {timeout}s"
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 3)
        return False, elapsed, {}, str(e)[:80]


def run_chat(model: str, timeout: int, question: str = DEFAULT_QUESTION) -> dict:
    """
    Call /api/chat with the given question text.
    question defaults to the built-in democracy question when not specified.
    """
    url     = f"{OLLAMA_LOCAL}/api/chat"
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": question}]}
    ok, elapsed, data, err = _post(url, payload, timeout)
    if not ok:
        return {"success": False, "elapsed_s": elapsed, "error": err,
                "eval_count": 0, "response_len": 0, "response_text": ""}
    msg     = data.get("message", {})
    content = msg.get("content", "")
    think   = msg.get("thinking", "")
    return {
        "success":       True,
        "elapsed_s":     elapsed,
        "error":         "",
        "eval_count":    data.get("eval_count") or 0,
        "response_len":  len(content),
        "think_len":     len(think),
        "response_text": content,
    }


def run_generate(model: str, timeout: int, raw: bool,
                 prompt: str = DEFAULT_PROMPT) -> dict:
    """
    Call /api/generate with the given prompt string.
    prompt defaults to the built-in Question/Answer framing when not specified.
    """
    url     = f"{OLLAMA_LOCAL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    if raw:
        payload["raw"] = True
    ok, elapsed, data, err = _post(url, payload, timeout)
    if not ok:
        return {"success": False, "elapsed_s": elapsed, "error": err,
                "eval_count": 0, "response_len": 0, "raw": raw,
                "response_text": ""}
    content = data.get("response", "")
    contaminated = any(kw in content[:300].lower()
                       for kw in ("question:", "answer:", "\nq:", "\na:"))
    return {
        "success":       True,
        "elapsed_s":     elapsed,
        "error":         "",
        "eval_count":    data.get("eval_count") or 0,
        "response_len":  len(content),
        "raw":           raw,
        "contaminated":  contaminated,
        "response_text": content,
    }


# ── Warmup ─────────────────────────────────────────────────────────────────────

WARMUP_PROMPT = "Reply with the word OK."

def warmup_model(model: str, timeout: int) -> bool:
    """
    Send a trivial untimed prompt to ensure the model is loaded into VRAM
    before sweep timing begins. Result is discarded. Returns True if loaded.
    """
    url     = f"{OLLAMA_LOCAL}/api/chat"
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": WARMUP_PROMPT}]}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


# ── Availability check ─────────────────────────────────────────────────────────

def get_available_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_LOCAL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def get_ollama_version() -> str:
    try:
        r = requests.get(f"{OLLAMA_LOCAL}/api/version", timeout=5)
        return r.json().get("version", "?")
    except Exception:
        return "?"


# ── Sweep runner ───────────────────────────────────────────────────────────────

def run_sweep(
    models: list[str],
    timeout: int,
    no_generate: bool,
    sweep_num: int,
    total_sweeps: int,
    question: str = DEFAULT_QUESTION,
    prompt: str   = DEFAULT_PROMPT,
    registry: dict | None = None,
) -> dict[str, dict]:
    """Run one sweep across all models. Returns {model: {chat, gen_raw, gen_plain}}."""
    results  = {}
    registry = registry or {}
    print(f"\n  Sweep {sweep_num}/{total_sweeps}  {datetime.now():%H:%M:%S}")
    print(f"  {'─'*60}")

    for model in models:
        print(f"  {model:<45} ", end="", flush=True)
        entry  = registry.get(model, {"name": model, "backend": "ollama"})

        # xAI / OpenAI-compatible: chat only, no generate endpoints
        if is_openai_backend(entry):
            oai = query_chat_openai(entry, question, timeout)
            chat = {
                "success":       oai["success"],
                "elapsed_s":     oai["elapsed_s"],
                "error":         oai.get("error", ""),
                "eval_count":    oai.get("eval_count") or 0,
                "response_len":  len(oai.get("raw_response", "")),
                "think_len":     len(oai.get("think_block", "")),
                "response_text": oai.get("answer", ""),
            }
            elapsed_s = f"{chat['elapsed_s']:.1f}s" if chat["success"] else "FAIL"
            print(f"chat:{elapsed_s} raw:-- plain:-- [chat-only]", flush=True)
            results[model] = {"chat": chat}
            time.sleep(0.5)
            continue

        chat = run_chat(model, timeout, question=question)
        chat_status = f"{chat['elapsed_s']:.1f}s" if chat["success"] else "✗"
        print(f"chat:{chat_status} ", end="", flush=True)

        model_result = {"chat": chat}

        if not no_generate:
            gen_raw   = run_generate(model, timeout, raw=True,  prompt=prompt)
            gen_plain = run_generate(model, timeout, raw=False, prompt=prompt)

            raw_status   = f"{gen_raw['elapsed_s']:.1f}s"   if gen_raw["success"]   else "✗"
            plain_status = f"{gen_plain['elapsed_s']:.1f}s" if gen_plain["success"] else "✗"
            print(f"raw:{raw_status} plain:{plain_status}", flush=True)

            model_result["gen_raw"]   = gen_raw
            model_result["gen_plain"] = gen_plain
        else:
            print(flush=True)

        results[model] = model_result
        time.sleep(0.5)

    return results


# ── Statistics aggregator ──────────────────────────────────────────────────────

class ModelStats:
    """Aggregates sweep results for a single model."""

    def __init__(self, name: str):
        self.name = name
        self.total_sweeps:      int = 0

        self.chat_times:        list[float] = []
        self.chat_successes:    int = 0
        self.chat_failures:     int = 0
        self.chat_texts:        list[str]   = []

        self.gen_raw_times:        list[float] = []
        self.gen_raw_successes:    int = 0
        self.gen_raw_failures:     int = 0
        self.gen_raw_tokens:       list[int]   = []
        self.gen_raw_contaminated: int = 0
        self.gen_raw_texts:        list[str]   = []

        self.gen_plain_times:     list[float] = []
        self.gen_plain_successes: int = 0
        self.gen_plain_failures:  int = 0
        self.gen_plain_tokens:    list[int]   = []
        self.gen_plain_texts:     list[str]   = []

    def absorb(self, sweep_result: dict):
        self.total_sweeps += 1

        chat = sweep_result.get("chat", {})
        if chat.get("success"):
            self.chat_successes += 1
            self.chat_times.append(chat["elapsed_s"])
            self.chat_texts.append(chat.get("response_text", ""))
        else:
            self.chat_failures += 1

        gen_raw = sweep_result.get("gen_raw")
        if gen_raw is not None:
            if gen_raw.get("success"):
                self.gen_raw_successes += 1
                self.gen_raw_times.append(gen_raw["elapsed_s"])
                self.gen_raw_tokens.append(gen_raw["eval_count"])
                self.gen_raw_texts.append(gen_raw.get("response_text", ""))
                if gen_raw.get("contaminated"):
                    self.gen_raw_contaminated += 1
            else:
                self.gen_raw_failures += 1

        gen_plain = sweep_result.get("gen_plain")
        if gen_plain is not None:
            if gen_plain.get("success"):
                self.gen_plain_successes += 1
                self.gen_plain_times.append(gen_plain["elapsed_s"])
                self.gen_plain_tokens.append(gen_plain["eval_count"])
                self.gen_plain_texts.append(gen_plain.get("response_text", ""))
            else:
                self.gen_plain_failures += 1

    @property
    def chat_fail_rate(self) -> float:
        total = self.chat_successes + self.chat_failures
        return self.chat_failures / total if total else 0.0

    @property
    def gen_raw_fail_rate(self) -> float:
        total = self.gen_raw_successes + self.gen_raw_failures
        return self.gen_raw_failures / total if total else 0.0

    @property
    def gen_plain_fail_rate(self) -> float:
        total = self.gen_plain_successes + self.gen_plain_failures
        return self.gen_plain_failures / total if total else 0.0

    @property
    def avg_gen_raw_tokens(self) -> float:
        return mean(self.gen_raw_tokens) if self.gen_raw_tokens else 0.0

    @property
    def avg_gen_plain_tokens(self) -> float:
        return mean(self.gen_plain_tokens) if self.gen_plain_tokens else 0.0

    @property
    def token_ratio(self) -> float:
        if self.avg_gen_raw_tokens < 1:
            return 0.0
        return self.avg_gen_plain_tokens / self.avg_gen_raw_tokens

    def similarity_scores(self, texts: list[str]) -> list[float]:
        return pairwise_similarities(texts)

    def avg_similarity(self, texts: list[str]) -> Optional[float]:
        scores = self.similarity_scores(texts)
        return round(mean(scores), 3) if scores else None

    def min_similarity(self, texts: list[str]) -> Optional[float]:
        scores = self.similarity_scores(texts)
        return round(min(scores), 3) if scores else None

    def avg(self, times: list[float]) -> str:
        return f"{mean(times):.1f}s" if times else "—"

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "total_sweeps": self.total_sweeps,
            "chat": {
                "successes":       self.chat_successes,
                "failures":        self.chat_failures,
                "fail_rate":       round(self.chat_fail_rate, 3),
                "avg_elapsed_s":   round(mean(self.chat_times), 3) if self.chat_times else None,
                "stdev_elapsed_s": round(stdev(self.chat_times), 3) if len(self.chat_times) > 1 else None,
                "avg_similarity":  self.avg_similarity(self.chat_texts),
                "min_similarity":  self.min_similarity(self.chat_texts),
                "responses":       self.chat_texts,
            },
            "gen_raw": {
                "successes":      self.gen_raw_successes,
                "failures":       self.gen_raw_failures,
                "fail_rate":      round(self.gen_raw_fail_rate, 3),
                "avg_elapsed_s":  round(mean(self.gen_raw_times), 3) if self.gen_raw_times else None,
                "avg_tokens":     round(self.avg_gen_raw_tokens, 1),
                "contaminated":   self.gen_raw_contaminated,
                "avg_similarity": self.avg_similarity(self.gen_raw_texts),
                "min_similarity": self.min_similarity(self.gen_raw_texts),
                "responses":      self.gen_raw_texts,
            },
            "gen_plain": {
                "successes":      self.gen_plain_successes,
                "failures":       self.gen_plain_failures,
                "fail_rate":      round(self.gen_plain_fail_rate, 3),
                "avg_elapsed_s":  round(mean(self.gen_plain_times), 3) if self.gen_plain_times else None,
                "avg_tokens":     round(self.avg_gen_plain_tokens, 1),
                "avg_similarity": self.avg_similarity(self.gen_plain_texts),
                "min_similarity": self.min_similarity(self.gen_plain_texts),
                "responses":      self.gen_plain_texts,
            },
            "token_ratio_plain_over_raw": round(self.token_ratio, 2),
        }


# ── Post-evaluation ────────────────────────────────────────────────────────────

class Recommendation:
    def __init__(self, model: str, field: str, current, suggested, reason: str, confidence: str):
        self.model      = model
        self.field      = field
        self.current    = current
        self.suggested  = suggested
        self.reason     = reason
        self.confidence = confidence

    def __str__(self):
        return (f"  [{self.confidence}] {self.model}\n"
                f"    field     : {self.field}\n"
                f"    current   : {self.current}\n"
                f"    suggested : {self.suggested}\n"
                f"    reason    : {self.reason}")

    def to_dict(self) -> dict:
        return {
            "model": self.model, "field": self.field,
            "current": self.current, "suggested": self.suggested,
            "reason": self.reason, "confidence": self.confidence,
        }


def load_registry() -> dict[str, dict]:
    data = json.loads(MODELS_FILE.read_text())
    return {m["name"]: m for m in data["models"]}


def post_evaluate(
    stats: dict[str, ModelStats],
    no_generate: bool,
    n_sweeps: int,
) -> list[Recommendation]:
    registry = load_registry()
    recs: list[Recommendation] = []

    for model, s in stats.items():
        entry = registry.get(model, {})

        # ── Chat failures ──────────────────────────────────────────────────────
        if s.chat_fail_rate > FAIL_RATE_THRESHOLD:
            recs.append(Recommendation(
                model, "enabled",
                current   = entry.get("enabled", True),
                suggested = False,
                reason    = (f"Chat failed {s.chat_failures}/{s.total_sweeps} sweeps "
                             f"({s.chat_fail_rate:.0%})"),
                confidence = "HIGH" if s.chat_fail_rate > 0.6 else "MEDIUM",
            ))

        # ── Chat low repeatability ─────────────────────────────────────────────
        chat_sim = s.avg_similarity(s.chat_texts)
        if chat_sim is not None and chat_sim < LOW_SIMILARITY_THRESHOLD:
            recs.append(Recommendation(
                model, "notes",
                current   = entry.get("notes", ""),
                suggested = (f"[low-repeatability chat — avg Jaccard={chat_sim:.2f}] "
                             + entry.get("notes", "")),
                reason    = (f"Chat responses have low word-overlap between sweeps "
                             f"(avg Jaccard={chat_sim:.2f}, threshold={LOW_SIMILARITY_THRESHOLD}). "
                             f"Model output is highly non-deterministic or temperature-sensitive."),
                confidence = "MEDIUM",
            ))

        if no_generate:
            continue

        # ── raw:true failure analysis — mutually exclusive tiers ───────────────
        raw_failure_rec_raised = False

        if s.gen_raw_fail_rate >= 0.90 and s.gen_plain_fail_rate < 0.20:
            current_notes = entry.get("notes", "")
            if "needs_raw_false" not in current_notes:
                recs.append(Recommendation(
                    model, "notes",
                    current   = current_notes[:60] + "..." if len(current_notes) > 60 else current_notes,
                    suggested = f"[needs_raw_false] {current_notes}".strip(),
                    reason    = (f"raw:true failed {s.gen_raw_failures}/{s.total_sweeps} sweeps "
                                 f"({s.gen_raw_fail_rate:.0%}), raw:false succeeded "
                                 f"({1-s.gen_plain_fail_rate:.0%})."),
                    confidence = "HIGH",
                ))
            raw_failure_rec_raised = True

        elif FAIL_RATE_THRESHOLD < s.gen_raw_fail_rate < 0.90:
            recs.append(Recommendation(
                model, "notes",
                current   = entry.get("notes", ""),
                suggested = (f"[raw:true intermittent — {s.gen_raw_fail_rate:.0%} fail rate "
                             f"over {n_sweeps} sweeps] " + entry.get("notes", "")),
                reason    = (f"raw:true failed {s.gen_raw_fail_rate:.0%} of sweeps. "
                             f"Likely VRAM contention artifact. Monitor across more sweeps."),
                confidence = "MEDIUM",
            ))
            raw_failure_rec_raised = True

        if not raw_failure_rec_raised:
            shallow      = (s.avg_gen_raw_tokens < SHALLOW_TOKEN_THRESHOLD
                            and s.gen_raw_times
                            and mean(s.gen_raw_times) < SHALLOW_TIME_THRESHOLD)
            high_ratio   = s.token_ratio > TOKEN_RATIO_THRESHOLD and s.avg_gen_raw_tokens > 0
            contaminated = (s.gen_raw_contaminated / s.total_sweeps) > 0.5

            if (shallow or high_ratio or contaminated) and s.gen_raw_fail_rate < 0.5:
                reasons = []
                if shallow:
                    reasons.append(f"avg raw:T tokens={s.avg_gen_raw_tokens:.0f} in "
                                   f"{mean(s.gen_raw_times):.1f}s (shallow)")
                if high_ratio:
                    reasons.append(f"raw:F generates {s.token_ratio:.0f}x more tokens than raw:T")
                if contaminated:
                    reasons.append(f"raw:T contaminated in "
                                   f"{s.gen_raw_contaminated}/{s.total_sweeps} sweeps")
                current_notes = entry.get("notes", "")
                if "needs_raw_false" not in current_notes:
                    recs.append(Recommendation(
                        model, "notes",
                        current   = current_notes[:60] + "..." if len(current_notes) > 60 else current_notes,
                        suggested = f"[needs_raw_false — shallow raw:T] {current_notes}".strip(),
                        reason    = "raw:true output is shallow/spurious: " + "; ".join(reasons),
                        confidence = "HIGH" if (shallow and high_ratio) else "MEDIUM",
                    ))

        if s.gen_raw_fail_rate > 0.80 and s.gen_plain_fail_rate > 0.80:
            current_notes = entry.get("notes", "")
            if "chat-only" not in current_notes:
                recs.append(Recommendation(
                    model, "notes",
                    current   = current_notes[:60] + "..." if len(current_notes) > 60 else current_notes,
                    suggested = f"[chat-only — generate endpoint unreliable] {current_notes}".strip(),
                    reason    = (f"Both generate modes failed: "
                                 f"raw:T={s.gen_raw_fail_rate:.0%}, raw:F={s.gen_plain_fail_rate:.0%}"),
                    confidence = "HIGH",
                ))

        plain_sim = s.avg_similarity(s.gen_plain_texts)
        if plain_sim is not None and plain_sim < LOW_SIMILARITY_THRESHOLD:
            recs.append(Recommendation(
                model, "notes",
                current   = entry.get("notes", ""),
                suggested = (f"[low-repeatability generate — avg Jaccard={plain_sim:.2f}] "
                             + entry.get("notes", "")),
                reason    = (f"Generate (raw:F) responses have low word-overlap between sweeps "
                             f"(avg Jaccard={plain_sim:.2f}). "
                             f"Consider lower temperature for research reproducibility."),
                confidence = "LOW",
            ))

    return recs


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_sweep_summary(stats: dict[str, ModelStats], no_generate: bool):
    SEP = "─" * 80
    print(f"\n{SEP}")
    print(f"  SWEEP SUMMARY  ({next(iter(stats.values())).total_sweeps} sweeps)\n")

    if no_generate:
        print(f"  {'Model':<40} {'Chat OK':>7} {'Chat avg':>9} {'Fail%':>6} {'Similarity':>11}")
        print(f"  {'─'*40} {'─'*7} {'─'*9} {'─'*6} {'─'*11}")
        for model, s in stats.items():
            total   = s.chat_successes + s.chat_failures
            sim     = s.avg_similarity(s.chat_texts)
            sim_str = f"{sim:.2f}" if sim is not None else "—"
            flag    = " ⚠" if sim is not None and sim < LOW_SIMILARITY_THRESHOLD else ""
            print(f"  {model:<40} {s.chat_successes:>3}/{total:<3} "
                  f"{s.avg(s.chat_times):>9} {s.chat_fail_rate:>5.0%} {sim_str:>10}{flag}")
    else:
        print(f"  {'Model':<36} {'Chat':>6} {'ChatSim':>8} "
              f"{'raw:T':>6} {'raw:TSim':>9} {'raw:F':>6} {'raw:FSim':>9} {'TokRatio':>9}")
        print(f"  {'─'*36} {'─'*6} {'─'*8} "
              f"{'─'*6} {'─'*9} {'─'*6} {'─'*9} {'─'*9}")
        for model, s in stats.items():
            total     = s.chat_successes + s.chat_failures
            raw_total = s.gen_raw_successes + s.gen_raw_failures
            pln_total = s.gen_plain_successes + s.gen_plain_failures

            chat_sim  = s.avg_similarity(s.chat_texts)
            raw_sim   = s.avg_similarity(s.gen_raw_texts)
            plain_sim = s.avg_similarity(s.gen_plain_texts)

            def _sim(v): return f"{v:.2f}" if v is not None else "—"
            def _flag(v): return "⚠" if v is not None and v < LOW_SIMILARITY_THRESHOLD else " "

            ratio_str = f"{s.token_ratio:.1f}x" if s.token_ratio > 0 else "—"
            print(
                f"  {model:<36} "
                f"{s.chat_successes:>3}/{total:<2} {_sim(chat_sim):>7}{_flag(chat_sim)} "
                f"{s.gen_raw_successes:>3}/{raw_total:<2} {_sim(raw_sim):>8}{_flag(raw_sim)} "
                f"{s.gen_plain_successes:>3}/{pln_total:<2} {_sim(plain_sim):>8}{_flag(plain_sim)} "
                f"{ratio_str:>9}"
            )
    print()
    print(f"  Similarity = word-level Jaccard between consecutive sweeps (0=different, 1=identical)")
    print(f"  ⚠ = below threshold ({LOW_SIMILARITY_THRESHOLD:.2f})")
    print()


def print_recommendations(recs: list[Recommendation]):
    SEP = "─" * 80
    print(f"{SEP}")
    if not recs:
        print(f"  ✓ POST-EVALUATION: No changes recommended to probe_models.json")
        print()
        return

    high   = [r for r in recs if r.confidence == "HIGH"]
    medium = [r for r in recs if r.confidence == "MEDIUM"]
    low    = [r for r in recs if r.confidence == "LOW"]

    print(f"  POST-EVALUATION RECOMMENDATIONS  "
          f"(HIGH:{len(high)}  MEDIUM:{len(medium)}  LOW:{len(low)})\n")

    for group, label in [(high, "HIGH CONFIDENCE"), (medium, "MEDIUM CONFIDENCE"),
                         (low, "LOW CONFIDENCE")]:
        if group:
            print(f"  ── {label} {'─'*(50-len(label))}")
            for rec in group:
                print(str(rec))
                print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-sweep endpoint reliability and repeatability test"
    )
    parser.add_argument("--sweeps",       type=int, default=11,
                        help="Number of sweeps to run (default: 11)")
    parser.add_argument("--model",        metavar="NAME",
                        help="Test a single model instead of all enabled")
    parser.add_argument("--timeout",      type=int, default=60,
                        help="Per-request timeout in seconds (default: 60)")
    parser.add_argument("--no-generate",  action="store_true",
                        help="Skip generate endpoint tests")
    parser.add_argument("--out",          metavar="PATH",
                        help="Save full results JSON to this path")
    parser.add_argument("--pause",        type=float, default=2.0,
                        help="Seconds to pause between sweeps (default: 2.0)")
    parser.add_argument("--warmup",       action="store_true", default=True,
                        help="Send untimed warmup prompt before sweeps (default: True)")
    parser.add_argument("--no-warmup",    dest="warmup", action="store_false",
                        help="Skip warmup pass")

    # Question selection — mutually exclusive
    qgroup = parser.add_mutually_exclusive_group()
    qgroup.add_argument(
        "--question-id", metavar="ID",
        help="Use a question from questions.json by ID (e.g. Q10b, Q03). "
             "Overrides the default democracy question.",
    )
    qgroup.add_argument(
        "--list-questions", action="store_true",
        help="List all available question IDs from questions.json and exit.",
    )

    args = parser.parse_args()

    # ── List questions and exit ────────────────────────────────────────────────
    if args.list_questions:
        try:
            questions = load_questions()
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ {e}")
            sys.exit(1)
        print(f"\n  Available questions ({len(questions)} total)\n")
        themes: dict[str, list] = {}
        for q in questions:
            themes.setdefault(q.get("theme", "other"), []).append(q)
        for theme, qs in themes.items():
            print(f"  [{theme}]")
            for q in qs:
                tags = ", ".join(q.get("tags", []))
                print(f"    {q['id']:<8} {q['text'][:65]}")
                if tags:
                    print(f"             tags: {tags}")
        print()
        return

    # ── Resolve question ───────────────────────────────────────────────────────
    question_id: Optional[str] = None
    if args.question_id:
        try:
            question, prompt = load_question_by_id(args.question_id)
            question_id = args.question_id.upper()
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ {e}")
            sys.exit(1)
    else:
        question = DEFAULT_QUESTION
        prompt   = DEFAULT_PROMPT

    # ── Ollama connectivity ────────────────────────────────────────────────────
    version   = get_ollama_version()
    available = get_available_models()
    if not available:
        print("✗ Cannot reach Ollama or no models available")
        sys.exit(1)

    if args.model:
        if args.model not in available:
            # Check if it's an xAI/OpenAI-backend model in the registry
            registry_data = json.loads(MODELS_FILE.read_text())
            reg_map = {m["name"]: m for m in registry_data["models"]}
            entry = reg_map.get(args.model, {})
            if not is_openai_backend(entry):
                print(f"✗ Model not available: {args.model}")
                sys.exit(1)
        models = [args.model]
    else:
        registry_data = json.loads(MODELS_FILE.read_text())
        models = [
            m["name"] for m in registry_data["models"]
            if m.get("enabled", True)
            and not is_openai_backend(m)
            and "cloud" not in m["name"].lower()
            and m["name"] in available
        ]

    if not models:
        print("✗ No models to test")
        sys.exit(1)

    # ── Run header ─────────────────────────────────────────────────────────────
    q_source = (f"questions.json [{question_id}]"
                if question_id else "built-in default")
    print(f"\n{'═'*80}")
    print(f"  probe_endpoint_sweep.py")
    print(f"  Ollama v{version} at {OLLAMA_LOCAL}")
    print(f"  Models    : {len(models)}")
    print(f"  Sweeps    : {args.sweeps}")
    print(f"  Timeout   : {args.timeout}s per request")
    print(f"  Endpoints : chat{'' if args.no_generate else ' + generate (raw:T and raw:F)'}")
    print(f"  Pause     : {args.pause}s between sweeps")
    print(f"  Warmup    : {'yes (untimed)' if args.warmup else 'no'}")
    print(f"  Q source  : {q_source}")
    print(f"  Question  : {question}")
    print(f"  Started   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'═'*80}")

    if args.warmup:
        print(f"\n  Warming up {len(models)} model(s) — loading into VRAM (untimed)...")
        for model in models:
            print(f"    {model:<45} ", end="", flush=True)
            ok = warmup_model(model, timeout=args.timeout)
            print("✓ loaded" if ok else "✗ failed")
        print()

    # ── Build registry map for run_sweep routing ──────────────────────────────
    reg_map = load_registry()

    # ── Run sweeps ─────────────────────────────────────────────────────────────
    stats: dict[str, ModelStats] = {m: ModelStats(m) for m in models}
    all_sweeps = []

    for sweep_num in range(1, args.sweeps + 1):
        sweep_results = run_sweep(
            models       = models,
            timeout      = args.timeout,
            no_generate  = args.no_generate,
            sweep_num    = sweep_num,
            total_sweeps = args.sweeps,
            question     = question,
            prompt       = prompt,
            registry     = reg_map,
        )
        all_sweeps.append(sweep_results)
        for model, result in sweep_results.items():
            stats[model].absorb(result)
        if sweep_num < args.sweeps:
            time.sleep(args.pause)

    print_sweep_summary(stats, args.no_generate)
    recs = post_evaluate(stats, args.no_generate, args.sweeps)
    print_recommendations(recs)

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "meta": {
            "run_at":       datetime.now().isoformat(),
            "ollama":       version,
            "sweeps":       args.sweeps,
            "timeout_s":    args.timeout,
            "no_generate":  args.no_generate,
            "warmup":       args.warmup,
            "question":     question,
            "question_id":  question_id,
            "question_source": q_source,
            "models":       models,
        },
        "stats":           {m: s.to_dict() for m, s in stats.items()},
        "recommendations": [r.to_dict() for r in recs],
        "raw_sweeps":      all_sweeps,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        q_tag    = f"_{question_id}" if question_id else ""
        out_path = SCRIPT_DIR / "results" / "data" / "sweeps" / f"endpoint_sweep{q_tag}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved → {out_path}")
    print(f"{'─'*80}\n")


if __name__ == "__main__":
    main()
