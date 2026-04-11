"""
probe_static.py  v2.6
─────────────────────────────────────────────────────────────────────────────
Static probe — /api/chat + /api/generate across Ollama models.
Questions and models loaded from JSON files; all selection at runtime.

Routing: 'cloud' in model name → https://ollama.com (Bearer OLLAMA_API_KEY)
         otherwise              → http://localhost:11434

Generate endpoint behaviour:
  Local GGUF models  — raw:true  (bypasses chat template, base weight access)
  Local raw:false models — raw:false (raw:true broken for these in Ollama v0.19)
                           Detected via needs_raw_false() — checks format and family.
                           Covers: safetensors/MLX models AND gemma4 family GGUF.
  Cloud models       — raw:true rejected (400); plain prompt, no system msg.
  Prompt frame recorded in metadata as prompt_frame field.

needs_raw_false() detection:
  format == "safetensors" (MLX backend, e.g. qwen3.5:35b-a3b-coding-nvfp4)
    → raw:true returns 49 tokens in 0.7s then stops (shallow/spurious)
  family == "gemma4" (GGUF but new arch, e.g. gemma4:26b)
    → raw:true hangs indefinitely in Ollama v0.19
  Both confirmed via probe_endpoint_test.py --all-enabled

Think block extraction — two formats:
  Tag-based  : <think>...</think> embedded in content (DeepSeek, Qwen)
  Field-based: message.thinking separate field in API response (Gemma4)
  Both captured and stored in think_block. answer is always think-free.

Changes v2.6: query_generate() calls needs_raw_false() to detect models that
              require raw:false. Covers both safetensors/MLX and gemma4 family.
              needs_raw_false() caches results via _mlx_cache to avoid repeated
              /api/show calls per run.
Changes v2.5: query_chat() captures message.thinking field (Gemma4 format).
Changes v2.4: Automatic duplicate-digest deduplication at run time.
Changes v2.3: model_is_available() always checks local /api/tags.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

OLLAMA_LOCAL = "http://localhost:11434"
OLLAMA_CLOUD = "https://ollama.com"
DEFAULT_QUESTIONS_FILE = Path(__file__).parent / "probes" / "questions.json"
DEFAULT_MODELS_FILE    = Path(__file__).parent / "probes" / "probe_models.json"
DEFAULT_OUTPUT_DIR     = Path(__file__).parent / "results" / "data" / "probes"

# Cache of needs_raw_false() results — keyed by model name
_mlx_cache: dict[str, bool] = {}

# Model families that break with raw:true regardless of format
# gemma4: GGUF but raw:true hangs in Ollama v0.19 (confirmed via probe_endpoint_test.py)
RAW_FALSE_FAMILIES = {"gemma4"}


def load_questions(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Questions file is not valid JSON: {path}\n  {e}")
    if "questions" not in data:
        raise KeyError(f"Questions file missing 'questions' key: {path}")
    questions = data["questions"]
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError(f"Questions file has empty or invalid 'questions': {path}")
    return questions


def load_models(path: Path) -> list[dict]:
    return json.loads(path.read_text())["models"]


def filter_questions(questions, ids, theme, tags, all_questions):
    if all_questions:
        return questions
    if ids:
        # Case-insensitive, preserves original casing (Q10b not Q10B)
        id_set = {q.lower() for q in ids}
        return [q for q in questions if q["id"].lower() in id_set]
    result = questions
    if theme:
        result = [q for q in result if q.get("theme") == theme]
    if tags:
        tag_set = set(tags)
        result = [q for q in result if tag_set & set(q.get("tags", []))]
    return result


def filter_models(models, names, families, tool_capable_only):
    result = [m for m in models if m.get("enabled", True)]
    if names:
        result = [m for m in result if m["name"] in names]
    if families:
        result = [m for m in result if m.get("family") in families]
    if tool_capable_only:
        result = [m for m in result if m.get("tool_capable")]
    return result


# ── Routing ────────────────────────────────────────────────────────────────────

def llm_url_for(model_name: str) -> str:
    return OLLAMA_CLOUD if "cloud" in model_name.lower() else OLLAMA_LOCAL


def is_cloud(model_name: str) -> bool:
    return "cloud" in model_name.lower()


def auth_headers(llm_url: str) -> dict:
    if llm_url == OLLAMA_CLOUD:
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


# ── raw:false detection ────────────────────────────────────────────────────────

def needs_raw_false(model_name: str, timeout: int = 10) -> bool:
    """
    Returns True if this model must use raw:false on /api/generate.

    Two cases confirmed via probe_endpoint_test.py:
      1. format == "safetensors" (MLX backend, e.g. qwen3.5:35b-a3b-coding-nvfp4)
         raw:true completes in ~0.7s with 49 tokens — shallow, spurious output.
      2. family == "gemma4" (GGUF, e.g. gemma4:26b)
         raw:true hangs indefinitely in Ollama v0.19.

    Cloud models always return False — they use plain prompt with no raw flag.
    Results are cached per model name to avoid repeated /api/show calls.
    """
    if is_cloud(model_name):
        return False
    if model_name in _mlx_cache:
        return _mlx_cache[model_name]
    try:
        r = requests.post(
            f"{OLLAMA_LOCAL}/api/show",
            json={"model": model_name},
            timeout=timeout,
        )
        if r.status_code == 200:
            details = r.json().get("details", {})
            fmt     = details.get("format", "")
            family  = details.get("family", "")
            result  = (fmt == "safetensors") or (family in RAW_FALSE_FAMILIES)
            _mlx_cache[model_name] = result
            return result
    except Exception:
        pass
    _mlx_cache[model_name] = False
    return False


# ── Digest deduplication ───────────────────────────────────────────────────────

def get_digest_map(timeout: int = 10) -> dict[str, str]:
    try:
        r = requests.get(f"{OLLAMA_LOCAL}/api/tags", timeout=timeout)
        if r.status_code == 200:
            return {
                m["name"]: m.get("digest", "")[:16]
                for m in r.json().get("models", [])
            }
    except Exception:
        pass
    return {}


def deduplicate_models(
    models: list[str], digest_map: dict[str, str]
) -> tuple[list[str], list[tuple[str, str]]]:
    seen:    dict[str, str] = {}
    kept:    list[str]      = []
    skipped: list[tuple[str, str]] = []
    for name in models:
        if is_cloud(name):
            kept.append(name)
            continue
        digest = digest_map.get(name)
        if not digest:
            kept.append(name)
            continue
        if digest in seen:
            skipped.append((name, seen[digest]))
        else:
            seen[digest] = name
            kept.append(name)
    return kept, skipped


# ── Query helpers ──────────────────────────────────────────────────────────────

def extract_think(text: str) -> tuple[str, str]:
    """Extract <think>...</think> tags. Returns (think, clean_answer)."""
    think_blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return "\n\n".join(think_blocks).strip(), clean


def query_chat(
    model: str, question: str, timeout: int, llm_url: str = OLLAMA_LOCAL
) -> dict:
    """
    Chat endpoint. Handles two think-block formats:
      - message.thinking field  (Gemma4, Ollama v0.19+)
      - <think>...</think> tags (DeepSeek, Qwen)
    """
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": question}]}
    t0 = time.time()
    try:
        r = requests.post(f"{llm_url}/api/chat", json=payload,
                          headers=auth_headers(llm_url), timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data        = r.json()
        msg         = data.get("message", {})
        content     = msg.get("content", "")
        field_think = msg.get("thinking", "").strip()
        tag_think, answer = extract_think(content)
        think_block = field_think or tag_think
        return {
            "endpoint": "chat", "success": True, "elapsed_s": elapsed,
            "raw_response": content, "think_block": think_block, "answer": answer,
            "think_format": "field" if field_think else ("tag" if tag_think else "none"),
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }
    except requests.exceptions.Timeout:
        return {"endpoint": "chat", "success": False,
                "error": "timeout", "elapsed_s": timeout}
    except Exception as e:
        return {"endpoint": "chat", "success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2)}


def query_generate(
    model: str, question: str, timeout: int, llm_url: str = OLLAMA_LOCAL
) -> dict:
    """
    Generate endpoint. Three routing paths:
      cloud            → plain prompt, no raw flag (raw:true returns 400)
      needs_raw_false  → plain prompt, no raw flag (raw:true broken in Ollama v0.19)
      all others       → raw:true (bypasses chat template, accesses base weights)
    """
    cloud    = is_cloud(model)
    use_raw_false = (not cloud) and needs_raw_false(model)
    framed   = f"Question: {question}\n\nAnswer:"

    if cloud:
        payload = {"model": model, "prompt": framed, "stream": False}
        frame   = "Question/Answer (no raw — cloud)"
    elif use_raw_false:
        payload = {"model": model, "prompt": framed, "stream": False}
        frame   = "Question/Answer (raw:false — Ollama v0.19 compat)"
    else:
        payload = {"model": model, "prompt": framed, "stream": False, "raw": True}
        frame   = "Question/Answer (raw:true — GGUF local)"

    t0 = time.time()
    try:
        r = requests.post(f"{llm_url}/api/generate", json=payload,
                          headers=auth_headers(llm_url), timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data    = r.json()
        content = data.get("response", "")
        think, answer = extract_think(content)
        return {
            "endpoint": "generate", "success": True, "elapsed_s": elapsed,
            "prompt_frame": frame, "cloud": cloud,
            "raw_response": content, "think_block": think, "answer": answer,
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }
    except requests.exceptions.Timeout:
        return {"endpoint": "generate", "success": False,
                "error": "timeout", "elapsed_s": timeout, "cloud": cloud}
    except Exception as e:
        return {"endpoint": "generate", "success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2), "cloud": cloud}


def model_is_available(
    model: str, timeout: int = 30, llm_url: str = OLLAMA_LOCAL
) -> bool:
    try:
        r = requests.get(f"{OLLAMA_LOCAL}/api/tags", timeout=timeout)
        if r.status_code == 200:
            names = [m["name"] for m in r.json().get("models", [])]
            return model in names
        return False
    except Exception:
        return False


# ── Output ────────────────────────────────────────────────────────────────────

def write_markdown(results: list, questions: list[dict], path: Path):
    lines = ["# Probe Results", f"*Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
             "", "## Questions", ""]
    for q in questions:
        lines.append(
            f"- **{q['id']}** [{q['theme']}] `{', '.join(q.get('tags',[]))}` — {q['text']}"
        )
    lines.append("")

    for entry in results:
        model = entry["model"]
        lines += ["---", f"## {model}", ""]
        if entry.get("skipped"):
            lines += [f"> Skipped — {entry.get('reason', 'not available')}", ""]
            continue
        for q in questions:
            qid = q["id"]
            lines += [f"### {qid} — {q['text']}", ""]
            for endpoint in ("chat", "generate"):
                key = f"{qid}_{endpoint}"
                r   = entry["responses"].get(key, {})
                if not r:
                    continue
                lines.append(f"#### `{endpoint}`")
                if not r.get("success"):
                    lines += [f"> Error: {r.get('error', 'unknown')}", ""]
                    continue
                frame = f" · {r['prompt_frame']}" if r.get("prompt_frame") else ""
                fmt   = (f" · think:{r['think_format']}"
                         if r.get("think_format") not in (None, "none") else "")
                lines.append(f"*{r.get('elapsed_s')}s{frame}{fmt}*")
                lines.append("")
                think = r.get("think_block", "").strip()
                if think:
                    lines += ["<details>", "<summary>thinking trace</summary>",
                               "", think, "", "</details>", ""]
                lines += [(r.get("answer") or r.get("raw_response", "")).strip(), ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── List helpers ──────────────────────────────────────────────────────────────

def list_questions(questions: list[dict]):
    themes: dict = {}
    for q in questions:
        themes.setdefault(q["theme"], []).append(q)
    for theme, qs in themes.items():
        print(f"\n  [{theme}]")
        for q in qs:
            print(f"    {q['id']}  {q['text'][:70]}")
            print(f"         tags: {', '.join(q.get('tags', []))}")


def list_models(models: list[dict]):
    families: dict = {}
    for m in models:
        families.setdefault(m.get("family", "other"), []).append(m)
    for fam, ms in families.items():
        print(f"\n  [{fam}]")
        for m in ms:
            enabled = "" if m.get("enabled", True) else "  [disabled]"
            tool    = "tool+" if m.get("tool_capable") else "      "
            think   = "think+" if m.get("think_blocks") else "       "
            bug     = "BUG" if m.get("type_coercion_bug") else "   "
            cloud   = " [cloud→ollama.com]" if "cloud" in m["name"].lower() else ""
            print(f"    {m['name']:<45} {tool} {think} {bug}{cloud}{enabled}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Static probe v2.6")

    qg = parser.add_argument_group("question selection")
    qg.add_argument("--questions", nargs="+", metavar="ID")
    qg.add_argument("--theme", metavar="THEME")
    qg.add_argument("--tag", nargs="+", metavar="TAG")
    qg.add_argument("--all-questions", action="store_true")

    mg = parser.add_argument_group("model selection")
    mg.add_argument("--models", nargs="+", metavar="MODEL")
    mg.add_argument("--family", nargs="+", metavar="FAMILY")

    cfg = parser.add_argument_group("configuration")
    cfg.add_argument("--questions-file",   default=str(DEFAULT_QUESTIONS_FILE))
    cfg.add_argument("--models-file",      default=str(DEFAULT_MODELS_FILE))
    cfg.add_argument("--output-dir",       default=str(DEFAULT_OUTPUT_DIR))
    cfg.add_argument("--timeout",          default=300,  type=int,
                     help="Chat timeout in seconds (default: 300)")
    cfg.add_argument("--generate-timeout", default=1200, type=int,
                     help="Generate timeout in seconds (default: 1200)")
    cfg.add_argument("--no-generate",      action="store_true")
    cfg.add_argument("--label",            default="")

    info = parser.add_argument_group("information")
    info.add_argument("--list-questions", action="store_true")
    info.add_argument("--list-models",    action="store_true")

    args = parser.parse_args()

    all_questions  = load_questions(Path(args.questions_file))
    all_model_defs = load_models(Path(args.models_file))

    if args.list_questions:
        list_questions(all_questions); return
    if args.list_models:
        list_models(all_model_defs);   return

    questions = filter_questions(
        all_questions, args.questions, args.theme, args.tag, args.all_questions
    )
    if not questions:
        parser.error("No questions matched. Use --list-questions.")

    if args.models:
        models = args.models
    else:
        defs   = filter_models(all_model_defs, None, args.family, False)
        models = [m["name"] for m in defs]

    if not models:
        parser.error("No models matched. Use --list-models.")

    digest_map          = get_digest_map()
    models, dup_skipped = deduplicate_models(models, digest_map)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_ids     = "_".join(q["id"] for q in questions[:6])
    if len(questions) > 6:
        q_ids += f"_plus{len(questions)-6}"
    label     = f"_{args.label}" if args.label else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = f"probe_{q_ids}{label}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path   = output_dir / f"{stem}.md"

    # Pre-detect raw:false models so they show in the run header
    raw_false_models = (
        [m for m in models if needs_raw_false(m)]
        if not args.no_generate else []
    )

    print(f"\n{'─'*65}")
    print(f"  Static Probe  v2.6")
    print(f"  Questions : {len(questions)}  ({', '.join(q['id'] for q in questions)})")
    print(f"  Models    : {len(models)}")
    if dup_skipped:
        print(f"  Duplicates skipped (same digest):")
        for skipped_name, kept_name in dup_skipped:
            print(f"    {skipped_name} → same weights as {kept_name}")
    print(f"  Endpoints : chat{'' if args.no_generate else ' + generate'}")
    print(f"  Timeouts  : chat={args.timeout}s  generate={args.generate_timeout}s")
    if not args.no_generate:
        print(f"  Generate  : GGUF=raw:true  v0.19-compat=raw:false  cloud=plain")
        if raw_false_models:
            print(f"  raw:false : {', '.join(raw_false_models)}")
    print(f"  Output    : {output_dir}")
    print(f"{'─'*65}\n")

    all_results = []

    for i, model in enumerate(models, 1):
        llm_url  = llm_url_for(model)
        label_s  = "cloud" if llm_url == OLLAMA_CLOUD else "local"
        compat_tag = " [raw:false]" if model in raw_false_models else ""
        print(f"[{i}/{len(models)}] {model}  [{label_s}]{compat_tag}")

        if not model_is_available(model, llm_url=llm_url):
            print(f"  ⚠ Skipping — not available")
            all_results.append({"model": model, "skipped": True,
                                 "reason": "not available", "responses": {}})
            continue

        model_result = {"model": model, "skipped": False,
                        "llm_url": llm_url, "responses": {}}

        for q in questions:
            qid = q["id"]
            print(f"  {qid} ", end="", flush=True)

            chat = query_chat(model, q["text"], args.timeout, llm_url=llm_url)
            print(f"chat:{'✓' if chat['success'] else '✗'} ", end="", flush=True)
            model_result["responses"][f"{qid}_chat"] = chat

            if not args.no_generate:
                gen = query_generate(
                    model, q["text"], args.generate_timeout, llm_url=llm_url
                )
                print(f"gen:{'✓' if gen['success'] else '✗'}", flush=True)
                model_result["responses"][f"{qid}_generate"] = gen
            else:
                print(flush=True)

            time.sleep(1)

        all_results.append(model_result)
        json_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_markdown(all_results, questions, md_path)
        print(f"  → saved\n")

    print(f"{'─'*65}")
    print(f"  Done — {len(all_results)} models, {len(questions)} questions")
    if dup_skipped:
        print(f"  Duplicates skipped : {len(dup_skipped)}"
              f"  ({', '.join(n for n, _ in dup_skipped)})")
    print(f"  JSON : {json_path}")
    print(f"  MD   : {md_path}")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()
