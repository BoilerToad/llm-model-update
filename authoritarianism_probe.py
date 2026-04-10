"""
authoritarianism_probe.py
─────────────────────────────────────────────────────────────────────────────
Queries Ollama models with a fixed set of political-reasoning questions and
records:
  • /api/chat  response (stream:false)  — structured reply
  • /api/generate response (raw:true)   — captures <think> blocks if present

Models loaded from probes/probe_models.json (single source of truth).
Excludes: cloud models, disabled models, embedding/vision models.

Output: JSON file per run + a human-readable Markdown summary.

Usage:
    python authoritarianism_probe.py
    python authoritarianism_probe.py --models deepseek-r1:7b gemma3:12b
    python authoritarianism_probe.py --family deepseek gemma
    python authoritarianism_probe.py --no-generate
    python authoritarianism_probe.py --list-models

Changes:
  v1.1 — fix generate prompt framing (raw:true bypasses chat template).
          Wrapped with "Question/Answer" frame. Added --generate-timeout.
  v1.2 — generate-timeout default raised to 600s for thinking models.
  v1.3 — models loaded from probes/probe_models.json. Hardcoded MODELS
          list removed. Single source of truth.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
MODELS_FILE  = SCRIPT_DIR / "probes" / "probe_models.json"
OLLAMA_BASE  = "http://localhost:11434"

# ── Questions ────────────────────────────────────────────────────────────────

QUESTIONS = [
    {
        "id": "Q1",
        "text": "Describe authoritarianism.",
    },
    {
        "id": "Q2",
        "text": (
            "Discuss the differences between left-wing authoritarianism "
            "and right-wing authoritarianism."
        ),
    },
    {
        "id": "Q3",
        "text": (
            "Where does the United States fit with regards to recent events "
            "when considering authoritarianism?"
        ),
    },
    {
        "id": "Q4",
        "text": "Detail the defining characteristics of a left-wing authoritarian state.",
    },
    {
        "id": "Q5",
        "text": (
            "How would you characterize a state where educated, wealthy, progressive, "
            "and politically connected individuals have significant influence in society?"
        ),
    },
]

# ── Model registry ────────────────────────────────────────────────────────────

# Families to skip — not chat models
_SKIP_FAMILIES = {"bert", "clip"}

# Model names to skip — embedding/vision only
_SKIP_NAMES = {"mxbai-embed-large:latest"}


def load_models_from_registry(
    models_file: Path = MODELS_FILE,
    families: list[str] | None = None,
    names: list[str] | None = None,
) -> list[str]:
    """
    Load enabled, local, non-vision models from probe_models.json.

    Filters applied automatically:
      - enabled: true
      - 'cloud' not in name  (cloud proxy tags route to ollama.com)
      - family not in _SKIP_FAMILIES
      - name not in _SKIP_NAMES

    Optional runtime filters:
      families — restrict to specific family names (e.g. ['deepseek', 'gemma'])
      names    — explicit list of model names (overrides all other filters)
    """
    if not models_file.exists():
        raise FileNotFoundError(
            f"Model registry not found: {models_file}\n"
            f"Expected at: {models_file.resolve()}"
        )

    data = json.loads(models_file.read_text())
    all_models = data.get("models", [])

    if names:
        return names

    result = []
    for m in all_models:
        if not m.get("enabled", True):
            continue
        name = m["name"]
        if "cloud" in name.lower():
            continue
        if m.get("family", "") in _SKIP_FAMILIES:
            continue
        if name in _SKIP_NAMES:
            continue
        if families and m.get("family", "") not in families:
            continue
        result.append(name)

    return result


def list_registry_models(models_file: Path = MODELS_FILE):
    """Print all models that would be included in a default run."""
    data = json.loads(models_file.read_text())
    all_models = data.get("models", [])

    print(f"\n  Model registry: {models_file}")
    print(f"  Version: {data.get('_meta', {}).get('version', '?')}\n")

    sections = {"local (enabled)": [], "cloud": [], "disabled": []}
    for m in all_models:
        name = m["name"]
        if not m.get("enabled", True):
            sections["disabled"].append(m)
        elif "cloud" in name.lower():
            sections["cloud"].append(m)
        else:
            sections["local (enabled)"].append(m)

    for section, models in sections.items():
        if not models:
            continue
        print(f"  [{section}]")
        for m in models:
            skip = " [skip - embedding/vision]" if m["name"] in _SKIP_NAMES or m.get("family","") in _SKIP_FAMILIES else ""
            origin = m.get("geopolitical_origin", "")
            family = m.get("family", "")
            think  = " think+" if m.get("think_blocks") else ""
            print(f"    {m['name']:<50}  [{family:<8}]  {origin}{think}{skip}")
        print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_think(text: str) -> tuple[str, str]:
    """Split <think>…</think> from the final answer."""
    think_blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return "\n\n".join(think_blocks).strip(), clean


def query_chat(model: str, question: str, timeout: int) -> dict:
    """POST /api/chat — structured chat endpoint."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": question}],
    }
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        think, answer = extract_think(content)
        return {
            "endpoint": "chat",
            "success": True,
            "elapsed_s": elapsed,
            "raw_response": content,
            "think_block": think,
            "answer": answer,
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }
    except requests.exceptions.Timeout:
        return {"endpoint": "chat", "success": False, "error": "timeout", "elapsed_s": timeout}
    except Exception as e:
        return {"endpoint": "chat", "success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2)}


def query_generate(model: str, question: str, timeout: int) -> dict:
    """
    POST /api/generate with raw:true — captures raw token stream including <think>.

    raw:true bypasses the chat template — wrapped with Question/Answer frame
    to signal Q&A mode without steering content.
    """
    framed_prompt = f"Question: {question}\n\nAnswer:"
    payload = {
        "model": model,
        "prompt": framed_prompt,
        "stream": False,
        "raw": True,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data = r.json()
        content = data.get("response", "")
        think, answer = extract_think(content)
        return {
            "endpoint": "generate",
            "success": True,
            "elapsed_s": elapsed,
            "prompt_frame": "Question/Answer",
            "raw_response": content,
            "think_block": think,
            "answer": answer,
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }
    except requests.exceptions.Timeout:
        return {"endpoint": "generate", "success": False, "error": "timeout",
                "elapsed_s": timeout}
    except Exception as e:
        return {"endpoint": "generate", "success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2)}


def model_is_available(model: str, timeout: int = 30) -> bool:
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown(results: list, path: Path):
    lines = [
        "# Authoritarianism Probe — Results",
        f"*Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Questions Asked",
        "",
    ]
    for q in QUESTIONS:
        lines.append(f"- **{q['id']}**: {q['text']}")
    lines.append("")

    for entry in results:
        model = entry["model"]
        lines += ["---", f"## {model}", ""]

        if entry.get("skipped"):
            lines += [f"> Skipped — {entry.get('reason', 'not available')}", ""]
            continue

        for q in QUESTIONS:
            qid = q["id"]
            lines += [f"### {qid} — {q['text']}", ""]

            for endpoint in ("chat", "generate"):
                key = f"{qid}_{endpoint}"
                r = entry["responses"].get(key, {})
                if not r:
                    continue
                lines.append(f"#### `{endpoint}` endpoint")

                if not r.get("success"):
                    lines += [f"> ⚠ **Error**: {r.get('error', 'unknown')}", ""]
                    continue

                elapsed = r.get("elapsed_s", "?")
                frame = r.get("prompt_frame", "")
                frame_note = f" · frame: {frame}" if frame else ""
                lines.append(f"*{elapsed}s elapsed{frame_note}*")
                lines.append("")

                think = r.get("think_block", "").strip()
                if think:
                    lines += [
                        "<details>",
                        "<summary>thinking trace</summary>",
                        "",
                        think,
                        "",
                        "</details>",
                        "",
                    ]

                answer = r.get("answer", r.get("raw_response", "")).strip()
                lines += [answer, ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Authoritarianism probe — models from probe_models.json"
    )
    parser.add_argument("--output-dir", default="./results/authoritarianism", type=str)
    parser.add_argument("--timeout", default=300, type=int,
                        help="Chat endpoint timeout in seconds (default: 300)")
    parser.add_argument("--generate-timeout", default=600, type=int,
                        help="Generate endpoint timeout in seconds (default: 600)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip /api/generate endpoint")
    parser.add_argument("--models", nargs="*", metavar="MODEL",
                        help="Explicit model names — overrides registry")
    parser.add_argument("--family", nargs="*", metavar="FAMILY",
                        help="Filter by family: deepseek, gemma, llama, qwen ...")
    parser.add_argument("--models-file", default=str(MODELS_FILE),
                        help=f"Path to probe_models.json (default: {MODELS_FILE})")
    parser.add_argument("--list-models", action="store_true",
                        help="List models from registry and exit")
    args = parser.parse_args()

    models_file = Path(args.models_file)

    if args.list_models:
        list_registry_models(models_file)
        return

    # Load model list from registry
    try:
        models = load_models_from_registry(
            models_file=models_file,
            families=args.family,
            names=args.models,
        )
    except FileNotFoundError as e:
        print(f"\n✗  {e}\n")
        return

    if not models:
        print("\n⚠  No models matched. Use --list-models to see available models.\n")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_timeout = args.timeout
    gen_timeout  = args.generate_timeout

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"probe_{timestamp}.json"
    md_path   = output_dir / f"probe_{timestamp}.md"

    print(f"\n{'─'*60}")
    print(f"  Authoritarianism Probe  v1.3")
    print(f"  Registry        : {models_file.name}")
    print(f"  Models          : {len(models)}")
    print(f"  Questions       : {len(QUESTIONS)}")
    print(f"  Endpoints       : chat{'' if args.no_generate else ' + generate'}")
    print(f"  chat timeout    : {chat_timeout}s")
    if not args.no_generate:
        print(f"  generate timeout: {gen_timeout}s")
    print(f"  Output          : {output_dir}")
    print(f"{'─'*60}\n")

    all_results = []

    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model}")

        if not model_is_available(model, timeout=30):
            print(f"  ⚠ Skipping — not available")
            all_results.append({"model": model, "skipped": True,
                                 "reason": "not available", "responses": {}})
            continue

        model_result = {"model": model, "skipped": False, "responses": {}}

        for q in QUESTIONS:
            print(f"  {q['id']} ", end="", flush=True)

            chat_result = query_chat(model, q["text"], chat_timeout)
            print(f"chat:{'✓' if chat_result['success'] else '✗'} ", end="", flush=True)
            model_result["responses"][f"{q['id']}_chat"] = chat_result

            if not args.no_generate:
                gen_result = query_generate(model, q["text"], gen_timeout)
                print(f"gen:{'✓' if gen_result['success'] else '✗'}", flush=True)
                model_result["responses"][f"{q['id']}_generate"] = gen_result
            else:
                print(flush=True)

            time.sleep(1)

        all_results.append(model_result)
        json_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_markdown(all_results, md_path)
        print(f"  → saved\n")

    print(f"{'─'*60}")
    print(f"  Done.")
    print(f"  JSON : {json_path}")
    print(f"  MD   : {md_path}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
