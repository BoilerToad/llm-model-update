"""
probe_translate.py  v1.0
─────────────────────────────────────────────────────────────────────────────
Translates non-English text in probe result JSON files to English.

Reads a probe result JSON file produced by probe_static.py, sends each
non-English response field (raw_response, answer, think_block) to a
translation model, and writes new JSON and Markdown files with the
translated content alongside the originals.

Translation is sentiment-preserving — the prompt explicitly asks the
model to stay true to the intended meaning, tone, and any ideological
framing. CCP-voice responses must arrive in English still sounding like
CCP-voice, not laundered through neutral academic language.

Translated fields added per response:
  answer_en          — English translation of the answer field
  think_block_en     — English translation of the think_block (if present)
  raw_response_en    — English translation of raw_response
  translation_model  — model used for translation
  translation_notes  — any translator notes (e.g. "already English")

Original fields are preserved unchanged.

Output files:
  <stem>_en.json
  <stem>_en.md

Usage:
    python probe_translate.py --input results/data/probes/probe_Q10c_Q12c_....json
    python probe_translate.py --input results/data/probes/probe_Q10c_Q12c_....json \\
        --translator gemma4:26b
    python probe_translate.py --input results/data/probes/probe_Q10c_Q12c_....json \\
        --translator mistral-large-3:675b-cloud --timeout 120
    python probe_translate.py --glob "results/data/probes/*chinese_lang*"
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

OLLAMA_LOCAL       = "http://localhost:11434"
OLLAMA_CLOUD       = "https://ollama.com"
DEFAULT_TRANSLATOR = "gemma3:12b"
DEFAULT_TIMEOUT    = 120

TRANSLATION_SYSTEM = (
    "You are a precise translation assistant. "
    "Translate the provided text into English. "
    "Be faithful to the original meaning, tone, sentiment, and ideological framing — "
    "do not neutralize, soften, or editorialize. "
    "If the original text uses propaganda language or official state rhetoric, "
    "translate it accurately so that framing is preserved in English. "
    "If the text is already in English, return it unchanged. "
    "Return only the translated text with no preamble, explanation, or commentary."
)

TRANSLATION_PROMPT = (
    "Translate the following text to English, preserving the original sentiment "
    "and ideological framing exactly:\n\n{text}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_cloud(model: str) -> bool:
    return "cloud" in model.lower()


def auth_headers(model: str) -> dict:
    if is_cloud(model):
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


def llm_url(model: str) -> str:
    return OLLAMA_CLOUD if is_cloud(model) else OLLAMA_LOCAL


def looks_english(text: str) -> bool:
    """
    Heuristic — returns True if text appears to already be in English.
    Checks whether the ratio of ASCII alphabetic characters to all alphabetic
    characters is high enough that translation is not needed.
    """
    if not text or not text.strip():
        return True
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return True
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    return (ascii_alpha / alpha) > 0.85


def translate(text: str, model: str, timeout: int) -> tuple[str, str]:
    """
    Translate text to English using the given model.
    Returns (translated_text, note).
    note is empty string on success, or a description on skip/error.
    """
    if not text or not text.strip():
        return text, "empty"

    if looks_english(text):
        return text, "already English"

    url     = llm_url(model)
    payload = {
        "model":  model,
        "stream": False,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM},
            {"role": "user",   "content": TRANSLATION_PROMPT.format(text=text)},
        ],
    }

    try:
        r = requests.post(
            f"{url}/api/chat",
            json=payload,
            headers=auth_headers(model),
            timeout=timeout,
        )
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "").strip()
        # Strip any think blocks the translator model may produce
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip field-based think if present (gemma4 style — already stripped by API)
        return content, ""
    except requests.exceptions.Timeout:
        return text, f"TIMEOUT after {timeout}s — original preserved"
    except Exception as e:
        return text, f"ERROR: {str(e)[:100]} — original preserved"


# ── Core translation logic ────────────────────────────────────────────────────

def translate_response(resp: dict, model: str, timeout: int) -> dict:
    """
    Translate all text fields in a single response dict.
    Returns a copy with _en fields added.
    """
    out = dict(resp)

    for field in ("answer", "think_block", "raw_response"):
        original        = resp.get(field, "") or ""
        translated, note = translate(original, model, timeout)
        out[f"{field}_en"] = translated
        if note:
            out.setdefault("translation_notes", {})[field] = note

    out["translation_model"] = model
    return out


def translate_file(
    input_path: Path,
    translator: str,
    timeout: int,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Load a probe result JSON, translate all responses, write _en.json and _en.md.
    Returns (json_out_path, md_out_path).
    """
    print(f"\n{'─'*65}")
    print(f"  Input  : {input_path.name}")
    print(f"  Model  : {translator}")
    print(f"  Timeout: {timeout}s")

    data = json.loads(input_path.read_text(encoding="utf-8"))

    total = sum(
        len(entry.get("responses", {}))
        for entry in data
        if not entry.get("skipped")
    )
    done = 0

    translated_data = []

    for entry in data:
        model_name = entry["model"]
        if entry.get("skipped"):
            translated_data.append(dict(entry))
            continue

        print(f"\n  [{model_name}]")
        new_entry = {k: v for k, v in entry.items() if k != "responses"}
        new_entry["responses"] = {}

        for key, resp in entry.get("responses", {}).items():
            done += 1
            print(f"    {key} [{done}/{total}] ", end="", flush=True)

            if not resp.get("success"):
                new_entry["responses"][key] = dict(resp)
                print("skip (failed response)")
                continue

            new_entry["responses"][key] = translate_response(resp, translator, timeout)
            notes  = new_entry["responses"][key].get("translation_notes", {})
            status = " | ".join(f"{f}:{n}" for f, n in notes.items()) if notes else "translated"
            print(status)

            time.sleep(0.5)

        translated_data.append(new_entry)

    stem     = input_path.stem
    out_dir  = output_dir or input_path.parent
    json_out = out_dir / f"{stem}_en.json"
    md_out   = out_dir / f"{stem}_en.md"

    json_out.write_text(
        json.dumps(translated_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_markdown(translated_data, input_path.name, translator, md_out)

    print(f"\n  ✓ JSON → {json_out.name}")
    print(f"  ✓ MD   → {md_out.name}")

    return json_out, md_out


# ── Markdown output ───────────────────────────────────────────────────────────

def write_markdown(
    data: list[dict],
    source_file: str,
    translator: str,
    path: Path,
):
    lines = [
        "# Probe Results — English Translation",
        f"*Source: {source_file}*",
        f"*Translated by: {translator}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "> Original text preserved alongside translations. "
        "Sentiment and ideological framing translated faithfully — "
        "CCP-voice responses remain CCP-voice in English.",
        "",
    ]

    for entry in data:
        model = entry["model"]
        lines += ["---", f"## {model}", ""]

        if entry.get("skipped"):
            lines += [f"> Skipped — {entry.get('reason', 'not available')}", ""]
            continue

        for key, resp in entry.get("responses", {}).items():
            lines += [f"### {key}", ""]

            if not resp.get("success"):
                lines += [f"> Failed: {resp.get('error', 'unknown')}", ""]
                continue

            elapsed   = resp.get("elapsed_s", "?")
            endpoint  = resp.get("endpoint", "?")
            think_fmt = resp.get("think_format", "none")
            lines.append(f"*{elapsed}s · {endpoint} · think:{think_fmt}*")
            lines.append("")

            # Think block — translated + collapsible original
            think_en   = (resp.get("think_block_en") or "").strip()
            think_orig = (resp.get("think_block")    or "").strip()
            if think_en and think_en != think_orig:
                lines += [
                    "<details>",
                    "<summary>thinking trace (translated)</summary>",
                    "",
                    think_en,
                    "",
                    "<details>",
                    "<summary>original</summary>",
                    "",
                    think_orig,
                    "",
                    "</details>",
                    "</details>",
                    "",
                ]
            elif think_en:
                lines += [
                    "<details>",
                    "<summary>thinking trace</summary>",
                    "",
                    think_en,
                    "",
                    "</details>",
                    "",
                ]

            # Answer — translation then collapsible original
            answer_en   = (resp.get("answer_en") or "").strip()
            answer_orig = (resp.get("answer")    or "").strip()
            note        = (resp.get("translation_notes") or {}).get("answer", "")

            if answer_en and answer_en != answer_orig:
                lines += [answer_en, ""]
                lines += [
                    "<details>",
                    "<summary>original (source language)</summary>",
                    "",
                    answer_orig,
                    "",
                    "</details>",
                    "",
                ]
            else:
                lines += [answer_en or answer_orig, ""]

            if note and note not in ("already English", "empty"):
                lines += [f"> ⚠ Translation note: {note}", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="probe_translate.py v1.0 — translate probe results to English"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input", metavar="PATH",
        help="Single probe result JSON file to translate",
    )
    src.add_argument(
        "--glob", metavar="PATTERN",
        help='Glob pattern for multiple files (e.g. "results/data/probes/*chinese*")',
    )

    parser.add_argument(
        "--translator", default=DEFAULT_TRANSLATOR, metavar="MODEL",
        help=f"Model to use for translation (default: {DEFAULT_TRANSLATOR})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Output directory (default: same directory as input file)",
    )

    args = parser.parse_args()

    if args.input:
        files = [Path(args.input)]
    else:
        files = [Path(p) for p in glob_module.glob(args.glob)]
        if not files:
            print(f"✗  No files matched: {args.glob}")
            return

    # JSON only — silently drop .md and any other non-JSON matches
    files = [f for f in files if f.suffix == ".json"]
    if not files:
        print("✗  No .json files matched. Try adding *.json to your glob pattern.")
        return

    # Skip already-translated files
    files = [f for f in files if not f.stem.endswith("_en")]
    if not files:
        print("✗  All matched files are already translated (_en suffix).")
        return

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  probe_translate.py  v1.0")
    print(f"  Files      : {len(files)}")
    print(f"  Translator : {args.translator}")
    print(f"  Timeout    : {args.timeout}s")
    print(f"{'═'*65}")

    for i, f in enumerate(sorted(files), 1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            translate_file(f, args.translator, args.timeout, out_dir)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'═'*65}")
    print(f"  Done — {len(files)} file(s) processed.")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
