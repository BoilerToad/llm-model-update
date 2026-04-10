"""
probe_relay.py
─────────────────────────────────────────────────────────────────────────────
Relay probe — queries tool-capable Ollama models via /api/chat with tools,
routing all tool calls through mcp-relay (mcp-server-fetch upstream).

Questions and models loaded from JSON files; all selection at runtime.

Usage examples:
    # List available questions
    python probe_relay.py --list-questions

    # List tool-capable models
    python probe_relay.py --list-models

    # Q3 only across all tool-capable models
    python probe_relay.py --questions Q03

    # Full trade theme on Qwen and Llama families
    python probe_relay.py --theme trade --family qwen llama

    # Specific questions + specific models
    python probe_relay.py --questions Q03 Q10 Q15 --models qwen2.5:latest gpt-oss:20b llama3.3:latest

    # Skip models with known tool bugs
    python probe_relay.py --questions Q03 --skip-buggy

Configuration:
    --questions-file   Path to questions JSON  (default: probes/questions.json)
    --models-file      Path to models JSON     (default: probes/probe_models.json)
    --output-dir       Results directory        (default: results/probes)
    --timeout          Per-request timeout      (default: 300s)
    --relay-config     Path to relay.yaml       (optional)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import os

import requests
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

try:
    from mcp_relay.relay import Relay
    from mcp_relay.config import (
        RelayConfig, UpstreamConfig, StorageConfig,
        LoggingConfig, PolicyConfigSection,
    )
    MCP_RELAY_AVAILABLE = True
except ImportError:
    MCP_RELAY_AVAILABLE = False

OLLAMA_LOCAL = "http://localhost:11434"
OLLAMA_CLOUD = "https://ollama.com"


def llm_url_for(model_def: dict) -> str:
    return OLLAMA_CLOUD if model_def.get("cloud") else OLLAMA_LOCAL


def auth_headers(llm_url: str) -> dict:
    if llm_url == OLLAMA_CLOUD:
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}
    return {}
DEFAULT_QUESTIONS_FILE = Path(__file__).parent / "probes" / "questions.json"
DEFAULT_MODELS_FILE    = Path(__file__).parent / "probes" / "probe_models.json"
DEFAULT_OUTPUT_DIR     = Path(__file__).parent / "results" / "probes"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text())["questions"]


def load_models(path: Path) -> list[dict]:
    return json.loads(path.read_text())["models"]


def filter_questions(
    questions: list[dict],
    ids: list[str] | None,
    theme: str | None,
    tags: list[str] | None,
    all_questions: bool,
) -> list[dict]:
    if all_questions:
        return questions
    if ids:
        id_set = set(ids)
        return [q for q in questions if q["id"] in id_set]
    result = questions
    if theme:
        result = [q for q in result if q.get("theme") == theme]
    if tags:
        tag_set = set(tags)
        result = [q for q in result if tag_set & set(q.get("tags", []))]
    return result


def filter_models(
    models: list[dict],
    names: list[str] | None,
    families: list[str] | None,
    skip_buggy: bool,
) -> list[dict]:
    result = [m for m in models if m.get("enabled", True) and m.get("tool_capable")]
    if names:
        result = [m for m in result if m["name"] in names]
    if families:
        result = [m for m in result if m.get("family") in families]
    if skip_buggy:
        result = [m for m in result if not m.get("type_coercion_bug")]
    return result


# ── Ollama tool-call API ──────────────────────────────────────────────────────

def query_with_tools(model: str, question: str, tools: list[dict], timeout: int,
                     llm_url: str = OLLAMA_LOCAL) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": question}],
        "tools": tools,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{llm_url}/api/chat", json=payload,
                          headers=auth_headers(llm_url), timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        return {
            "success": True,
            "elapsed_s": elapsed,
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", []),
            "eval_count": data.get("eval_count"),
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2)}


def submit_tool_result(model: str, messages: list[dict], timeout: int,
                       llm_url: str = OLLAMA_LOCAL) -> dict:
    payload = {"model": model, "stream": False, "messages": messages}
    t0 = time.time()
    try:
        r = requests.post(f"{llm_url}/api/chat", json=payload,
                          headers=auth_headers(llm_url), timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        return {
            "success": True,
            "elapsed_s": elapsed,
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", []),
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "elapsed_s": round(time.time() - t0, 2)}


# ── mcp-relay ─────────────────────────────────────────────────────────────────

def build_relay(relay_config_path: str | None) -> "Relay":
    if relay_config_path and Path(relay_config_path).exists():
        config = RelayConfig.from_file(relay_config_path)
    else:
        config = RelayConfig(
            upstream=UpstreamConfig(command="uvx", args=["mcp-server-fetch"]),
            storage=StorageConfig(path="~/.mcp-relay/probe_relay.db"),
            logging=LoggingConfig(output="~/.mcp-relay/probe_relay.log"),
            policy=PolicyConfigSection(enabled=True, ssrf_protection=True),
        )
    return Relay(config=config)


async def relay_tool_call(relay, tool_name: str, arguments: dict, model_name: str) -> dict:
    async with relay.session(model_name=model_name) as session:
        try:
            result, latency_ms = await session.call_tool(tool_name, arguments)
            content_text = "".join(
                c.text for c in result.content if hasattr(c, "text")
            )
            return {
                "relayed": True,
                "blocked": False,
                "latency_ms": round(latency_ms, 1),
                "content": content_text[:2000],
                "is_error": result.isError,
            }
        except Exception as exc:
            exc_str = str(exc)
            return {
                "relayed": True,
                "blocked": "BLOCKED" in exc_str or "PolicyViolation" in exc_str,
                "latency_ms": 0,
                "content": "",
                "error": exc_str,
                "is_error": True,
            }


async def get_ollama_tools(relay, model_name: str) -> list[dict]:
    async with relay.session(model_name=model_name) as session:
        tools_raw = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema if hasattr(t, "inputSchema") else {},
            }
        }
        for t in tools_raw
    ]


# ── Per-model probe ───────────────────────────────────────────────────────────

async def probe_model(
    model: str,
    questions: list[dict],
    relay,
    timeout: int,
    output_path: Path,
    all_results: list,
    llm_url: str = OLLAMA_LOCAL,
) -> dict:

    try:
        ollama_tools = await get_ollama_tools(relay, model)
    except Exception as exc:
        entry = {"model": model, "skipped": True,
                 "reason": f"tool init failed: {exc}", "responses": {}}
        all_results.append(entry)
        output_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return entry

    model_result = {
        "model": model,
        "skipped": False,
        "llm_url": llm_url,
        "tools_available": [t["function"]["name"] for t in ollama_tools],
        "questions_run": [q["id"] for q in questions],
        "responses": {},
    }

    for q in questions:
        qid = q["id"]
        print(f"    {qid} ", end="", flush=True)

        first = query_with_tools(model, q["text"], ollama_tools, timeout, llm_url=llm_url)
        if not first["success"]:
            model_result["responses"][qid] = {
                "success": False, "error": first.get("error")
            }
            print("✗")
            continue

        tool_calls = first.get("tool_calls", [])
        relay_results = []
        final_answer = first.get("content", "")

        messages = [
            {"role": "user", "content": q["text"]},
            {"role": "assistant", "content": first.get("content", ""),
             "tool_calls": tool_calls},
        ]

        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {"raw": arguments}

            url_hint = arguments.get("url", arguments.get("query", ""))[:35]
            print(f"[{tool_name}:{url_hint}] ", end="", flush=True)

            rr = await relay_tool_call(relay, tool_name, arguments, model)
            relay_results.append({"tool_name": tool_name, "arguments": arguments, **rr})

            messages.append({
                "role": "tool",
                "content": (
                    rr.get("content", "[empty]")
                    if not rr["blocked"]
                    else "[BLOCKED by mcp-relay policy]"
                ),
            })

        if tool_calls:
            final_resp = submit_tool_result(model, messages, timeout, llm_url=llm_url)
            if final_resp["success"]:
                final_answer = final_resp.get("content", final_answer)

        model_result["responses"][qid] = {
            "success": True,
            "elapsed_s": first.get("elapsed_s"),
            "tool_calls_attempted": len(tool_calls),
            "tool_calls": tool_calls,
            "relay_results": relay_results,
            "initial_answer": first.get("content", ""),
            "final_answer": final_answer,
            "answer_changed": (
                bool(tool_calls)
                and final_answer != first.get("content", "")
            ),
        }

        blocked = sum(1 for r in relay_results if r.get("blocked"))
        print(f"✓ ({len(tool_calls)} calls, {blocked} blocked)")

    all_results.append(model_result)
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model_result


# ── Output ─────────────────────────────────────────────────────────────────────

def write_markdown(results: list, questions: list[dict], path: Path):
    lines = [
        "# Relay Probe Results",
        f"*Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Questions",
        "",
    ]
    for q in questions:
        lines.append(f"- **{q['id']}** [{q['theme']}] — {q['text']}")
    lines.append("")

    total_calls = sum(
        sum(r.get("tool_calls_attempted", 0) for r in e.get("responses", {}).values())
        for e in results if not e.get("skipped")
    )
    blocked_total = sum(
        sum(
            sum(1 for rr in r.get("relay_results", []) if rr.get("blocked"))
            for r in e.get("responses", {}).values()
        )
        for e in results if not e.get("skipped")
    )
    lines += [
        "## Summary", "",
        f"- Models: **{sum(1 for e in results if not e.get('skipped'))}**",
        f"- Tool calls: **{total_calls}**",
        f"- Blocked: **{blocked_total}**",
        "",
    ]

    for entry in results:
        model = entry["model"]
        lines += ["---", f"## {model}", ""]
        if entry.get("skipped"):
            lines += [f"> Skipped — {entry.get('reason','unknown')}", ""]
            continue
        tools = entry.get("tools_available", [])
        lines += [f"*Tools: {', '.join(tools)}*", ""]

        for q in questions:
            qid = q["id"]
            r = entry["responses"].get(qid, {})
            lines += [f"### {qid} — {q['text']}", ""]
            if not r.get("success"):
                lines += [f"> Error: {r.get('error','unknown')}", ""]
                continue
            tc = r.get("tool_calls_attempted", 0)
            lines.append(f"*{tc} tool calls · {r.get('elapsed_s')}s*")
            lines.append("")
            for rr in r.get("relay_results", []):
                status = "BLOCKED" if rr.get("blocked") else "ALLOWED"
                url = rr.get("arguments", {}).get("url",
                      rr.get("arguments", {}).get("query", ""))
                lines.append(
                    f"- `{rr['tool_name']}` → **{status}** `{url}` "
                    f"({rr.get('latency_ms',0):.0f}ms · {len(rr.get('content',''))}c)"
                )
            if r.get("relay_results"):
                lines.append("")
            if tc > 0 and r.get("answer_changed"):
                lines += ["**Initial:**", "", r.get("initial_answer","").strip(), "",
                           "**Final (after tool results):**", "",
                           r.get("final_answer","").strip(), ""]
            else:
                lines += [r.get("final_answer", r.get("initial_answer","")).strip(), ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── List helpers ───────────────────────────────────────────────────────────────

def list_questions(questions: list[dict]):
    themes = {}
    for q in questions:
        themes.setdefault(q["theme"], []).append(q)
    for theme, qs in themes.items():
        print(f"\n  [{theme}]")
        for q in qs:
            print(f"    {q['id']}  {q['text'][:72]}")
            print(f"         tags: {', '.join(q.get('tags',[]))}")


def list_models(models: list[dict]):
    capable = [m for m in models if m.get("tool_capable") and m.get("enabled", True)]
    buggy   = [m for m in capable if m.get("type_coercion_bug")]
    clean   = [m for m in capable if not m.get("type_coercion_bug")]
    print(f"\n  Tool-capable, no bugs ({len(clean)}):")
    for m in clean:
        print(f"    {m['name']:<45} [{m['family']}]  {m['size_gb']}GB")
    if buggy:
        print(f"\n  Tool-capable but type-coercion bug ({len(buggy)}) — use --skip-buggy:")
        for m in buggy:
            print(f"    {m['name']:<45} [{m['family']}]")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args):
    if not MCP_RELAY_AVAILABLE:
        print("\n✗  mcp-relay not importable.")
        print(f"   pip install -e {Path('~/AI-Development/mcp-relay').expanduser()}\n")
        sys.exit(1)

    all_questions = load_questions(Path(args.questions_file))
    all_model_defs = load_models(Path(args.models_file))
    model_lookup = {m["name"]: m for m in all_model_defs}

    if args.list_questions:
        list_questions(all_questions)
        return
    if args.list_models:
        list_models(all_model_defs)
        return

    questions = filter_questions(
        all_questions,
        ids=args.questions,
        theme=args.theme,
        tags=args.tag,
        all_questions=args.all_questions,
    )
    if not questions:
        print("No questions matched. Use --list-questions.")
        sys.exit(1)

    if args.models:
        model_names = args.models
    else:
        model_defs = filter_models(
            all_model_defs,
            names=None,
            families=args.family,
            skip_buggy=args.skip_buggy,
        )
        model_names = [m["name"] for m in model_defs]

    if not model_names:
        print("No models matched. Use --list-models.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_ids = "_".join(q["id"] for q in questions[:6])
    if len(questions) > 6:
        q_ids += f"_plus{len(questions)-6}"
    label = f"_{args.label}" if args.label else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"probe_relay_{q_ids}{label}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path   = output_dir / f"{stem}.md"

    print(f"\n{'─'*65}")
    print(f"  Relay Probe  v2.0")
    print(f"  Questions : {len(questions)}  ({', '.join(q['id'] for q in questions)})")
    print(f"  Models    : {len(model_names)}")
    print(f"  Upstream  : mcp-server-fetch (via uvx)")
    print(f"  Policy    : SSRF protection enabled")
    print(f"  DB        : ~/.mcp-relay/probe_relay.db")
    print(f"  Output    : {output_dir}")
    print(f"{'─'*65}\n")

    relay = build_relay(args.relay_config)
    all_results = []

    for i, model in enumerate(model_names, 1):
        model_def = model_lookup.get(model, {"cloud": "cloud" in model.lower()})
        llm_url   = llm_url_for(model_def)
        endpoint_label = "cloud" if llm_url == OLLAMA_CLOUD else "local"
        print(f"[{i}/{len(model_names)}] {model}  [{endpoint_label}]")
        await probe_model(
            model=model,
            questions=questions,
            relay=relay,
            timeout=args.timeout,
            output_path=json_path,
            all_results=all_results,
            llm_url=llm_url,
        )
        write_markdown(all_results, questions, md_path)
        print(f"  → saved\n")

    print(f"{'─'*65}")
    print(f"  Done.")
    print(f"  JSON : {json_path}")
    print(f"  MD   : {md_path}")
    print(f"{'─'*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Relay probe — tool-capable models through mcp-relay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    qg = parser.add_argument_group("question selection")
    qg.add_argument("--questions", nargs="+", metavar="ID")
    qg.add_argument("--theme", metavar="THEME")
    qg.add_argument("--tag", nargs="+", metavar="TAG")
    qg.add_argument("--all-questions", action="store_true")

    mg = parser.add_argument_group("model selection")
    mg.add_argument("--models", nargs="+", metavar="MODEL")
    mg.add_argument("--family", nargs="+", metavar="FAMILY")
    mg.add_argument("--skip-buggy", action="store_true",
                    help="Exclude models with known type-coercion tool bugs (Llama3.1)")

    cfg = parser.add_argument_group("configuration")
    cfg.add_argument("--questions-file", default=str(DEFAULT_QUESTIONS_FILE))
    cfg.add_argument("--models-file", default=str(DEFAULT_MODELS_FILE))
    cfg.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    cfg.add_argument("--timeout", default=300, type=int)
    cfg.add_argument("--relay-config", default=None)
    cfg.add_argument("--label", default="")

    info = parser.add_argument_group("information")
    info.add_argument("--list-questions", action="store_true")
    info.add_argument("--list-models", action="store_true")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
