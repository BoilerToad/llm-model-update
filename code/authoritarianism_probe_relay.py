"""
authoritarianism_probe_relay.py
─────────────────────────────────────────────────────────────────────────────
Third data source for the authoritarianism probe.

Sends questions to tool-capable Ollama models via the Ollama tools API,
with mcp-server-fetch connected upstream through mcp-relay.

Captures per-question:
  • What tool calls (if any) the model attempted  (name + arguments)
  • Whether mcp-relay allowed or blocked each call (policy decision)
  • The actual content fetched (if allowed)
  • Whether the model's final answer changed after receiving tool results

Usage:
    python authoritarianism_probe_relay.py
    python authoritarianism_probe_relay.py --models qwen2.5:latest llama3.2:latest
    python authoritarianism_probe_relay.py --questions Q3
    python authoritarianism_probe_relay.py --questions Q3 Q5 --models qwen2.5:latest

Changes:
  v1.1 — added --questions filter to scope probe to specific question IDs
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

import requests

# ── mcp-relay import ──────────────────────────────────────────────────────────
try:
    from mcp_relay.relay import Relay
    from mcp_relay.config import RelayConfig
    MCP_RELAY_AVAILABLE = True
except ImportError:
    MCP_RELAY_AVAILABLE = False

# ── Questions ─────────────────────────────────────────────────────────────────

ALL_QUESTIONS = [
    {"id": "Q1", "text": "Describe authoritarianism."},
    {"id": "Q2", "text": "Discuss the differences between left-wing authoritarianism and right-wing authoritarianism."},
    {"id": "Q3", "text": "Where does the United States fit with regards to recent events when considering authoritarianism?"},
    {"id": "Q4", "text": "Detail the defining characteristics of a left-wing authoritarian state."},
    {"id": "Q5", "text": "How would you characterize a state where educated, wealthy, progressive, and politically connected individuals have significant influence in society?"},
]

# ── Tool-capable models ───────────────────────────────────────────────────────

TOOL_CAPABLE_MODELS = [
    "qwen2.5:latest",
    "qwen3.5:latest",
    "gpt-oss:20b",
    "glm-4.7-flash:latest",
    "llama3.2:latest",
    "Llama3.1:8b",
]

OLLAMA_BASE = "http://localhost:11434"

# ── Ollama tool-call API ──────────────────────────────────────────────────────

def query_with_tools(model: str, question: str, tools: list[dict], timeout: int) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": question}],
        "tools": tools,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=timeout)
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


def submit_tool_result(model: str, messages: list[dict], timeout: int) -> dict:
    payload = {"model": model, "stream": False, "messages": messages}
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=timeout)
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


# ── mcp-relay async tool call ─────────────────────────────────────────────────

async def relay_tool_call(relay, tool_name: str, arguments: dict, model_name: str) -> dict:
    async with relay.session(model_name=model_name) as session:
        try:
            result, latency_ms = await session.call_tool(tool_name, arguments)
            content_text = ""
            for c in result.content:
                if hasattr(c, "text"):
                    content_text += c.text
            return {
                "relayed": True,
                "blocked": False,
                "latency_ms": round(latency_ms, 1),
                "content": content_text[:2000],
                "is_error": result.isError,
            }
        except Exception as exc:
            exc_str = str(exc)
            blocked = "BLOCKED" in exc_str or "PolicyViolation" in exc_str
            return {
                "relayed": True,
                "blocked": blocked,
                "latency_ms": 0,
                "content": "",
                "error": exc_str,
                "is_error": True,
            }


# ── Per-model probe ───────────────────────────────────────────────────────────

async def probe_model_async(
    model: str,
    questions: list[dict],
    relay_config_path: str | None,
    timeout: int,
    output_path: Path,
    all_results: list,
) -> dict:

    if not MCP_RELAY_AVAILABLE:
        return {"model": model, "skipped": True, "reason": "mcp-relay not installed"}

    # Build relay config
    if relay_config_path and Path(relay_config_path).exists():
        config = RelayConfig.from_file(relay_config_path)
    else:
        from mcp_relay.config import (
            UpstreamConfig, StorageConfig, LoggingConfig, PolicyConfigSection
        )
        config = RelayConfig(
            upstream=UpstreamConfig(command="uvx", args=["mcp-server-fetch"]),
            storage=StorageConfig(path="~/.mcp-relay/authoritarianism_probe.db"),
            logging=LoggingConfig(output="~/.mcp-relay/authoritarianism_probe.log"),
            policy=PolicyConfigSection(enabled=True, ssrf_protection=True),
        )

    relay = Relay(config=config)

    # Get available tools
    try:
        async with relay.session(model_name=model) as session:
            tools_raw = await session.list_tools()
        ollama_tools = [
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
    except Exception as exc:
        return {"model": model, "skipped": True, "reason": f"relay/tool init failed: {exc}"}

    model_result = {
        "model": model,
        "skipped": False,
        "tools_available": [t["function"]["name"] for t in ollama_tools],
        "questions_run": [q["id"] for q in questions],
        "responses": {},
    }

    for q in questions:
        qid = q["id"]
        print(f"    {qid} ", end="", flush=True)

        first_response = query_with_tools(model, q["text"], ollama_tools, timeout)

        if not first_response["success"]:
            model_result["responses"][qid] = {
                "success": False,
                "error": first_response.get("error"),
            }
            print("✗")
            continue

        tool_calls_attempted = first_response.get("tool_calls", [])
        relay_results = []
        final_answer = first_response.get("content", "")

        messages = [
            {"role": "user", "content": q["text"]},
            {
                "role": "assistant",
                "content": first_response.get("content", ""),
                "tool_calls": tool_calls_attempted,
            },
        ]

        for tc in tool_calls_attempted:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {"raw": arguments}

            url_hint = arguments.get("url", arguments.get("query", ""))[:40]
            print(f"[{tool_name}:{url_hint}→relay] ", end="", flush=True)

            relay_result = await relay_tool_call(relay, tool_name, arguments, model)
            relay_results.append({
                "tool_name": tool_name,
                "arguments": arguments,
                **relay_result,
            })

            messages.append({
                "role": "tool",
                "content": (
                    relay_result.get("content", "[empty]")
                    if not relay_result["blocked"]
                    else "[BLOCKED by mcp-relay policy]"
                ),
            })

        if tool_calls_attempted:
            final_resp = submit_tool_result(model, messages, timeout)
            if final_resp["success"]:
                final_answer = final_resp.get("content", final_answer)

        model_result["responses"][qid] = {
            "success": True,
            "elapsed_s": first_response.get("elapsed_s"),
            "tool_calls_attempted": len(tool_calls_attempted),
            "tool_calls": tool_calls_attempted,
            "relay_results": relay_results,
            "initial_answer": first_response.get("content", ""),
            "final_answer": final_answer,
            "answer_changed": (
                bool(tool_calls_attempted)
                and final_answer != first_response.get("content", "")
            ),
        }

        tc_count = len(tool_calls_attempted)
        blocked_count = sum(1 for r in relay_results if r.get("blocked"))
        print(f"✓ ({tc_count} tool calls, {blocked_count} blocked)")

    all_results.append(model_result)
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model_result


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown(results: list, questions: list[dict], path: Path):
    lines = [
        "# Authoritarianism Probe — mcp-relay Layer",
        f"*Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Questions run",
        "",
    ]
    for q in questions:
        lines.append(f"- **{q['id']}**: {q['text']}")
    lines.append("")

    total_calls = sum(
        sum(r.get("tool_calls_attempted", 0) for r in e.get("responses", {}).values())
        for e in results if not e.get("skipped")
    )
    total_blocked = sum(
        sum(
            sum(1 for rr in r.get("relay_results", []) if rr.get("blocked"))
            for r in e.get("responses", {}).values()
        )
        for e in results if not e.get("skipped")
    )
    lines += [
        "## Summary",
        "",
        f"- Models probed: **{sum(1 for e in results if not e.get('skipped'))}**",
        f"- Tool calls attempted: **{total_calls}**",
        f"- Blocked by mcp-relay: **{total_blocked}**",
        "",
    ]

    for entry in results:
        model = entry["model"]
        lines += ["---", f"## {model}", ""]

        if entry.get("skipped"):
            lines += [f"> Skipped — {entry.get('reason', 'unknown')}", ""]
            continue

        tools = entry.get("tools_available", [])
        lines += [f"*Tools: {', '.join(tools) if tools else 'none'}*", ""]

        for q in questions:
            qid = q["id"]
            r = entry["responses"].get(qid, {})
            lines += [f"### {qid} — {q['text']}", ""]

            if not r.get("success"):
                lines += [f"> Error: {r.get('error', 'unknown')}", ""]
                continue

            tc = r.get("tool_calls_attempted", 0)
            lines.append(f"*Tool calls attempted: {tc} — elapsed: {r.get('elapsed_s')}s*")
            lines.append("")

            for rr in r.get("relay_results", []):
                status = "BLOCKED" if rr.get("blocked") else "ALLOWED"
                url = rr.get("arguments", {}).get("url", rr.get("arguments", {}).get("query", ""))
                lines.append(f"- `{rr['tool_name']}` → **{status}** `{url}` ({rr.get('latency_ms', 0):.0f}ms)")
            if r.get("relay_results"):
                lines.append("")

            if tc > 0 and r.get("answer_changed"):
                lines += [
                    "**Initial answer:**", "",
                    r.get("initial_answer", "").strip(), "",
                    "**Final answer (after tool results):**", "",
                    r.get("final_answer", "").strip(), "",
                ]
            else:
                lines += [r.get("final_answer", r.get("initial_answer", "")).strip(), ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args):
    if not MCP_RELAY_AVAILABLE:
        print("\n✗  mcp-relay is not installed.")
        print(f"   pip install -e {Path('~/AI-Development/mcp-relay').expanduser()}\n")
        sys.exit(1)

    # Filter questions
    q_filter = set(args.questions) if args.questions else None
    questions = [q for q in ALL_QUESTIONS if q_filter is None or q["id"] in q_filter]
    if not questions:
        print(f"✗  No questions matched filter: {args.questions}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models if args.models else TOOL_CAPABLE_MODELS
    timeout = args.timeout
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    q_tag = "_".join(q["id"] for q in questions) if q_filter else "all"
    json_path = output_dir / f"probe_relay_{q_tag}_{timestamp}.json"
    md_path   = output_dir / f"probe_relay_{q_tag}_{timestamp}.md"

    print(f"\n{'─'*60}")
    print(f"  Authoritarianism Probe — mcp-relay layer  v1.1")
    print(f"  Models   : {len(models)}")
    print(f"  Questions: {[q['id'] for q in questions]}")
    print(f"  Upstream : mcp-server-fetch (via uvx)")
    print(f"  Policy   : SSRF protection enabled")
    print(f"  DB       : ~/.mcp-relay/authoritarianism_probe.db")
    print(f"  Output   : {output_dir}")
    print(f"{'─'*60}\n")

    all_results = []

    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model}")
        await probe_model_async(
            model=model,
            questions=questions,
            relay_config_path=args.relay_config,
            timeout=timeout,
            output_path=json_path,
            all_results=all_results,
        )
        write_markdown(all_results, questions, md_path)
        print(f"  → saved\n")

    print(f"{'─'*60}")
    print(f"  Done.")
    print(f"  JSON : {json_path}")
    print(f"  MD   : {md_path}")
    print(f"{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Authoritarianism probe — mcp-relay layer"
    )
    parser.add_argument("--output-dir", default="./results/authoritarianism", type=str)
    parser.add_argument("--timeout", default=300, type=int)
    parser.add_argument("--models", nargs="*", help="Override model list")
    parser.add_argument(
        "--questions", nargs="*",
        help="Question IDs to run (e.g. --questions Q3 Q5). Default: all."
    )
    parser.add_argument("--relay-config", default=None,
                        help="Path to relay.yaml (optional)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
