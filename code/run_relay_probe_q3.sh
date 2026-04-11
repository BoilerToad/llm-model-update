#!/usr/bin/env bash
# run_relay_probe_q3.sh
# ─────────────────────────────────────────────────────────────────────────────
# Preflight checks then launches the relay probe scoped to Q3 only,
# across all tool-capable Ollama models.
#
# Run from anywhere:
#   chmod +x ~/AI-Development/llm-model-update/run_relay_probe_q3.sh
#   ~/AI-Development/llm-model-update/run_relay_probe_q3.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RELAY_DIR="$HOME/AI-Development/mcp-relay"
PROBE="$HOME/AI-Development/llm-model-update/authoritarianism_probe_relay.py"
VENV="$RELAY_DIR/.venv"

echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│   relay probe — Q3 agentic behavior comparison          │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────

echo "  Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "  ✗  Ollama not running. Start with: ollama serve"
  exit 1
fi
MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']))")
echo "  ✓  Ollama running ($MODEL_COUNT models)"

echo "  Checking mcp-server-fetch..."
if ! uvx mcp-server-fetch --help > /dev/null 2>&1; then
  echo "  ✗  mcp-server-fetch not available. Install with: pip install mcp-server-fetch"
  echo "     or ensure uvx is on PATH"
  exit 1
fi
echo "  ✓  mcp-server-fetch available via uvx"

echo "  Checking mcp-relay venv..."
if [ ! -f "$VENV/bin/python" ]; then
  echo "  ✗  No venv at $VENV"
  exit 1
fi
if ! "$VENV/bin/python" -c "from mcp_relay.relay import Relay" 2>/dev/null; then
  echo "  ✗  mcp-relay not importable from $VENV"
  exit 1
fi
echo "  ✓  mcp-relay importable"

echo "  Checking relay DB path..."
mkdir -p "$HOME/.mcp-relay"
echo "  ✓  ~/.mcp-relay/ ready"

echo ""
echo "  Tool-capable models to probe:"
MODELS=(
  "qwen2.5:latest"
  "qwen3.5:latest"
  "gpt-oss:20b"
  "glm-4.7-flash:latest"
  "llama3.2:latest"
  "Llama3.1:8b"
)
for m in "${MODELS[@]}"; do
  echo "    • $m"
done

echo ""
echo "  Question: Q3 — Where does the United States fit with regards"
echo "            to recent events when considering authoritarianism?"
echo ""
echo "  The probe will:"
echo "    1. Offer mcp-server-fetch tools to each model"
echo "    2. Record any fetch/tool calls attempted"
echo "    3. Route each call through mcp-relay (SSRF policy active)"
echo "    4. Capture final answer with and without tool results"
echo "    5. Write JSON + Markdown to results/authoritarianism/"
echo ""
read -r -p "  Ready to run? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "  Aborted."
  exit 0
fi

echo ""
echo "  Starting relay monitor in background (tail: ~/.mcp-relay/authoritarianism_probe.log)"
tail -f "$HOME/.mcp-relay/authoritarianism_probe.log" 2>/dev/null &
TAIL_PID=$!
trap "kill $TAIL_PID 2>/dev/null; echo ''" EXIT

# ── Run ───────────────────────────────────────────────────────────────────────

echo ""
echo "  ── Running probe ────────────────────────────────────────"
echo ""

"$VENV/bin/python" "$PROBE" \
  --models "${MODELS[@]}" \
  --output-dir "$HOME/AI-Development/llm-model-update/results/authoritarianism" \
  --timeout 300

echo ""
echo "  ── Done ─────────────────────────────────────────────────"
echo ""
echo "  Results written to:"
echo "    ~/AI-Development/llm-model-update/results/authoritarianism/"
echo ""
echo "  Relay DB at:  ~/.mcp-relay/authoritarianism_probe.db"
echo "  Query it:     sqlite3 ~/.mcp-relay/authoritarianism_probe.db"
echo ""
