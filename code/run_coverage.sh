#!/usr/bin/env bash
# run_coverage.sh — Run pytest with coverage and report gaps.
# Usage: bash code/run_coverage.sh [--html]
#   --html   Also generate an HTML report at results/reports/coverage/html/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$HOME/VirtualEnvs/venv-llm-model-update"
HTML_DIR="$ROOT/results/reports/coverage/html"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
else
    echo "⚠  Virtual environment not found at $VENV"
    echo "   Run: bash code/setup_venv.sh"
    exit 1
fi

cd "$ROOT"

HTML=0
for arg in "$@"; do
    [[ "$arg" == "--html" ]] && HTML=1
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  LLM Probe — Test Coverage Report"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
echo ""

# Modules under test — add new ones here as test coverage is written
SOURCES="code/probe_healthcheck.py,\
code/probe_static.py,\
code/probe_endpoint_sweep.py,\
code/probe_endpoint_test.py,\
code/probe_sweep_judge.py,\
code/probe_tool_capable.py,\
code/probe_classify_with_model.py,\
code/probe_classify.py,\
code/probe_coverage.py,\
code/probe_db.py,\
code/probe_analysis.py,\
code/probe_query_openai.py,\
code/probe_translate.py,\
code/probe_relay.py,\
code/probe_healthcheck.py,\
mlx/mlx_updater.py,\
ollama/ollama_updater.py"

if [[ $HTML -eq 1 ]]; then
    mkdir -p "$HTML_DIR"
    python -m pytest code/tests/ \
        --cov="$ROOT/code" \
        --cov-report=term-missing \
        --cov-report="html:$HTML_DIR" \
        -q
    echo ""
    echo "  HTML report written to: $HTML_DIR/index.html"
else
    python -m pytest code/tests/ \
        --cov="$ROOT/code" \
        --cov-report=term-missing \
        -q
fi

EXIT_CODE=$?
echo ""
exit $EXIT_CODE
