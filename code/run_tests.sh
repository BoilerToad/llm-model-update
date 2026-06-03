#!/usr/bin/env bash
# run_tests.sh — Run the full pytest suite and report results.
# Usage: bash code/run_tests.sh [pytest options]
# Example: bash code/run_tests.sh -x          # stop on first failure
#          bash code/run_tests.sh -k healthcheck  # run one test module

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$HOME/VirtualEnvs/venv-llm-model-update"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
else
    echo "⚠  Virtual environment not found at $VENV"
    echo "   Run: bash code/setup_venv.sh"
    exit 1
fi

cd "$ROOT"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  LLM Probe — Test Suite"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
echo ""

python -m pytest code/tests/ -v "$@"
EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  ✓ All tests passed"
else
    echo "  ✗ Some tests failed (exit code $EXIT_CODE)"
fi
echo ""

exit $EXIT_CODE
