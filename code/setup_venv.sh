#!/usr/bin/env bash
# setup_venv.sh
# ─────────────────────────────────────────────────────────────────────────────
# Creates a Python 3.12 virtual environment for llm-model-update and installs
# all dependencies.
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
#
# The venv is placed at:  ~/VirtualEnvs/venv-llm-model-update
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON_BIN="$HOME/.pyenv/versions/3.12.0/bin/python3.12"
VENV_DIR="$HOME/VirtualEnvs/venv-llm-model-update"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Preflight ─────────────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│         llm-model-update  •  venv setup                 │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

if [ ! -f "$PYTHON_BIN" ]; then
  echo "✗  Python not found at: $PYTHON_BIN"
  echo "   Run: pyenv install 3.12.0"
  exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
echo "  Python   : $PYTHON_VERSION  ($PYTHON_BIN)"
echo "  Venv     : $VENV_DIR"
echo "  Project  : $PROJECT_DIR"
echo ""

# ── Guard against collisions ──────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  echo "⚠  Venv already exists at $VENV_DIR"
  read -r -p "   Recreate it from scratch? [y/N] " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "   Removing existing venv …"
    rm -rf "$VENV_DIR"
  else
    echo "   Aborted. Existing venv untouched."
    exit 0
  fi
fi

# ── Create venv ───────────────────────────────────────────────────────────────
echo "  Creating venv …"
mkdir -p "$HOME/VirtualEnvs"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# ── Activate & install ────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "  Upgrading pip …"
pip install --upgrade pip --quiet

echo "  Installing dependencies …"
pip install -r "$PROJECT_DIR/requirements.txt"

# ── Verify installs ───────────────────────────────────────────────────────────
echo ""
echo "  Installed packages:"
pip show pyyaml huggingface_hub pytest 2>/dev/null \
  | grep -E "^(Name|Version):" \
  | paste - - \
  | awk '{printf "    %-22s %s\n", $2, $4}'

# ── Smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "  Running test suite …"
cd "$PROJECT_DIR"
python -m pytest tests/ -q --tb=short
TEST_EXIT=$?

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
if [ $TEST_EXIT -eq 0 ]; then
  echo "✓  Setup complete — all tests passed."
else
  echo "⚠  Setup complete but some tests failed (exit $TEST_EXIT)."
fi

echo ""
echo "  To activate:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Quick start:"
echo "    python ollama/ollama_updater.py --list"
echo "    python mlx/mlx_updater.py --list"
echo ""
