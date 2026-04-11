#!/usr/bin/env bash
# upload_results.sh
# ─────────────────────────────────────────────────────────────────────────────
# Copies all authoritarianism probe JSON results to ~/Downloads so they can
# be easily dragged into Claude for analysis.
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/results/authoritarianism"
DEST="$HOME/Downloads/probe_results"

mkdir -p "$DEST"

echo ""
echo "Copying probe results to: $DEST"
echo ""

count=0
for f in "$RESULTS_DIR"/probe_*.json; do
    [ -f "$f" ] || continue
    cp "$f" "$DEST/"
    echo "  ✓ $(basename "$f")  ($(du -h "$f" | cut -f1))"
    ((count++))
done

echo ""
echo "  $count file(s) copied."
echo ""
echo "  Now drag them from ~/Downloads/probe_results/ into Claude."
echo ""
