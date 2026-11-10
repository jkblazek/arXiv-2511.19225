#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/out"
mkdir -p "$OUT_DIR"

args=("$@")
if (( ${#args[@]} == 0 )); then
  args=(0 160 10)  # start end step
fi

CSV=$("$ROOT_DIR/run_sweep.sh" "${args[@]}" | tail -n 1)

echo "CSV: $CSV" >&2
test -f "$CSV" || { echo "ERROR: CSV not found: $CSV" >&2; exit 2; }
test "$(wc -l < "$CSV")" -gt 1 || { echo "ERROR: CSV has no data rows: $CSV" >&2; head -n 5 "$CSV" >&2; exit 2; }

python3 "$ROOT_DIR/plot_results.py" "$CSV" "$OUT_DIR/mauct_pavg_vs_conn.png"

echo "Wrote: $CSV"
echo "Wrote: $OUT_DIR/mauct_pavg_vs_conn.png"

