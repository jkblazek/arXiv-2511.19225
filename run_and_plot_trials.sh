#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/out"
mkdir -p "$OUT_DIR"

TRIALS=${1:-1}
BASE_SEED=${2:-1000}

echo "Running ensemble: trials=$TRIALS base_seed=$BASE_SEED" >&2
"$ROOT_DIR/run_ensemble.sh" "$TRIALS" "$BASE_SEED"

echo "Aggregating ensemble..." >&2
python3 "$ROOT_DIR/aggregate_ensemble.py"

ENS_CSV="$OUT_DIR/results_mauct_ensemble.csv"
test -f "$ENS_CSV" || { echo "ERROR: missing $ENS_CSV" >&2; exit 2; }
test "$(wc -l < "$ENS_CSV")" -gt 1 || { echo "ERROR: $ENS_CSV has no data rows" >&2; exit 2; }

echo "Plotting (interactive)..." >&2
python3 - "$ENS_CSV" <<'PY'
import sys
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

PALETTE = [
    "#9400d3",  # purple
    "#009e73",  # green
    "#56b4e9",  # light blue
    "#e69f00",  # orange
]

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

df = df.sort_values(["j", "conn"])

fig, ax = plt.subplots()

for j in sorted(df["j"].unique()):
    g = df[df["j"] == j]

    x = g["conn"].to_numpy()
    y = g["Ej_mean"].to_numpy()

    color = PALETTE[j % len(PALETTE)]

    ax.plot(
        x, y,
        color=color,
        linewidth=2.5,
        label=rf"$\mathbb{{E}}[p^{{{j+1}}}]$"
    )

    # Band selection (ensemble-aware)
    if "sqrtV_mean" in g.columns:
        s = g["sqrtV_mean"].to_numpy()
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18)

ax.set_xlabel("percent bidding in both auctions")
ax.set_ylabel("average marginal price")
ax.set_title(r"Average marginal price $E_j$ with band $E_j ± \sqrt{V}$")
ax.grid(False)
ax.legend()

plt.show()

PY

