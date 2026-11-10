#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PALETTE = [
    "#9400d3",  # purple
    "#009e73",  # green
    "#56b4e9",  # light blue
    "#e69f00",  # orange
]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main() -> int:
    if len(sys.argv) < 2:
        eprint("Usage: plot_results.py out/results_mauct.csv [out/out.png]")
        return 2

    csv_path = Path(sys.argv[1]).expanduser()
    out_path = Path(sys.argv[2]).expanduser() if len(sys.argv) >= 3 else Path("out/mauct_pavg_vs_conn.png")

    if not csv_path.exists():
        eprint(f"ERROR: CSV not found: {csv_path}")
        return 2

    df = pd.read_csv(csv_path)
    if df.empty:
        eprint(f"ERROR: CSV has no rows: {csv_path}")
        return 2

    # Required columns
    required = {"conn", "j", "pavg", "pstd"}
    missing = sorted(required - set(df.columns))
    if missing:
        eprint(f"ERROR: missing columns {missing} in {csv_path}")
        eprint(f"Found columns: {list(df.columns)}")
        return 2

    # Ensure numeric
    for c in ["conn", "j", "pavg", "pstd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["conn", "j", "pavg", "pstd"])
    if df.empty:
        eprint("ERROR: after numeric coercion, no valid rows remain.")
        return 2

    df = df.sort_values(["j", "conn"])

    # Debug summary (stderr so bash capture is safe)
    eprint(f"Read {len(df)} rows from {csv_path}")
    eprint(f"conn range: {df['conn'].min()} .. {df['conn'].max()}")
    eprint(f"markets (j): {sorted(df['j'].unique().tolist())}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    js = sorted(df["j"].unique())
    for idx, j in enumerate(js):
        g = df[df["j"] == j].sort_values("conn")
        x = g["conn"].to_numpy()
        y = g["pavg"].to_numpy()     # E_j
        s = g["pstd"].to_numpy()     # sqrt(V_j)

        color = PALETTE[idx % len(PALETTE)]

        ax.fill_between(x, y - s, y + s, color=color, alpha=0.20, linewidth=0, zorder=1)
        ax.plot(x, y, color=color, linewidth=2.2, zorder=2, label=rf"$\mathbb{{E}}[p^{idx+1}]$")

    ax.set_xlabel("percent bidding in next and previous auction")
    ax.set_ylabel("average marginal price")
    ax.set_title(r"Average marginal price $E_j$ with band $E_j \pm \sqrt{V_j}$")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    eprint(f"Wrote {out_path}")
    print(str(out_path))  # stdout: only the output path

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

