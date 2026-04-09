"""
Analysis for Section 6.3 — Connectivity Sweep (Figure 8).

Loads raw trajectories from data/connectivity/traj_NNN.dat, computes
ensemble statistics, writes prices.dat, and produces Figure 8.

compute_prices() is generic — it works on any set of traj files that
share the same format (buyer blocks + pstar_j columns).

Run:
    python -m analysis.connectivity
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import glob
import numpy as np
import pandas as pd

from psp.metrics import print_df
from psp.plot import plot_prices_vs_percent
from psp.io import load_conf, load_traj, load_prices, write_prices

DATA = "data/connectivity"


def compute_prices(traj_files: list[str], J: int, out_path: str) -> pd.DataFrame:
    """Aggregate ensemble trajectories into prices.dat.

    Each traj file has one row per connectivity level (t = percent).
    Averages pavg_j, pstd_j, pstar_j, vtot, ctot, utot, az across
    all trials at each level, then writes prices.dat.

    Parameters
    ----------
    traj_files : list of paths to traj_NNN.dat files
    J          : number of sellers
    out_path   : where to write prices.dat

    Returns
    -------
    prices DataFrame
    """
    from collections import defaultdict
    level_data = defaultdict(list)   # t -> list of row dicts

    for path in sorted(traj_files):
        df = load_traj(path)
        I  = sum(1 for c in df.columns if c.startswith("v") and c[1:].isdigit())

        for _, row in df.iterrows():
            t = float(row["t"])
            r = {"e": t}

            for j in range(J):
                a_cols = [f"a{i}_{j}" for i in range(I)]
                a_j = np.array([float(row[c]) for c in a_cols])
                A   = float(a_j.sum())
                v_i = np.array([float(row[f"v{i}"]) for i in range(I)])
                if A > 0:
                    pavg = float((a_j * v_i).sum() / A)
                    pstd = float(np.sqrt((a_j * (v_i - pavg)**2).sum() / A))
                else:
                    pavg = pstd = 0.0
                r[f"pavg_{j}"] = pavg
                r[f"pstd_{j}"] = pstd
                r[f"plo_{j}"]  = float(row[f"plo_{j}"])
                r[f"phi_{j}"]  = float(row[f"phi_{j}"])

            r["vtot"] = sum(float(row[f"v{i}"]) for i in range(I))
            r["ctot"] = sum(float(row[f"c{i}"]) for i in range(I))
            r["utot"] = sum(float(row[f"u{i}"]) for i in range(I))
            atot = sum(float(row[f"a{i}_{j}"]) for i in range(I) for j in range(J))
            r["az"] = atot / I
            level_data[t].append(r)

    levels = sorted(level_data.keys())
    summary = []
    for t in levels:
        rows = level_data[t]
        keys = [k for k in rows[0] if k != "e"]
        avg  = {"e": t}
        for k in keys:
            avg[k] = float(np.mean([r[k] for r in rows]))
        summary.append(avg)

    write_prices(out_path, summary)
    print(f"prices.dat written → {out_path}")
    return load_prices(out_path)


def main():
    traj_files = sorted(glob.glob(f"{DATA}/traj_*.dat"))
    if not traj_files:
        print(f"No trajectory files in {DATA}/. Run experiments/connectivity.py first.")
        return

    conf = load_conf(f"{DATA}/run.conf")
    J    = conf["J"]

    prices_path = f"{DATA}/prices.dat"
    prices = compute_prices(traj_files, J, prices_path)

    P    = prices["e"].to_numpy()
    E    = np.column_stack([prices[f"pavg_{j}"].to_numpy() for j in range(J)])
    V    = np.column_stack([prices[f"pstd_{j}"].to_numpy() ** 2 for j in range(J)])
    PLO  = np.column_stack([prices[f"plo_{j}"].to_numpy() for j in range(J)])
    PHI  = np.column_stack([prices[f"phi_{j}"].to_numpy() for j in range(J)])

    # Summary table
    rows = []
    for li, p in enumerate(P):
        row = {"connectivity": f"{int(100*p)}%"}
        for j in range(J):
            row[f"plo_{j}"]  = f"{PLO[li,j]:.3f}"
            row[f"phi_{j}"]  = f"{PHI[li,j]:.3f}"
            row[f"margin_{j}"] = f"{PLO[li,j]-PHI[li,j]:.3f}"
        row["utot"] = f"{prices['utot'].iloc[li]:.3f}"
        row["az"]   = f"{prices['az'].iloc[li]:.3f}"
        rows.append(row)
    print_df(pd.DataFrame(rows), header="Margin intervals by connectivity level")

    # Figure 8 — E(p_i) with std bands + margin interval shading
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    x = 100.0 * P
    colors = [f"C{j}" for j in range(J)]

    for j in range(J):
        c = colors[j]
        label = f"Seller {j}"
        # E(p_i) with std band
        ax.plot(x, E[:, j], lw=2, color=c, label=f"{label} – E(p_i)")
        std = np.sqrt(np.maximum(V[:, j], 0.0))
        ax.fill_between(x, E[:, j] - std, E[:, j] + std, alpha=0.15, color=c)
        # margin interval bounds
        ax.plot(x, PLO[:, j], lw=1.2, color=c, linestyle="--",
                label=f"{label} – $p_j$ (lowest winner)")
        ax.plot(x, PHI[:, j], lw=1.2, color=c, linestyle=":",
                label=f"{label} – $\\bar{{p}}_j$ (highest loser)")
        ax.fill_between(x, PHI[:, j], PLO[:, j], alpha=0.25, color=c,
                        hatch="//", label=f"{label} – margin interval")

    ax.set_xlabel("% of buyers bidding in multiple auctions")
    ax.set_ylabel("Price")
    ax.set_title("Marginal Value vs % Multi-Auction Buyers")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
