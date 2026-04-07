"""
Analysis for Section 6.3 — Connectivity Sweep (Figure 8).

Loads data/connectivity/prices.dat and produces plots and summary tables.
Individual trial trajectories (traj_NNN.dat) are also available for
per-trial inspection.

Run:
    python -m analysis.connectivity
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from psp.metrics import print_df
from psp.plot import plot_prices_vs_percent
from psp.io import load_conf, load_prices, load_traj

DATA = "data/connectivity"


def main():
    conf   = load_conf(f"{DATA}/run.conf")
    prices = load_prices(f"{DATA}/prices.dat")

    J = conf["J"]
    P = prices["e"].to_numpy()

    E = np.column_stack([prices[f"pavg_{j}"].to_numpy() for j in range(J)])
    V = np.column_stack([prices[f"pstd_{j}"].to_numpy() ** 2 for j in range(J)])  # variance from std

    # prices.dat doesn't carry p* separately — use pavg as proxy, or load from traj
    # For Figure 8 we plot E(p_i) with std bands; p* can be added if saved in traj
    T = None

    # --- Summary table ---
    tbl = pd.DataFrame(
        {f"Seller {j}": [f"{prices[f'pavg_{j}'].iloc[li]:.3f} ± {prices[f'pstd_{j}'].iloc[li]:.3f}"
                         for li in range(len(P))]
         for j in range(J)},
        index=[f"{int(100*p)}%" for p in P],
    )
    print_df(tbl, header="E(p_i) ± std by connectivity level")

    # --- Figure 8 ---
    plot_prices_vs_percent(
        P, E, V, T,
        labels=[f"Seller {j}" for j in range(J)],
        title="Marginal Value vs % Multi-Auction Buyers",
    )


if __name__ == "__main__":
    main()
