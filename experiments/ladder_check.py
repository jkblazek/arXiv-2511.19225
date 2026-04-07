"""
Section 6.2 — Price-Ladder Verification (Lemma 5.2 unit test)

Writes:
  data/ladder_check/run.conf     experiment configuration
  data/ladder_check/traj_000.dat market state snapshot
  data/ladder_check/ladder.dat   ladder tuples

Run:
    python -m experiments.ladder_check
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from psp.market import make_market_multi
from psp.ladder import make_ladder_report
from psp.metrics import compute_market_metrics
from psp.io import write_conf, write_traj_header, append_traj_row

OUT = "data/ladder_check"

CONF = {
    "experiment": "ladder_check",
    "section":    "6.2",
    "I": 4, "J": 2,
    "Q_max":   [15.0, 8.0],
    "epsilon": 2.5,
    "price_tol": 5e-3,
    "bids": [
        [0, 1, 8.0, 40.0],
        [0, 0, 8.0, 40.0],
        [1, 0, 2.0,  4.0],
        [2, 0, 6.0,  1.0],
        [3, 1, 2.0,  1.2],
    ],
}


def run():
    os.makedirs(OUT, exist_ok=True)
    write_conf(f"{OUT}/run.conf", CONF)

    I, J = CONF["I"], CONF["J"]
    ell, j = 0, 1

    adj = np.zeros((I, J), dtype=bool)
    adj[0, ell] = True;  adj[0, j] = True
    adj[1, ell] = True
    adj[2, ell] = True
    adj[3, j]   = True

    M = make_market_multi(I=I, J=J, Q_max=CONF["Q_max"], epsilon=CONF["epsilon"],
                          adj=adj, price_tol=CONF["price_tol"])

    for (i, jj, q, p) in CONF["bids"]:
        M["bid_q"][i, jj] = float(q)
        M["bid_p"][i, jj] = float(p)

    # --- snapshot ---
    metrics = compute_market_metrics(M)
    traj_path = f"{OUT}/traj_000.dat"
    write_traj_header(traj_path, M)
    append_traj_row(traj_path, t=0.0, metrics=metrics)

    # --- ladder ---
    df = make_ladder_report(M)
    if not df.empty:
        df.to_csv(f"{OUT}/ladder.dat", index=False, sep="\t")

    print(f"Output written to {OUT}/")


if __name__ == "__main__":
    run()
