"""
Section 6.3 — Connectivity Sweep (Figure 8)

Writes:
  data/connectivity/run.conf        experiment configuration
  data/connectivity/traj_NNN.dat    one raw trajectory per trial

Run:
    python -m experiments.connectivity
Then:
    python -m analysis.connectivity   (computes prices.dat and Figure 8)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from psp.market import make_market_multi
from psp.network import run, schedule_all_buyers_stable
from psp.init import make_membership_adj_banded, reset_market_for_new_adj, record_seeds
from psp.metrics import compute_market_metrics
from psp.io import write_conf, write_traj_header, append_traj_row

OUT = "data/connectivity"

CONF = {
    "experiment": "connectivity",
    "section":    "6.3",
    "I":         8,
    "J":         2,
    "Q_max":     [60.0, 40.0],
    "epsilon":   5.0,
    "reserve":   [0.0, 0.0],
    "steps":     2000,
    "base_seed": 20405008,
    "percents":  [x / 100.0 for x in range(0, 101, 5)],
    "trials":    1,
}


def run_experiment():
    os.makedirs(OUT, exist_ok=True)
    write_conf(f"{OUT}/run.conf", CONF)
    record_seeds(CONF["base_seed"], I=CONF["I"], J=CONF["J"],
                 percents=CONF["percents"], verbose=True)

    I         = CONF["I"]
    J         = CONF["J"]
    Q         = np.asarray(CONF["Q_max"], float)
    R         = np.asarray(CONF["reserve"], float)
    percents  = CONF["percents"]
    steps     = CONF["steps"]
    base_seed = CONF["base_seed"]
    trials    = CONF["trials"]

    for trial in range(trials):
        seed_base = base_seed + 1000 * trial
        traj_path = f"{OUT}/traj_{trial:03d}.dat"

        M = make_market_multi(I, J, Q_max=Q, epsilon=CONF["epsilon"], reserve=R,
                              seed=seed_base, adj=np.ones((I, J), dtype=bool))
        write_traj_header(traj_path, M)

        for li, pm in enumerate(percents):
            adj = make_membership_adj_banded(I, J, pm)
            reset_market_for_new_adj(
                M, adj,
                seed_bids=seed_base + 17 + 50 * li,
                seed_sched=seed_base + 313 + 50 * li,
            )
            M["deterministic_sched"] = True
            schedule_all_buyers_stable(M, t0=0.0, seed_order=42)
            run(M, steps=steps, verbose=False)

            metrics = compute_market_metrics(M)
            append_traj_row(traj_path, t=pm, metrics=metrics)

        print(f"Trial {trial} done → {traj_path}")


if __name__ == "__main__":
    run_experiment()
