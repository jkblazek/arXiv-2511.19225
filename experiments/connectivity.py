"""
Section 6.3 — Connectivity Sweep (Figure 8)

Writes:
  data/connectivity/run.conf        experiment configuration
  data/connectivity/traj_NNN.dat    one trajectory per trial
  data/connectivity/prices.dat      ensemble summary (analysis input)

Run:
    python -m experiments.connectivity
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from psp.market import make_market_multi
from psp.network import run, schedule_all_buyers_stable
from psp.init import make_membership_adj_banded, reset_market_for_new_adj, record_seeds
from psp.metrics import compute_market_metrics
from psp.io import write_conf, write_traj_header, append_traj_row, write_prices, prices_row_from

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

    # prices.dat accumulates one row per (trial, level) then averaged
    # We collect all rows and average at the end.
    from collections import defaultdict
    level_rows = defaultdict(list)   # percent -> list of prices_row dicts

    for trial in range(trials):
        seed_base  = base_seed + 1000 * trial
        traj_path  = f"{OUT}/traj_{trial:03d}.dat"

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
            # trajectory: use connectivity level as t
            append_traj_row(traj_path, t=pm, metrics=metrics)
            level_rows[pm].append(prices_row_from(pm, metrics))

        print(f"Trial {trial} done → {traj_path}")

    # --- prices.dat: average over trials ---
    import numpy as np
    summary_rows = []
    for pm in percents:
        rows = level_rows[pm]
        keys = [k for k in rows[0] if k != "e"]
        avg  = {"e": pm}
        for k in keys:
            avg[k] = float(np.mean([r[k] for r in rows]))
        summary_rows.append(avg)

    write_prices(f"{OUT}/prices.dat", summary_rows)
    print(f"Summary written to {OUT}/prices.dat")


if __name__ == "__main__":
    run_experiment()
