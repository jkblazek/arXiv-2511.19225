
from typing import Dict, Tuple, List, Optional
import numpy as np

from market import theta_i, theta_i_prime

# ------------------------------
# Initialization helpers
# ------------------------------

def randomize_bids(M: Dict, seed: int ) -> None:
    I, J = M["I"], M["J"]
    # Clear any existing bids
    M["bid_q"].fill(0.0)
    M["bid_p"].fill(0.0)
    adj = M["adj"]

    def buyer_rng(i: int) -> np.random.Generator:
        # Independent per-buyer stream → stable w.r.t. other randomness
        return np.random.default_rng(np.int64(seed) + np.int64(i) * 1_000_003)

    for i in range(I):
        rng_i = buyer_rng(i)
        z_total = rng_i.uniform(0.0, M["qbar"][i])
        if z_total <= M["tol"]:
            continue
        idxs = np.flatnonzero(adj[i])
        if idxs.size == 0:
            continue

        share = z_total / idxs.size
        p_uniform = theta_i_prime(i, z_total, M)  # same price across auctions
        for j in idxs:
            M["bid_q"][i, j] = share
            M["bid_p"][i, j] = p_uniform
    return

def make_membership_adj(I: int, J: int,
                        percent_multi: float,
                        rng: Optional[np.random.Generator]) -> np.ndarray:
    adj = np.zeros((I, J), dtype=bool)
    for i in range(I):
        K = 2 if rng.random() < percent_multi else 1
        idx = rng.choice(J, size=K, replace=False)
        adj[i, idx] = True
    return adj

def record_seeds(base_seed: int,
                 *,
                 I: int,
                 J: int,
                 percents,
                 seed_model: int | None = None,
                 seed_bids: int | None = None,
                 seed_adj: int | None = None,
                 seed_sched: int | None = None,
                 verbose: bool = True) -> dict:

    seed_model = base_seed if seed_model is None else seed_model
    seed_bids  = base_seed + 17 if seed_bids is None else seed_bids
    seed_sched = base_seed + 313 if seed_sched is None else seed_sched
    seed_adj0  = base_seed if seed_adj is None else seed_adj

    seeds = {
        "I": I,
        "J": J,
        "percents": list(map(float, percents)),
        "base_seed": base_seed,
        "seed_model": seed_model,
        "seed_bids": seed_bids,
        "seed_sched": seed_sched,
        "seed_adj_base": seed_adj0,
        "seed_adj_per_level": {
            f"{int(p*100)}%": seed_adj0 + 1009 * li
            for li, p in enumerate(percents)
        }
    }
    if verbose:
        print("=== Random Seed Configuration ===")
        for k, v in seeds.items():
            if isinstance(v, dict):
                continue
            print(f"{k:18s}: {v}")
        print("Adjacency per level:")
        for label, sval in seeds["seed_adj_per_level"].items():
            print(f"  {label:>6s} → {sval}")
        print("=================================\n")
    return seeds

def reset_market_for_new_adj(M: Dict,
                             adj: np.ndarray,
                             *,
                             seed_bids: int | None = None,
                             seed_sched: int | None = None,
                             jitter: float = 0.0) -> None:
    I, J = M["I"], M["J"]
    adj = np.asarray(adj, bool)
    M["adj"] = adj
    M["bid_q"].fill(0.0)
    M["bid_p"].fill(0.0)
    if seed_bids is not None:
        randomize_bids(M, seed=seed_bids)
        #fair_bids(M)
    else:
        print("Warning: Initial bids non-deterministic!")
    M["pq"].clear()
    M["seq"] = 0
    M["t"] = 0.0
    M["gen"].fill(0)
    M["pending_posts"] = 0
    M["events_since_apply"] = 0
    if seed_sched is not None:
        M["rng"] = np.random.default_rng(seed_sched)
    else:
        print("Warning: Non-deterministic scheduling!")
    M["jitter"] = float(jitter)
