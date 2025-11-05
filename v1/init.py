
from typing import Dict, Tuple, List, Optional
import numpy as np

from market import theta_i, theta_i_prime

# ------------------------------
# Initialization helpers
# ------------------------------

def randomize_bids(M: Dict, seed: int ) -> None:

    I, J = int(M["I"]), int(M["J"])

    # Clear any existing bids
    M["bid_q"].fill(0.0)
    M["bid_p"].fill(0.0)

    adj = np.asarray(M["adj"], bool)

    def buyer_rng(i: int) -> np.random.Generator:
        # Independent per-buyer stream → stable w.r.t. other randomness
        return np.random.default_rng(np.int64(seed) + np.int64(i) * 1_000_003)

    for i in range(I):
        rng_i = buyer_rng(i)
        z_total = float(rng_i.uniform(0.0, float(M["qbar"][i])))
        if z_total <= 1e-15:
            continue
        idxs = np.flatnonzero(adj[i])
        if idxs.size == 0:
            continue

        share = z_total / idxs.size
        p_uniform = float(theta_i_prime(i, z_total, M))  # same price across auctions
        for j in idxs:
            M["bid_q"][i, j] = share
            M["bid_p"][i, j] = p_uniform
    reserve = M["reserve"]
    Q_max = M["Q_max"]
    #for j in J:
     #   M["bid_q"][0, j] = reserve[j]
      #  M["bid_p"][0, j] = Q_max[j]
    return

def init_bids(M: Dict, seed: int | None = None) -> None:

    I, J = int(M["I"]), int(M["J"])

    # Clear any existing bids
    M["bid_q"].fill(0.0)
    M["bid_p"].fill(0.0)

    # Deterministic: set each buyer to their joint best response
    # against current opponents (we start from zeros above).
    for i in range(I):
        # Build ladders from current state (zeros initially, then
        # progressively populated as we sweep i = 0..I-1)
        ladders = build_ladders(i, M)
        w_star, caps, z_star, meta = sup_G_i_multi(i, M, ladders)
        q_row, p_row, cost, u_new, feasible = compute_t_i_multi(
            i, M, w_star, caps, z_star, meta, ladders
        )
        # Apply directly (no event queue, no jitter)
        for j in range(J):
            # Block on adjacency
            if M.get("adj") is not None and not bool(M["adj"][i, j]):
                M["bid_q"][i, j] = 0.0
                M["bid_p"][i, j] = 0.0
            else:
                M["bid_q"][i, j] = float(q_row[j])
                M["bid_p"][i, j] = float(p_row[j])
    return

def make_membership_adj(I: int, J: int,
                        percent_multi: float,
                        w: Optional[np.ndarray] = None,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Builds an (I x J) boolean adjacency with bounded participation (degree ≤ 2).
    Assumptions:
      - J ≥ 2
      - 0 ≤ percent_multi ≤ 1
      - If w is provided, it is length-J and normalized (sum=1, nonnegative)
    Rule per buyer i:
      - With probability percent_multi → degree K=2, else K=1
      - Choose K distinct sellers without replacement using probability weights w
    """
    if rng is None:
        print("Warning: Adjacency array is non-deterministic (not reproducible)!")
        rng = np.random.default_rng()
    if w is None:
        w = np.full(J, 1.0 / J)

    adj = np.zeros((I, J), dtype=bool)
    for i in range(I):
        K = 2 if rng.random() < percent_multi else 1
        idx = rng.choice(J, size=K, replace=False, p=w)
        adj[i, idx] = True
    return adj


def reset_market_for_new_adj(M: Dict,
                             adj: np.ndarray,
                             *,
                             seed_bids: int | None = None,
                             seed_sched: int | None = None,
                             jitter: float = 0.0) -> None:
    I, J = int(M["I"]), int(M["J"])

    adj = np.asarray(adj, bool)
    M["adj"] = adj

    M["bid_q"].fill(0.0)
    M["bid_p"].fill(0.0)
    if seed_bids is not None:
        #pass
        randomize_bids(M, seed=seed_bids)
        #init_bids(M, seed=seed_bids)
    else:
        print("Warning: Initial bids non-deterministic!")
        init_bids(M)

    M["pq"].clear()
    M["seq"] = 0
    M["t"] = 0.0
    M["gen"].fill(0)
    M["pending_posts"] = 0
    M["events_since_apply"] = 0

    # 4) Deterministic scheduling, if desired
    if seed_sched is not None:
        M["rng"] = np.random.default_rng(seed_sched)
    else:
        print("Warning: Non-deterministic scheduling!")
    M["jitter"] = float(jitter)


