
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from market import theta_i_prime, theta_i, a_row

# ------------------------------
# Seller-side helpers
# ------------------------------

def buyers_on_j(M: Dict, j: int):
    qcol = M["bid_q"][:, j]
    active = qcol > M["tol"]
    active &= M["adj"][:, j]
    return np.flatnonzero(active)

def seller_shell_1hop(M: Dict, j: int):
    """Sellers ℓ that share ≥1 active buyer with seller j."""
    shell = set()
    buyers = buyers_on_j(M, j)
    if buyers.size == 0:
        return shell
    for i in buyers:
        sellers_i = np.flatnonzero((M["bid_q"][i, :] > M["tol"]) & M["adj"][i, :])
        for ell in sellers_i:
            if ell != j:
                shell.add(ell)
    return shell

def pstar_j(M: Dict, j: int) -> float:
    I = M["I"]
    Qj = M["Q_max"][j]
    qcol = M["bid_q"][:, j]
    pcol = M["bid_p"][:, j]
    idx = np.nonzero(qcol > 0.0 + M["tol"])[0]
    if idx.size == 0:
        return 0.0
    bids = [(i, qcol[i], pcol[i]) for i in idx]
    bids.sort(key=lambda t: t[2], reverse=True)
    cum = 0.0
    for (i, q, p) in bids:
        if cum + q >= Qj - M["tol"]:
            return p
        cum += q
    return 0.0

def winners_on_j(M: Dict, j: int):
    """Buyers who fill up to Q_max[j] at current bids."""
    Qj = M["Q_max"][j]
    qcol = M["bid_q"][:, j]
    pcol = M["bid_p"][:, j]
    idx = np.nonzero(qcol > M["tol"])[0]
    if idx.size == 0:
        return []
    bids = [(i, qcol[i], pcol[i]) for i in idx]
    bids.sort(key=lambda t: t[2], reverse=True)
    winners = []
    cum = 0.0
    for i, q, p in bids:
        if cum + M["tol"] >= Qj:
            break
        take = min(q, Qj - cum)
        if take > 0.0:
            winners.append(i)
            cum += take
    return winners

def eval_u_i_two_sellers(M, i, j0, j1, z0, z1):
    q_row = M["bid_q"][i].copy()
    p_row = M["bid_p"][i].copy()
    try:
        w = theta_i_prime(i, z0 + z1, M)
        M["bid_q"][i, j0] = min(z0, M["Q_max"][j0])
        M["bid_q"][i, j1] = min(z1, M["Q_max"][j1])
        M["bid_p"][i, j0] = w
        M["bid_p"][i, j1] = w
        a_vec, z = a_row(i, M)
        p0 = pstar_j(M, j0)
        p1 = pstar_j(M, j1)
        pay = a_vec[j0] * p0 + a_vec[j1] * p1
        val = theta_i(i, z, M)
        return val - pay
    finally:
        M["bid_q"][i] = q_row
        M["bid_p"][i] = p_row

def active_sellers_for_i(M: Dict, i: int):
    qrow = M["bid_q"][i, :]
    mask = qrow > 0.0 + M["tol"]
    if "adj" in M:
        mask &= M["adj"][i, :].astype(bool)
    return np.nonzero(mask)[0]

import numpy as np
import matplotlib.pyplot as plt

def summarize_transition_from_arrays(P, T, metrics_list=None):
    """
    Input
      P : 1D array of percent-shared levels (same length as rows of T)
      T : 2D array (len(P) x J) of per-seller p*_j you already saved
      metrics_list : optional list of MarketMetrics (one per level)
                     If given, we also compute average z* and θ'(z*)
                     across all buyers and sellers for that level.

    Output
      p_star_avg         : mean p* across sellers at each level
      theta_prime_avg    : mean θ'(z*) (uses metrics if available;
                           otherwise equals p* as an interior-opt proxy)
      z_star_avg         : mean allocation per buyer (only if metrics given)
    """
    P = np.asarray(P)
    T = np.asarray(T)

    p_star_avg = T.mean(axis=1)

    theta_prime_avg = p_star_avg.copy()
    z_star_avg = None

    if metrics_list is not None:
        # average across buyers for each level
        z_star_avg = np.array([m.buyer_alloc.mean() for m in metrics_list])
        theta_prime_avg = np.array([m.buyer_marg.mean() for m in metrics_list])

    return p_star_avg, theta_prime_avg, z_star_avg



