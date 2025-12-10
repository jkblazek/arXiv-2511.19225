from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

# ------------------------------
# Market construction
# ------------------------------

def make_market_multi(I: int,
                      J: int,
                      Q_max: float | np.ndarray = 100.0,
                      epsilon: float = 5.0,
                      reserve: float | np.ndarray = 0.0,
                      budget_range=(1000.0, 1000.0),
                      q_range=(10.0, 60.0),
                      p_range=(10.0, 20.0),
                      kappa_range=(1.0, 3.5),
                      seed: int = 12345,
                      jitter: float = 0.01,
                      price_tol: float = 5e-3,
                      adj: Optional[np.ndarray] = None) -> Dict:
    """Create a multi-seller PSP market state as flat arrays.

    Buyer arrays (len I): b, qbar, kappa, gen
    Seller arrays (len J): Q_max, reserve
    Bid arrays (I×J): bid_q, bid_p
    """
    rng = np.random.default_rng(seed)
    b = rng_init.uniform(*budget_range, size=I)
    qbar = rng_init.uniform(*q_range, size=I)
    pbar = rng_init.uniform(*p_range, size=I)
    kappa = pbar/qbar

    Q_max = np.full(J, float(Q_max)) if np.isscalar(Q_max) else np.asarray(Q_max, dtype=float)
    assert Q_max.shape == (J,)
    reserve = np.full(J, float(reserve)) if np.isscalar(reserve) else np.asarray(reserve, dtype=float)
    assert reserve.shape == (J,)

    bid_q = np.zeros((I, J), dtype=float)
    bid_p = np.zeros((I, J), dtype=float)

    M = {
        "I": I,
        "J": J,
        "b": np.ascontiguousarray(b, dtype=float),
        "qbar": np.ascontiguousarray(qbar, dtype=float),
        "kappa": np.ascontiguousarray(kappa, dtype=float),
        "Q_max": np.ascontiguousarray(Q_max, dtype=float),
        "reserve": np.ascontiguousarray(reserve, dtype=float),
        "bid_q": np.ascontiguousarray(bid_q, dtype=float),
        "bid_p": np.ascontiguousarray(bid_p, dtype=float),
        "adj": np.ascontiguousarray(adj, dtype=bool),
        "epsilon": float(epsilon),
        # async engine
        "pq": [],
        "seq": 0,
        "t": 0.0,
        "jitter": float(jitter),
        "rng": rng,
        # buyer generations (staleness guard for POST_BID)
        "gen": np.zeros(I, dtype=np.int64),
        # tolerances
        "tol": 1e-12,
        # convergence bookkeeping
        "pending_posts": 0,
        "events_since_apply": 0,
        "price_tol": float(price_tol),
        "seller_converged": np.ones(J, dtype=bool),
    }

    return M

# ------------------------------
# Buyer valuation and derivatives
# ------------------------------

def theta_i(i: int, z: float, M: Dict) -> float:
    """θ_i(z) = κ_i * qbar_i * m − 0.5 * κ_i * m^2, with m = min(z, qbar_i)."""
    m = min(z, float(M["qbar"][i]))
    k = float(M["kappa"][i])
    return k * float(M["qbar"][i]) * m - 0.5 * k * m * m


def theta_i_prime(i: int, z: float, M: Dict) -> float:
    """θ'_i(z) = κ_i * (qbar_i − z) for z ≤ qbar_i; 0 beyond."""
    q = float(M["qbar"][i])
    k = float(M["kappa"][i])
    return k * (q - z) if z < q else 0.0

# ------------------------------
# Seller-side primitives (columnwise)
# ------------------------------

def _others_mask(i: int, I: int) -> np.ndarray:
    m = np.ones(I, dtype=bool); m[i] = False; return m

def Q_i_j(i: int, j: int, p: float, M: Dict) -> float:
    """Q_i^j(p): remaining capacity at seller j after removing opponents with price > p."""
    mask = _others_mask(i, M["I"]) ; qcol = M["bid_q"][mask, j] ; pcol = M["bid_p"][mask, j]
    rem = M["Q_max"][j] - float(np.sum(qcol[pcol > p]))
    return rem if rem > 0.0 else 0.0

def Qbar_i_j(i: int, j: int, p: float, M: Dict) -> float:
    """Q̄_i^j(p): remaining capacity at seller j after removing opponents with price ≥ p."""
    mask = _others_mask(i, M["I"]) ; qcol = M["bid_q"][mask, j] ; pcol = M["bid_p"][mask, j]
    rem = M["Q_max"][j] - float(np.sum(qcol[pcol >= p]))
    return rem if rem > 0.0 else 0.0

def P_i_j(i: int, j: int, z: float, M: Dict) -> float:
    """Price density P_i^j(z) = inf{ y ≥ 0 : Q_i^j(y) ≥ z }. Evaluate across candidate prices."""
    others = _others_mask(i, M["I"]) ; cand = np.unique(np.concatenate(([0.0], M["bid_p"][others, j])))
    for y in cand:
        if Q_i_j(i, j, float(y), M) >= z:
            return float(y)
    return float("inf")

def integral_P_i_j(i: int, j: int, a: float, M: Dict, N: int = 100) -> float:
    if a <= 0.0: return 0.0
    zs = np.linspace(0.0, a, N + 1)
    Ps = np.array([P_i_j(i, j, float(zk), M) for zk in zs])
    dz = a / N
    return float(np.trapz(Ps, dx=dz))

# ------------------------------
# Current allocation/cost/utility for buyer i (snapshot based)
# ------------------------------

def a_row(i: int, M: Dict) -> Tuple[np.ndarray, float]:
    """Return per-seller allocations a_ij for current bids, and their total."""
    J = M["J"]
    a = np.zeros(J, dtype=float)
    for j in range(J):
        q, p = float(M["bid_q"][i, j]), float(M["bid_p"][i, j])
        a[j] = min(q, Qbar_i_j(i, j, p, M))
    return a, float(np.sum(a))

def cost_row(i: int, a: np.ndarray, M: Dict) -> float:
    J = M["J"]; return float(sum(integral_P_i_j(i, j, float(a[j]), M) for j in range(J)))

def u_i_current(i: int, M: Dict) -> float:
    a, atot = a_row(i, M)
    cost = cost_row(i, a, M)
    return theta_i(i, atot, M) - cost

# ------------------------------
# Joint best response
# ------------------------------


def build_ladders(i: int, M: Dict) -> Dict:
    """
    Build per-seller price ladders of opponents for buyer i:
      - per_seller[j]: dict with fields
          'p'    : opponents' posted prices at seller j (ascending, filtered by adjacency)
          'q'    : aligned opponents' quantities
          'pref' : prefix sums of q (pref[t] = sum_{<t} q_sorted)
          'suf'  : suffix sums of q (suf[t]  = sum_{>=t} q_sorted)
      - steps: sorted global union of {0} ∪ {opponent prices across all sellers} ∪ {w_max}
      - w_max: θ'_i(0)
    """
    I, J = int(M["I"]), int(M["J"])
    others = np.ones(I, dtype=bool); others[i] = False
    has_adj = M.get("adj") is not None

    per_seller = []
    all_steps = [0.0]
    for j in range(J):
        if has_adj:
            opp_ok = M["adj"][others, j]
        else:
            opp_ok = np.ones(I - 1, dtype=bool)
        pcol = np.asarray(M["bid_p"][others, j][opp_ok], dtype=float)
        qcol = np.asarray(M["bid_q"][others, j][opp_ok], dtype=float)
        if pcol.size:
            idx = np.argsort(pcol, kind="mergesort")
            p_sorted = pcol[idx]
            q_sorted = qcol[idx]
            n = p_sorted.size
            pref = np.empty(n + 1, dtype=float); pref[0] = 0.0
            for t in range(n): pref[t + 1] = pref[t] + float(q_sorted[t])
            suf  = np.empty(n + 1, dtype=float); suf[n] = 0.0
            for t in range(n - 1, -1, -1): suf[t] = suf[t + 1] + float(q_sorted[t])
            per_seller.append({"p": p_sorted, "q": q_sorted, "pref": pref, "suf": suf})
            all_steps.extend(np.unique(p_sorted).tolist())
        else:
            per_seller.append({"p": np.array([], float),
                               "q": np.array([], float),
                               "pref": np.array([0.0], float),
                               "suf":  np.array([0.0], float)})
    w_max = float(theta_i_prime(i, 0.0, M))
    steps = np.unique(np.concatenate([np.asarray(all_steps, float), np.array([w_max], float)]))
    return {"per_seller": per_seller, "steps": steps, "w_max": w_max}


def _avail_lt_price(i: int, j: int, y: float, M: Dict, ladders: Dict) -> float:
    L = ladders["per_seller"][j]
    p = L["p"]; suf = L["suf"]
    if p.size == 0:
        return M["Q_max"][j]
    idx = np.searchsorted(p, y, side="right")
    taken = suf[idx]
    rem = M["Q_max"][j] - taken
    return max(0.0, rem)

def _avail_at_price(j: int, y: float, ladders: Dict) -> float:
    L = ladders["per_seller"][j]
    p = L["p"]; pref = L["pref"]
    if p.size == 0:
        return 0.0
    l = np.searchsorted(p, y, side="left")
    r = np.searchsorted(p, y, side="right")
    return pref[r] - pref[l]

def _count_bids_with_equal_price(j: int, y: float, ladders: Dict) -> int:
    L = ladders["per_seller"][j]
    p = L["p"]
    if p.size == 0:
        return 0
    l = np.searchsorted(p, y, side="left")
    r = np.searchsorted(p, y, side="right")
    return r - l

def avail_at_bprice(i: int, y: float, M: Dict, ladders: Dict) -> np.ndarray:
    #return avail_at_bprice_qjc(i, y, M, ladders)
    return avail_at_bprice_fair(i, y, M, ladders)

def avail_at_bprice_fair(i: int, y: float, M: Dict, ladders: Dict) -> np.ndarray:
    """
    Caps z_caps[j] available to buyer i at price boundary yy under a tie policy,
        split residual among equals by COUNT (approx QJC)
    """
    J = int(M["J"]); caps = np.zeros(J, dtype=float)
    for j in range(J):
        rem = _avail_lt_price(i, j, y, M, ladders)
        # Split residual among equal-price group (opponents at y plus buyer i)
        eq_cnt = 1 + _count_bids_with_equal_price(j, y, ladders)
        caps[j] = (rem / float(eq_cnt)) if rem > 0.0 else 0.0
        # We cannot determine a quantity-proportional cap without q_ij.
        # Return the open cap; we'll enforce QJC proportionally after we pick q_row.
    return caps

def avail_at_bprice_qjc(i: int, y: float, M: Dict, ladders: Dict) -> np.ndarray:
    """
    For each seller j: let rem be the capacity left after serving bids with p > y.
    If buyer i participates in the y-tie on seller j with quantity q_i^j (proposed or current),
    allocate to i the proportional share:  rem * q_i^j / (q_i^j + sum_{k!=i, p_k^j=y} q_k^j).
    """
    J = M["J"]
    tol = M["tol"]
    caps = np.zeros(J, float)
    for j in range(J):
        rem = _avail_lt_price(i, j, y, M, ladders)  # capacity after strictly higher prices
        qi = M["bid_q"][i, j]
        if rem <= 0.0 or qi <= 0.0:
            caps[j] = 0.0
            continue
        eq_mask = np.abs(M["bid_p"][:, j] - y) <= tol
        # sum of others at price y
        sum_others = np.sum(M["bid_q"][eq_mask, j]) - (M["bid_q"][i, j] if eq_mask[i] else 0.0)
        denom = qi + sum_others
        caps[j] = rem * qi / denom if denom > 0.0 else 0.0
    return caps

def sup_G_i_multi(i: int, M: Dict, ladders: Dict) -> Tuple[float, np.ndarray, float, Dict]:
    tol = M["tol"]
    steps = ladders["steps"]  # sorted breakpoints y_0 < y_1 < ...
    w_max = ladders["w_max"]

    # 1) Look for an interior solution on each open interval (y_k, y_{k+1}).
    #    Availability is constant on (y_k, y_{k+1}), so evaluate at y_k (strict '>').
    for k in range(len(steps) - 1):
        y_left, y_right = steps[k], steps[k + 1]
        if y_right <= y_left + tol:
            continue
        caps = avail_at_bprice(i, y_left, M, ladders)
        Z = np.sum(caps)
        w_imp = theta_i_prime(i, Z, M)
        if (w_imp > y_left + tol) and (w_imp < y_right - tol):
            z_star = min(M["qbar"][i], Z)
            return w_imp, caps, z_star, {"type": "interior"}

    # 2) No interior: take the largest left boundary y_k with theta'(Z_k) >= y_k.
    k_star = 0
    for k in range(len(steps) - 1):
        yk = steps[k]
        Zk = avail_at_bprice(i, yk, M, ladders).sum()
        if theta_i_prime(i, Zk, M) >= yk - tol:
            k_star = k

    y_lower = steps[k_star] # left boundary of that interval
    lower_caps = avail_at_bprice(i, y_lower, M, ladders)
    Z_low = np.sum(caps)
    C_low = y_lower*Z_low
    #print("[boundary] Lower boundary cost: ", C_low)

    y_upper = steps[min(k_star + 1, len(steps) - 1)]  # right boundary of that interval
    upper_caps = avail_at_bprice(i, y_upper, M, ladders)
    Z_up = np.sum(caps)
    C_up = y_upper*Z_up
    #print("[boundary] Upper boundary cost: ", C_up)

    if C_low >= C_up:
      y_star = y_lower
      z_star = min(M["qbar"][i], Z_low)
    else:
      y_star = y_upper
      z_star = min(M["qbar"][i], Z_up)

    return y_star, caps, z_star, {"type": "boundary", "y": y_star}

def _min_cost_split(i, M: Dict, z_star, caps, ladders, y_ref):
    J = M["J"]
    q = np.zeros(J, dtype=float)
    if z_star <= 0:
        return q
    segments = []
    for j in range(J):
        if caps[j] > 0:
            segments += _per_seller_segments_upto(i, j, y_ref, M, ladders)
    segments.sort(key=lambda t: t[0])
    rem = z_star
    for price, length, j in segments:
        if rem <= 0:
            break
        room = caps[j] - q[j]
        if room <= 0:
            continue
        take = min(rem, room, length)
        if take > 0:
            q[j] += take
            rem  -= take
    return np.minimum(q, caps)

def _per_seller_segments_upto(i, j, y_ref, M: Dict, ladders):
    if not M["adj"][i, j]:
        return []
    L = ladders["per_seller"][j]
    p_sorted = L["p"]
    if p_sorted.size:
        steps = np.unique(np.concatenate(([0.0], p_sorted[p_sorted <= y_ref])))
    else:
        steps = np.array([0.0])
    segs = []
    prev = 0.0
    for y in steps:
        rem = _avail_lt_price(i, j, y, M, ladders)   # remaining capacity with strict '>'
        delta = rem - prev
        if delta > 0:
            segs.append((y, delta, j))
        prev = rem
    return segs

def apply_budget_policy(i, M: Dict, q_row, base_cost):
    b = M["b"][i]
    def util(qv, c):
        return theta_i(i, np.sum(qv), M) - c
    feasible = (base_cost <= b + M["tol"])
    return q_row, base_cost, util(q_row, base_cost), feasible

def compute_t_i_multi(i, M: Dict, w_star, caps, z_star, meta, ladders):
    y_ref = meta.get("y", w_star) if isinstance(meta, dict) else w_star
    q_row = _min_cost_split(i, M, z_star, np.asarray(caps), ladders, y_ref)
    p_row = np.full(M["J"], w_star)
    base_cost = cost_row(i, q_row, M)
    q_row, cost_new, u_new, feasible = apply_budget_policy(i, M, q_row, base_cost)
    return q_row, p_row, cost_new, u_new, feasible

def joint_best_response_plan(i: int, M: Dict) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    ladders = build_ladders(i, M)
    w_star, caps, z_star, meta = sup_G_i_multi(
        i, M, ladders)
    q_row, p_row, cost, u_new, feasible = compute_t_i_multi(
        i, M, w_star, caps, z_star, meta, ladders)
    return q_row, p_row, feasible, u_new

