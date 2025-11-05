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
    b = rng.uniform(*budget_range, size=I)
    qbar = rng.uniform(*q_range, size=I)
    kappa = rng.uniform(*kappa_range, size=I)

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

def _rem_gt_at(i: int, j: int, y: float, M: Dict, ladders: Dict) -> float:
    """
    Remaining capacity at seller j after removing opponents with price > y (strict '>').
    O(log I) using the ladder. Respects adjacency (i,j).
    """
    if M.get("adj") is not None and not bool(M["adj"][i, j]):
        return 0.0
    L = ladders["per_seller"][j]; p_sorted, suf = L["p"], L["suf"]
    if p_sorted.size == 0:
        return float(M["Q_max"][j])
    idx = int(np.searchsorted(p_sorted, float(y), side="right"))
    taken = float(suf[idx])
    rem = float(M["Q_max"][j]) - taken
    return rem if rem > 0.0 else 0.0

def _equals_mass_at(j: int, y: float, ladders: Dict) -> float:
    """
    Total opponents' quantity at seller j with price exactly y.
    """
    L = ladders["per_seller"][j]; p_sorted, pref = L["p"], L["pref"]
    if p_sorted.size == 0:
        return 0.0
    left  = int(np.searchsorted(p_sorted, float(y), side="left"))
    right = int(np.searchsorted(p_sorted, float(y), side="right"))
    return float(pref[right] - pref[left])

def _equals_count_at(j: int, y: float, ladders: Dict) -> int:
    """
    Number of opponents at seller j with price exactly y (for count-based QJC).
    """
    L = ladders["per_seller"][j]; p_sorted = L["p"]
    if p_sorted.size == 0:
        return 0
    left  = int(np.searchsorted(p_sorted, float(y), side="left"))
    right = int(np.searchsorted(p_sorted, float(y), side="right"))
    return int(max(0, right - left))

def seller_caps_at(i: int, y: float, M: Dict, ladders: Dict) -> np.ndarray:
    """
    Caps z_caps[j] available to buyer i at price boundary yy under a tie policy,
        split residual among equals by COUNT (approx QJC)
    """
    J = int(M["J"]); caps = np.zeros(J, dtype=float)
    for j in range(J):
        rem = _rem_gt_at(i, j, y, M, ladders)
        # Split residual among equal-price group (opponents at y plus buyer i)
        eq_cnt = 1 + _equals_count_at(j, y, ladders)
        caps[j] = (rem / float(eq_cnt)) if rem > 0.0 else 0.0
        # We cannot determine a quantity-proportional cap without q_ij.
        # Return the open cap; we'll enforce QJC proportionally after we pick q_row.
        #caps[j] = rem
    return caps

def sup_G_i_multi(i: int, M: Dict, ladders: Dict
                  ) -> Tuple[float, np.ndarray, float, Dict]:
    """
    Stepwise selection of w*: scan open intervals (y_k, y_{k+1}) for interior fixed point.
    If none, choose a boundary according to Jia Caine
    Returns: w_star, caps, z_star, meta  (meta = {'type': 'interior'|'boundary', 'y': y_star})
    """
    tol = float(M["tol"])
    steps = ladders["steps"]; w_max = float(ladders["w_max"])

    # 1) Try to find interior solution
    for k in range(steps.size - 1):
        yk, yk1 = float(steps[k]), float(steps[k + 1])
        if yk1 <= yk + tol:
            continue
        # availability is constant on (yk, yk1): evaluate at left boundary with strict '>'
        z_caps = seller_caps_at(i, yk, M, ladders)
        Z = float(np.sum(z_caps))
        w_imp = float(theta_i_prime(i, Z, M))
        if (w_imp > yk + tol) and (w_imp < yk1 - tol):
            z_star = min(float(M["qbar"][i]), Z)
            return w_imp, z_caps, z_star, {"type": "interior", "y": None}

    # 2) No interior: compute f(y_k) = θ'(Z_k) - y_k at each left boundary and locate crossing
    f_vals = []
    for k in range(steps.size - 1):  # exclude last boundary as a left boundary
        yk = float(steps[k])
        z_caps = seller_caps_at(i, yk, M, ladders)
        Zk = float(np.sum(z_caps))
        f_vals.append(float(theta_i_prime(i, Zk, M) - yk))

    # If degenerate (no opponents), take y*=0 as "lower"
    if len(f_vals) == 0:
        k_star = 0
    else:
        k_star = max([idx for idx, fv in enumerate(f_vals) if fv >= -tol], default=0)

    y_lower = float(steps[k_star])
    y_upper = float(steps[min(k_star + 1, steps.size - 1)])  # right boundary of that interval

    # Helper to build a *complete* plan at a boundary and return (w, caps, z_star, q_row, p_row, cost, u, feasible, meta)
    def eval_boundary(which: str):
        if which == "lower":
            w = y_lower
            caps = seller_caps_at(i, y_lower, M, ladders)
            meta = {"type": "boundary", "y": y_lower}
        elif which == "upper":
            # choose w just below the upper boundary; caps from open interval at y_lower
            w = max(y_upper - tol, 0.0)
            caps = seller_caps_at(i, y_lower, M, ladders)
            meta = {"type": "boundary", "y": y_upper}
        else:
            raise ValueError(which)

        z_star = min(float(M["qbar"][i]), float(np.sum(caps)))
        q_row, p_row, cost, u_new, feasible = compute_t_i_multi(
            i, M, w, caps, z_star, meta
        )
        return w, caps, z_star, q_row, p_row, cost, u_new, feasible, meta
    endpoint_policy = "upper"
    ep = str(endpoint_policy).lower()
    # legacy endpoint behaviors
    if ep == "upper":
        w_star = max(y_upper - tol, 0.0)
        caps   = seller_caps_at(i, y_lower, M, ladders)
        z_star = min(float(M["qbar"][i]), float(np.sum(caps)))
        return w_star, caps, z_star, {"type": "boundary", "y": y_upper}
    elif ep in ("lower", "qjc"):
        w_star = y_lower
        caps   = seller_caps_at(i, y_lower, M, ladders)
        z_star = min(float(M["qbar"][i]), float(np.sum(caps)))
        return w_star, caps, z_star, {"type": "boundary", "y": y_lower}
    else:
        # default to lower
        w_star = y_lower
        caps   = seller_caps_at(i, y_lower, M, ladders)
        z_star = min(float(M["qbar"][i]), float(np.sum(caps)))
        return w_star, caps, z_star, {"type": "boundary", "y": y_lower}

def _subset_uniform_split(z_star: float, caps: np.ndarray) -> np.ndarray:
    """
    Choose the smallest subset whose caps sum ≥ z_star (descending caps),
    then equal-split water-fill under per-seller caps.
    """
    J = caps.size
    order = np.argsort(-caps, kind="mergesort")
    cum = 0.0; S: List[int] = []
    for j in order:
        if caps[j] <= 0.0: continue
        S.append(int(j)); cum += float(caps[j])
        if cum >= z_star - 1e-15: break
    q_row = np.zeros(J, dtype=float)
    if z_star <= 0.0 or not S:
        return q_row
    tol_cap = 1e-12
    cap_map = {j: float(caps[j]) for j in S}
    while True:
        active = [j for j in S if q_row[j] < cap_map[j] - tol_cap]
        rem = z_star - float(np.sum(q_row))
        if rem <= tol_cap or not active:
            break
        share = rem / len(active)
        progress = 0.0
        for j in active:
            add = min(share, cap_map[j] - q_row[j])
            if add > 0.0:
                q_row[j] += add; progress += add
        if progress <= tol_cap:
            break
    return q_row

def _min_cost_split(i: int, M: Dict, z_star: float, caps: np.ndarray, ladders: Dict, y_ref: float) -> np.ndarray:
    """
    Utility-max split: choose q_row to minimize PSP cost subject to sum q = z_star and 0 ≤ q ≤ caps.
    Construct per-seller (price, length) segments up to y_ref, merge globally, and take the cheapest units first.
    """
    J = int(M["J"])
    q_row = np.zeros(J, dtype=float)
    if z_star <= 0.0:
        return q_row

    segments: List[Tuple[float, float, int]] = []
    for j in range(J):
        if caps[j] > 1e-15:
            segments.extend(_per_seller_segments_upto(i, j, float(y_ref), M, ladders))

    # Sort by price ascending (stable)
    segments.sort(key=lambda t: t[0])

    rem = float(z_star)
    for price, length, j in segments:
        if rem <= 1e-15:
            break
        room = float(caps[j]) - float(q_row[j])
        if room <= 1e-15:
            continue
        take = min(rem, room, float(length))
        if take > 0.0:
            q_row[j] += take
            rem -= take

    # Numerical clip
    q_row = np.minimum(q_row, caps)
    return q_row

def _per_seller_segments_upto(i: int, j: int, y_ref: float, M: Dict, ladders: Dict) -> List[Tuple[float, float, int]]:
    """
    Return price segments for seller j up to boundary y_ref as a list of
      (price_level, segment_length, j).
    Segment at price y_t represents the additional capacity unlocked exactly at y_t.
    Includes the y=0 segment if any capacity is available at 0.
    Respects adjacency (i,j).
    """
    if M.get("adj") is not None and not bool(M["adj"][i, j]):
        return []

    L = ladders["per_seller"][j]
    p_sorted = L["p"]

    # All local step prices up to y_ref (inclusive), plus 0
    if p_sorted.size:
        steps_j = np.unique(np.concatenate(([0.0], p_sorted[p_sorted <= float(y_ref)])))
    else:
        steps_j = np.array([0.0], dtype=float)

    segs: List[Tuple[float, float, int]] = []
    prev_rem = 0.0
    for y in steps_j:
        rem = _rem_gt_at(i, j, float(y), M, ladders)   # remaining capacity with strict '>'
        delta = rem - prev_rem
        if delta > 1e-15:
            # Units first available at price y get priced at y
            segs.append((float(y), float(delta), int(j)))
        prev_rem = rem
    return segs

def apply_budget_policy(i: int, M: Dict, q_row: np.ndarray, base_cost: float
                         ) -> Tuple[np.ndarray, float, float, bool]:
    """
    Returns (q_row_new, cost_new, u_new, feasible).
        if cost>b_i, reject
    """
    b_i = float(M["b"][i])
    Z0 = float(np.sum(q_row))

    def util_from(qv: np.ndarray, c: float) -> float:
        return theta_i(i, float(np.sum(qv)), M) - float(c)

    # Accept if already within budget (small tolerance)
    if base_cost <= b_i + 1e-9:
        return q_row, base_cost, util_from(q_row, base_cost), True

    return q_row, base_cost, util_from(q_row, base_cost), False

def compute_t_i_multi(i: int, M: Dict,
                      w_star: float,
                      caps: np.ndarray,
                      z_star: float,
                      meta: Dict,
                      ladders: Dict | None = None,
                      split_policy: str = "min_cost"
                      ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
    """
    Build (q_row, p_row) from (w_star, caps, z_star):
      split_policy ∈ {"uniform_subset", "min_cost"}
        - "uniform_subset": current bounded-participation splitter
        - "min_cost": greedy-by-price (requires ladders and a price reference)
    """
    # 1) choose a split
    if split_policy == "min_cost":
        if ladders is None:
            raise ValueError("min_cost split requires ladders")
        # y_ref: the boundary at which capacity is priced.
        #   - interior: use w_star
        #   - boundary: use meta['y'] (upper or lower step), but pricing of unlocked
        #               capacity up to that boundary still uses the revealed step prices.
        y_ref = float(w_star)
        if isinstance(meta, dict):
            y_candidate = meta.get("y", None)
            if y_candidate is not None:
                y_ref = float(y_candidate)
        q_row = _min_cost_split(i, M, float(z_star), np.asarray(caps, float), ladders, y_ref)
    else:
        q_row = _subset_uniform_split(float(z_star), np.asarray(caps, float))

    # 2) prices: one price across auctions
    p_row = np.full(int(M["J"]), float(w_star), dtype=float)

    # 3) cost & budget
    base_cost = cost_row(i, q_row, M)
    q_row, cost_new, u_new, feasible = apply_budget_policy(i, M, q_row, base_cost)
    return q_row, p_row, cost_new, u_new, feasible


def joint_best_response_plan(i: int, M: Dict) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    # read policies (with defaults)

    ladders = build_ladders(i, M)
    w_star, caps, z_star, meta = sup_G_i_multi(
        i, M, ladders)

    q_row, p_row, cost, u_new, feasible = compute_t_i_multi(
        i, M, w_star, caps, z_star, meta, ladders)
    return q_row, p_row, feasible, u_new

