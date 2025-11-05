
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from market import theta_i_prime, theta_i

def local_award_for_seller(M, j, *, exclude_buyers=None):
    """
    Compute a one-shot PSP clear for seller j from the *current snapshot* in M,
    returning (awarded_vector, p_star). Pure function: does not mutate M.

    Rules:
      - Sort active bids (q>0 and (adj True if provided)) by price desc.
      - Fill to capacity Q_max[j].
      - If the marginal tier is a tie at price p*, split the remaining capacity
        proportionally to requested quantities among *only* those tied bidders.
      - Charge the threshold price p* to all served buyers.
      - If undersubscribed (sum q <= Q_max[j]): everyone gets their q, p* = 0.0.

    Parameters
    ----------
    M : dict
        Market dictionary with at least:
          I, J (ints), Q_max (J,), bid_q (I,J), bid_p (I,J).
        Optional: adj (I,J) bool mask of feasible links.
    j : int
        Seller index.
    exclude_buyers : Iterable[int] | None
        Optional set/list of buyer indices to ignore (treated as q=0).

    Returns
    -------
    a_j : np.ndarray shape (I,)
        Awarded quantities to *each buyer* at seller j.
    p_star : float
        Threshold (marginal) price for seller j; 0.0 if undersubscribed.
    """
    I = int(M["I"])
    Qj = float(np.asarray(M["Q_max"])[j])
    bid_q = np.asarray(M["bid_q"], dtype=float)[:, j].copy()
    bid_p = np.asarray(M["bid_p"], dtype=float)[:, j].copy()

    # Feasibility mask (adj) and optional exclusions
    if "adj" in M:
        mask = np.asarray(M["adj"])[:, j].astype(bool)
        bid_q[~mask] = 0.0
    if exclude_buyers:
        bid_q[np.array(list(exclude_buyers), dtype=int)] = 0.0

    # Active bids only
    idx = np.nonzero(bid_q > 0.0)[0]
    if idx.size == 0:
        return np.zeros(I, dtype=float), 0.0

    bids = [(int(i), float(bid_q[i]), float(bid_p[i])) for i in idx]
    bids.sort(key=lambda t: t[2], reverse=True)  # by price desc

    a = np.zeros(I, dtype=float)
    cum = 0.0

    # Quick undersubscription check
    total_q = sum(q for _, q, _ in bids)
    if total_q < Qj - 1e-9:
        for i, q, _ in bids:
            a[i] = q
        return a, 0.0  # no binding threshold

    # Fill strictly-above-threshold tiers
    k = 0
    while k < len(bids):
        i, q, p = bids[k]
        # Look ahead to see if next price is strictly lower (then this is a strict tier)
        next_is_lower = (k == len(bids) - 1) or (bids[k+1][2] < p - 0.0)  # exact tie only if equal
        if next_is_lower:
            # Strict tier: can we take entire q?
            if cum + q <= Qj + 1e-12:
                take = min(q, Qj - cum)
                a[i] += max(take, 0.0)
                cum += take
                if cum >= Qj - 1e-12:
                    # Filled exactly at a strict tier; p* is the price of this tier
                    return a, float(p)
                k += 1
                continue
            else:
                # We hit capacity inside this single strict bidder (rare because divisible & sorted)
                rem = max(Qj - cum, 0.0)
                a[i] += rem
                cum += rem
                return a, float(p)
        else:
            # We are at the *start* of a tie tier at price p
            p_star = p
            # collect all bidders at this tied price
            tie_start = k
            while k < len(bids) and abs(bids[k][2] - p_star) <= 0.0:
                k += 1
            tie_end = k  # slice [tie_start, tie_end) are the tied bidders at p*
            # Allocate fully all strictly-above tiers already done; now split the remainder across ties
            rem = max(Qj - cum, 0.0)
            if rem <= 1e-12:
                # Already filled before the tie — threshold determined by previous price
                return a, float(p_star)
            tie_total_q = sum(bids[t][1] for t in range(tie_start, tie_end))
            if tie_total_q <= rem + 1e-12:
                # Tie group fits entirely; allocate all of them and continue to lower prices
                for t in range(tie_start, tie_end):
                    i_t, q_t, _ = bids[t]
                    a[i_t] += q_t
                    cum += q_t
                if cum >= Qj - 1e-12:
                    return a, float(p_star)
                # Continue to next (lower) price levels; but note p* would then actually be LOWER.
                # However, by definition p* is the *marginal* threshold: if we pass the tie entirely,
                # the true threshold will be found later. So keep looping.
                continue
            else:
                # Tie-splitting at the marginal tier p*
                if tie_total_q <= 0.0:
                    return a, float(p_star)
                for t in range(tie_start, tie_end):
                    i_t, q_t, _ = bids[t]
                    share = rem * (q_t / tie_total_q)
                    a[i_t] += share
                cum = Qj
                return a, float(p_star)

    # If we exit the loop without returning, capacity was matched exactly at the last strict tier
    # Use its price as p*
    last_price = bids[min(len(bids)-1, k-1)][2]
    return a, float(last_price)

def eval_u_i_two_sellers(M, i, j0, j1, z0, z1, *, price_mode="theta_prime", w_fixed=None):
    """
    Pure counterfactual utility for buyer i given requested (z0,z1) at sellers (j0,j1).
    - Freezes opponents from current M
    - Applies same price rule
    - Runs the same local PSP award (threshold + tie-split) for j0 and j1
    - Returns utility computed from *awarded* amounts
    """
    # snapshot buyer i row
    q_row = M["bid_q"][i].copy()
    p_row = M["bid_p"][i].copy()

    try:
        w = float(theta_i_prime(i, float(z0 + z1), M))

        # set the counterfactual bids just for i at j0,j1
        M["bid_q"][i, j0] = float(min(z0, M["Q_max"][j0]))
        M["bid_q"][i, j1] = float(min(z1, M["Q_max"][j1]))
        M["bid_p"][i, j0] = w
        M["bid_p"][i, j1] = w

        a0, price0 = local_award_for_seller(M, j0)  # returns awarded to each buyer; read a0[i], price0
        a1, price1 = local_award_for_seller(M, j1)

        ai = float(a0[i] + a1[i])
        pay = float(a0[i] * price0 + a1[i] * price1)
        val = float(theta_i(i, ai, M))
        return val - pay

    finally:
        # restore
        M["bid_q"][i] = q_row
        M["bid_p"][i] = p_row



