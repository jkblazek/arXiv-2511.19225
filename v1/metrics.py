
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

from market import a_row, theta_i, theta_i_prime, cost_row, integral_P_i_j
from ladder import LadderDict

@dataclass
class MarketMetrics:
    # core arrays
    a_mat: np.ndarray           # (I,J) allocations a_ij
    buyer_alloc: np.ndarray     # (I,)
    buyer_value: np.ndarray     # (I,)
    buyer_util: np.ndarray      # (I,)
    buyer_marg: np.ndarray      # (I,)
    bid_p: np.ndarray           # (I,J)
    bid_q: np.ndarray           # (I,J)
    adj:   np.ndarray           # (I,J) bool

    # quick seller totals
    alloc_j: np.ndarray         # (J,)
    revenue_j: np.ndarray       # (J,)
    Q_max: np.ndarray           # (J,)
    E_j: np.ndarray             # (J,)
    V_j: np.ndarray             # (J,)
    p_star_j: np.ndarray        # (J,)

def compute_market_metrics(M: Dict) -> MarketMetrics:
    """
    Compute per-buyer and per-seller market metrics in one pass.

    Notation:
      - a_ij: allocated quantity from seller j to buyer i (mechanism outcome)
      - p_ij: posted bid price (buyer i -> seller j)
      - E_j:  allocation-weighted average posted price ("avg_bid_j") for seller j
      - V_j:  allocation-weighted variance of posted prices for seller j
              V_j = sum_i a_ij * (p_ij - E_j)^2 / sum_i a_ij
      - revenue_j: mechanism-consistent seller revenue via \int P_i^j(a) da
      - p_star_j: mechanism-true clearing threshold for seller j (optional; for ladder checks)
    """
    I, J = int(M["I"]), int(M["J"])
    adj   = np.asarray(M["adj"], dtype=bool)
    bid_q = np.asarray(M.get("bid_q", np.zeros((I, J))), dtype=float)
    bid_p = np.asarray(M.get("bid_p", np.zeros((I, J))), dtype=float)

    a_mat       = np.zeros((I, J), dtype=float)
    buyer_alloc = np.zeros(I, dtype=float)
    buyer_value = np.zeros(I, dtype=float)
    buyer_util  = np.zeros(I, dtype=float)
    buyer_marg  = np.zeros(I, dtype=float)

    # --- Single pass over buyers to populate allocations, values, utils, marginals.
    for i in range(I):
        a_vec, z = a_row(i, M)             # allocation row and total z_i
        a_mat[i, :]    = a_vec
        buyer_alloc[i] = z
        val_i          = theta_i(i, z, M)  # ∫_0^z θ'_i(q) dq or your θ_i(i, z, M)
        buyer_value[i] = val_i
        cost_i         = cost_row(i, a_vec, M)  # ∑_j ∫_0^{a_ij} P_i^j(q) dq (your implementation)
        buyer_util[i]  = val_i - cost_i
        buyer_marg[i]  = theta_i_prime(i, z, M) if z > 0.0 else 0.0

    # --- Per-seller aggregates
    alloc_j   = a_mat.sum(axis=0)                 # A_j = ∑_i a_ij
    E_j       = np.zeros(J, dtype=float)          
    V_j       = np.zeros(J, dtype=float)          
    revenue_j = np.zeros(J, dtype=float)          
    p_star_j  = np.zeros(J, dtype=float)          

    ladder = LadderDict(M)
    for j in range(J):
        A_j = alloc_j[j]
        if A_j > 0.0:
            for i in range(I):
                a_ij = a_mat[i, j]
                if a_ij > 1e-18:
                    revenue_j[j] += float(integral_P_i_j(i, j, a_ij, M))

            # Allocation-weighted mean and variance of posted prices
            a_col = a_mat[:, j]              # weights
            p_col = bid_p[:, j]              # posted prices
            Ej = float((a_col * p_col).sum() / A_j)  # mean
            E_j[j] = Ej

            dif = p_col - Ej
            V_j[j] = float((a_col * (dif * dif)).sum() / A_j)  # variance
        else:
            E_j[j] = 0.0
            V_j[j] = 0.0
            revenue_j[j] = 0.0

        p_j = ladder._pstar(j)
        p_star_j[j] = p_j

    Q_max = M["Q_max"]

    return MarketMetrics(
        a_mat=a_mat,
        buyer_alloc=buyer_alloc,
        buyer_value=buyer_value,
        buyer_util=buyer_util,
        buyer_marg=buyer_marg,
        bid_p=bid_p,
        bid_q=bid_q,
        adj=adj,
        alloc_j=alloc_j,
        revenue_j=revenue_j,
        Q_max=Q_max,
        E_j=E_j,
        V_j=V_j,
        p_star_j=p_star_j,
    )


def market_report_from(metrics: MarketMetrics) -> pd.DataFrame:
    a_mat       = metrics.a_mat
    buyer_alloc = metrics.buyer_alloc
    buyer_value = metrics.buyer_value
    buyer_util  = metrics.buyer_util
    buyer_marg  = metrics.buyer_marg
    alloc_j     = metrics.alloc_j
    revenue_j   = metrics.revenue_j
    E_j         = metrics.E_j
    V_j         = metrics.V_j
    Q_max       = metrics.Q_max
    p_star_j    = metrics.p_star_j

    I, J = a_mat.shape

    # Attribute buyer totals to sellers proportionally to a_ij / z_i
    value_j = np.zeros(J, float)
    util_j  = np.zeros(J, float)
    mask_pos = buyer_alloc > 0
    if np.any(mask_pos):
        W = np.zeros_like(a_mat)
        # rows of W sum to 1 over sellers where a_ij>0
        W[mask_pos, :] = (a_mat[mask_pos, :].T / buyer_alloc[mask_pos]).T
        value_j = W.T @ buyer_value
        util_j  = W.T @ buyer_util

    df = pd.DataFrame({
        "seller":   np.arange(J),
        "alloc":    alloc_j,
        "value":    value_j,
        "util":     util_j,
        "revenue":  revenue_j,
        "E_j":      E_j,
        "V_j":      V_j,
        "Q_max":    Q_max,
        "p_star_j": p_star_j,
    }).round(3)

    return df

def print_round_from_metrics(metrics: MarketMetrics,
                            round_decimals: int = 3,
                            full_index: bool = True,
                            use_adj: bool = False,
                            header: str | None = None) -> pd.DataFrame:
    a_mat = metrics.a_mat
    bid_q = metrics.bid_q
    bid_p = metrics.bid_p
    adj = metrics.adj
    I, J = a_mat.shape
    base_idx = list(range(I)) if full_index else sorted(np.where(metrics.buyer_alloc> -1)[0].tolist())
    # ---------------- Buyer Totals ----------------
    totals = pd.DataFrame({
            "Buyer": np.array(base_idx, dtype=int),
            "z_i": metrics.buyer_alloc,
            "p_marg": metrics.buyer_marg,
            "p_i": metrics.avg_bid_j,
            "v_i": metrics.buyer_value,
            "u_i": metrics.buyer_util,
            "c_i": metrics.buyer_value - metrics.buyer_util,
        })
    totals = totals.set_index("Buyer").reindex(base_idx).reset_index()
    for c in totals.select_dtypes(include="number").columns:
        if c != "Buyer":
            totals[c] = totals[c].round(round_decimals)
    totals.columns = pd.MultiIndex.from_product([["Buyer Totals"], totals.columns])
    blocks = [totals]
    # ---------------- Seller blocks ----------------
    for j in range(J):
        sub = pd.DataFrame({
                "Buyer": np.array(base_idx, dtype=int),
                "q_i": bid_q[:, j],
                "p_i": bid_p[:, j],
                "a_i": a_mat[:, j],
            }).set_index("Buyer").reindex(base_idx)
        # Decide which cells are blanks
        if use_adj:
            no_edge = pd.Series([not bool(adj[b, j]) if (0 <= b < adj.shape[0]) else True for b in base_idx], index=base_idx)
            mask_blank = no_edge
        else:
            mask_blank = ((sub["q_i"].fillna(0.0) == 0.0) & (sub["p_i"].fillna(0.0) == 0.0))

        def fmt(series: pd.Series) -> pd.Series:
            s = series.copy()
            s = s.where(~mask_blank, np.nan)
            s = s.where(s.isna(), s.round(round_decimals))
            return s.astype(object).where(~s.isna(), "--")

        out = pd.DataFrame(index=sub.index)
        out["Buyer"] = out.index.astype(int)
        out["q_i"] = fmt(sub["q_i"])
        out["p_i"] = fmt(sub["p_i"])
        out["a_i"] = fmt(sub["a_i"])

        out = out.reset_index(drop=True)
        out.columns = pd.MultiIndex.from_product([[f"Seller {j}"], out.columns])
        blocks.append(out)

    wide = pd.concat(blocks, axis=1, join="outer")
    print_df(wide, header=(header or "Market snapshot"), index=False)
    return wide

def print_df(df: pd.DataFrame,
          header: Optional[str] = None,
          index: bool = False,
          max_rows: Optional[int] = 200,
          max_cols: Optional[int] = None,
          width: int = 120,
          round_decimals: Optional[int] = None) -> None:
    out = df.copy()
    if round_decimals is not None:
        num_cols = out.select_dtypes(include="number").columns
        out[num_cols] = out[num_cols].round(round_decimals)

    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", max_cols,
        "display.width", width,
        "display.expand_frame_repr", False,
    ):
        text = out.to_string(index=index)

    if header:
        print(f"\n{header}\n{text}")
    else:
        print(f"\n{text}")


def classify_buyers(M: dict, tol: float = 1e-12) -> pd.DataFrame:
    """
    Classify buyers at convergence, respecting adjacency:
      - has_pos_bid: any positive bid on a connected edge
      - zero_alloc_pos_bid: z_i == 0 but posted positive qty on some connected edge
      - zero_bid: no positive bids on any connected edge
      - zero_bid_no_gain: zero_bid AND JBR would also pick zero (on connected edges)
      - deg_i: number of connected sellers
    """
    I, J = int(M["I"]), int(M["J"])
    adj = M["adj"]

    rows = []
    for i in range(I):
        # Connected sellers for i
        conn = adj[i, :]
        deg_i = int(np.count_nonzero(conn))

        # Allocation snapshot (a_row already yields zeros for non-bids; we’ll still mask)
        a_vec, z_i = a_row(i, M)
        a_conn = np.where(conn, a_vec, 0.0)
        z_i_conn = float(np.sum(a_conn))

        # Positive bids only on connected edges
        bid_q_i = np.asarray(M["bid_q"][i, :], float)
        has_pos_bid = bool(np.any(bid_q_i[conn] > tol))
        zero_bid = (not has_pos_bid)  # only considering connected edges
        zero_alloc_pos_bid = (z_i_conn <= tol) and has_pos_bid

        # For zero-bid buyers, check if JBR would still choose zero *given connected edges*
        zero_bid_no_gain = False
        if zero_bid and deg_i > 0:
            ladders = build_ladders(i, M)  # build_ladders already honors adjacency
            _, _, z_star, _ = sup_G_i_multi(i, M, ladders)
            zero_bid_no_gain = (float(z_star) <= tol)
        elif zero_bid and deg_i == 0:
            # No connected sellers: treat as structurally out (not a failure to bid)
            zero_bid_no_gain = True

        rows.append({
            "buyer": i,
            "deg_i": deg_i,
            "z_i": float(z_i_conn),
            "has_pos_bid": has_pos_bid,
            "zero_alloc_pos_bid": zero_alloc_pos_bid,
            "zero_bid": zero_bid,
            "zero_bid_no_gain": zero_bid_no_gain,
        })
    return pd.DataFrame(rows)


def per_seller_outbid_losers(M: dict, tol: float = 1e-12) -> np.ndarray:
    """
    Per seller j: count buyers who posted positive qty on (i,j) AND are connected to j,
    but received zero allocation at j.
    """
    I, J = int(M["I"]), int(M["J"])
    adj = M["adj"]
    counts = np.zeros(J, dtype=int)
    for i in range(I):
        a_vec, _ = a_row(i, M)
        for j in range(J):
            if not adj[i, j]:
                continue
            if (M["bid_q"][i, j] > tol) and (a_vec[j] <= tol):
                counts[j] += 1
    return counts

