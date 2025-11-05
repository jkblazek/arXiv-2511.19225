
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

from market import a_row, theta_i, theta_i_prime, cost_row, integral_P_i_j
from helpers import pstar_j


@dataclass
class MarketMetrics:
    # core arrays
    a_mat: np.ndarray           # (I,J) allocations a_ij
    buyer_alloc: np.ndarray     # (I,)
    buyer_value: np.ndarray     # (I,)
    buyer_util: np.ndarray      # (I,)
    buyer_marg: np.ndarray      # (I,)
    buyer_cost: np.ndarray      # (I,)
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
    I, J = M["I"], M["J"]
    adj   = np.asarray(M["adj"], bool)
    bid_q = np.asarray(M.get("bid_q", np.zeros((I, J))), float)
    bid_p = np.asarray(M.get("bid_p", np.zeros((I, J))), float)

    a_mat       = np.zeros((I, J), float)
    buyer_alloc = np.zeros(I, float)
    buyer_value = np.zeros(I, float)
    buyer_util  = np.zeros(I, float)
    buyer_marg  = np.zeros(I, float)
    buyer_cost  = np.zeros(I, float)

    # One pass over buyers to build allocations and utilities at current bids
    for i in range(I):
        a_vec, z = a_row(i, M)                  # current allocation per seller and total
        a_mat[i, :]    = a_vec
        buyer_alloc[i] = z
        val_i          = theta_i(i, z, M)
        buyer_value[i] = val_i
        cost_i         = cost_row(i, a_vec, M)
        buyer_cost[i]  = cost_i
        buyer_util[i]  = val_i - cost_i
        buyer_marg[i]  = theta_i_prime(i, z, M) if z > 0.0 else 0.0

    alloc_j   = a_mat.sum(axis=0)
    revenue_j = np.zeros(J, float)
    E_j       = np.zeros(J, float)  # allocation-weighted average posted bid
    V_j       = np.zeros(J, float)  # allocation-weighted variance of posted bids
    p_star_j  = np.zeros(J, float)

    for j in range(J):
        A = alloc_j[j]
        if A <= 0.0:
            continue

        # Seller revenue at current PSP pricing via your integral (unchanged)
        for i in range(I):
            a_ij = a_mat[i, j]
            if a_ij > 0.0:
                revenue_j[j] += integral_P_i_j(i, j, a_ij, M)

        # Allocation-weighted average and variance of posted bid prices on seller j
        a_col = a_mat[:, j]
        p_col = bid_p[:, j]
        Ej = (a_col * p_col).sum() / A
        E_j[j] = Ej
        V_j[j] = ((a_col * (p_col - Ej) ** 2).sum()) / A

        p_star_j[j] = pstar_j(M, j)

    Q_max = M["Q_max"]

    return MarketMetrics(
        a_mat=a_mat, buyer_alloc=buyer_alloc, buyer_value=buyer_value,
        buyer_util=buyer_util, buyer_marg=buyer_marg, buyer_cost=buyer_cost, bid_p=bid_p,
        bid_q=bid_q, adj=adj, alloc_j=alloc_j, revenue_j=revenue_j, Q_max=Q_max,
        E_j=E_j, V_j=V_j, p_star_j=p_star_j,
    )

def market_report_from(metrics: MarketMetrics) -> pd.DataFrame:
    a_mat       = metrics.a_mat
    buyer_alloc = metrics.buyer_alloc
    buyer_value = metrics.buyer_value
    buyer_util  = metrics.buyer_util
    buyer_marg  = metrics.buyer_marg
    buyer_cost  = metrics.buyer_cost
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


def record_market_snapshot_from(metrics: MarketMetrics,
                                interval: float,
                                market_history: pd.DataFrame) -> pd.DataFrame:
    I, J = metrics.a_mat.shape
    rows = []
    for i in buyers_iter(M, include_reserve=False):
        z_i    = float(metrics.buyer_alloc[i])
        v_i    = float(metrics.buyer_value[i])
        c_i    = float(metrics.buyer_cost[i])
        p_marg = float(metrics.buyer_marg[i])
        for j in np.where(metrics.adj[i])[0]:
            rows.append({
                "interval": interval,
                "Seller":   int(j),
                "Buyer":    int(i),
                "q_i":      float(metrics.bid_q[i, j]),
                "p_i":      float(metrics.bid_p[i, j]),
                "a_i":      float(metrics.a_mat[i, j]),
                "z_i":      z_i,
                "p_marg":   p_marg if z_i > 0.0 else 0.0,
                "v_i":      v_i,
                "u_i":      u_i,
                "c_i":      c_i,
                "diff":     0,
            })
    if rows:
        snap = pd.DataFrame(rows)
        return pd.concat([market_history, snap], ignore_index=True)
    return market_history

def print_round_from_metrics(metrics: MarketMetrics,
                            round_decimals: int = 3,
                            full_index: bool = True,
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
            "v_i": metrics.buyer_value,
            "u_i": metrics.buyer_util,
            "c_i": metrics.buyer_cost
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
        no_edge = pd.Series([not bool(adj[b, j]) if (0 <= b < adj.shape[0]) else True for b in base_idx], index=base_idx)
        mask_blank = no_edge
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


