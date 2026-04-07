"""
Analysis for Section 6.2 — Price-Ladder Verification (Lemma 5.2).

Lemma 5.2 states that in a saturated one-hop shell, for any tuple
(ℓ, k, j, i) where buyer i wins at seller j, buyer k wins only at ℓ,
and seller ℓ is in the 1-hop shell of j, the marginal thresholds nest:

    p*_ℓ  ≤  p_k  <  p*_j  ≤  p_i

This script loads data/ladder_check/ladder.dat and reports:
  - Each tuple with its four prices
  - The three margins between successive prices
  - A per-tuple and overall pass/fail verdict

Run:
    python -m analysis.ladder_check
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from psp.market import make_market_multi
from psp.metrics import print_df, print_round_from_metrics, compute_market_metrics, market_report_from
from psp.plot import (plot_connectivity, plot_buyer_diagnostics,
                      plot_shared_buyer_surface_z0z1, plot_ladder_tuple)
from psp.io import load_conf, load_traj

DATA = "data/ladder_check"


def report_lemma_52(df: pd.DataFrame) -> None:
    """Print a structured verification report for Lemma 5.2."""
    print("\n" + "="*60)
    print("Lemma 5.2 — Price Ladder Verification")
    print("Condition:  p*_ℓ  ≤  p_k  <  p*_j  ≤  p_i")
    print("="*60)

    # Compute margins
    df = df.copy()
    df["margin_left"]  = df["p_k*"]  - df["p_ell*"]   # p_k  − p*_ℓ  (≥ 0)
    df["margin_mid"]   = df["p_j*"]  - df["p_k*"]     # p*_j − p_k   (> 0, strict)
    df["margin_right"] = df["p_i*"]  - df["p_j*"]     # p_i  − p*_j  (≥ 0)

    # Per-tuple report
    cols = ["ell", "k", "j", "i", "p_ell*", "p_k*", "p_j*", "p_i*",
            "margin_left", "margin_mid", "margin_right", "all_ok"]
    view = df[cols].copy().round(4)
    view.columns = ["ℓ", "k", "j", "i",
                    "p*_ℓ", "p_k", "p*_j", "p_i",
                    "p_k−p*_ℓ", "p*_j−p_k", "p_i−p*_j", "pass"]
    print_df(view, header="Tuples (ℓ, k, j, i)", index=False)

    # Summary
    n_total = len(df)
    n_pass  = int(df["all_ok"].sum())
    n_fail  = n_total - n_pass
    print(f"\nResult: {n_pass}/{n_total} tuples pass")

    if n_fail == 0:
        print("✓  Lemma 5.2 HOLDS — monotone price ladder verified.")
        tightest = df[["margin_left", "margin_mid", "margin_right"]].min()
        print(f"   Tightest margins:")
        print(f"     p_k  − p*_ℓ  = {tightest['margin_left']:.4g}")
        print(f"     p*_j − p_k   = {tightest['margin_mid']:.4g}  (strict inequality)")
        print(f"     p_i  − p*_j  = {tightest['margin_right']:.4g}")
    else:
        print(f"✗  Lemma 5.2 VIOLATED on {n_fail} tuple(s).")
        bad = df[~df["all_ok"]][cols].round(4)
        print_df(bad, header="Violations", index=False)

    print("="*60 + "\n")


def main():
    conf = load_conf(f"{DATA}/run.conf")

    # --- Lemma 5.2 report ---
    ladder_path = f"{DATA}/ladder.dat"
    if not os.path.exists(ladder_path):
        print(f"No ladder.dat found at {ladder_path}. Run experiments/ladder_check.py first.")
        return

    df_ladder = pd.read_csv(ladder_path, sep="\t")
    report_lemma_52(df_ladder)

    # --- Rebuild market state for plots ---
    I, J = conf["I"], conf["J"]
    adj = np.zeros((I, J), dtype=bool)
    adj[0, 0] = True;  adj[0, 1] = True
    adj[1, 0] = True
    adj[2, 0] = True
    adj[3, 1] = True

    M = make_market_multi(I=I, J=J, Q_max=conf["Q_max"], epsilon=conf["epsilon"],
                          adj=adj, price_tol=conf["price_tol"])
    for (i, jj, q, p) in conf["bids"]:
        M["bid_q"][i, jj] = float(q)
        M["bid_p"][i, jj] = float(p)

    # --- Market snapshot ---
    metrics = compute_market_metrics(M)
    print_round_from_metrics(metrics)
    print_df(market_report_from(metrics))

    # --- Plots ---
    plot_connectivity(M, title="Market connectivity (Section 6.2)", show_labels=True)

    # One ladder-tuple plot per unique (ell, k, j, i) — the core Lemma 5.2 picture
    tuples = df_ladder[["ell", "k", "j", "i"]].drop_duplicates()
    for _, row in tuples.iterrows():
        plot_ladder_tuple(M, int(row["ell"]), int(row["k"]),
                             int(row["j"]),  int(row["i"]))

    plot_shared_buyer_surface_z0z1(M, 0, sellers=(0, 1))


if __name__ == "__main__":
    main()
