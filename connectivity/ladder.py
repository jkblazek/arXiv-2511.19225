
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from helpers import *

@dataclass
class LadderResult:

    _ladder_impossible: set = field(default_factory=set)

def make_ladder_report_old(M: Dict) -> LadderResult:
      _ladder_impossible = set()

      df = check_all_price_ladders(M, lad=_ladder_impossible, t=M["t"])
      if df.empty:
        print("No ladder tuples found.")
      else:
        with pd.option_context("display.max_rows", None, "display.width", 100):
            print("\nLADDER TUPLES:")
            print(df.round(3).to_string(index=False))
      print_final_ladder_report(M, df)
      return df


def make_ladder_report(M: Dict):
    lad = set()
    df = check_all_price_ladders(M, lad=lad, t=M["t"])
    if df.empty:
        print("No ladder tuples found.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 100):
            print("LADDER TUPLES:")
            print(df.round(3).to_string(index=False))
    print_final_ladder_report(M, df, t_now=M["t"])
    return df

def _ladder_precheck(M: Dict, j: int):
    if not M["seller_converged"][j]:
        return False, f"seller {j} not converged (shell not saturated)"
    pj_star = pstar_j(M, j)
    if pj_star <= 0.0:
        return False, f"seller {j} undersubscribed / no marginal tier (p*_j=0)"
    shell = seller_shell_1hop(M, j)
    if not shell:
        return False, f"no 1-hop neighbors with shared active buyers for seller {j}"
    if all(not M["seller_converged"][ell] for ell in shell):
        return False, f"no converged neighbors with a marginal tier around seller {j}"
    Wj = winners_on_j(M, j)
    if not Wj:
        return False, f"no winners on seller {j} (empty W^j)"
    has_valid_neighbor = any(M["seller_converged"][ell] and pstar_j(M, ell) > 0.0 for ell in shell)
    if not has_valid_neighbor:
        return False, f"no converged neighbors with a marginal tier around seller {j}"
    return True, ""

def _price_ladder_checks(M: Dict, j: int, lad: set(), hops: int=1):
    ok, reason = _ladder_precheck(M, j)
    if not ok:
        print(f"[ladder] seller {j}: skipped — {reason}")
        return pd.DataFrame({"__no_ladder_reason__": [reason]})

    pj_star = pstar_j(M, j)
    shell = seller_shell_1hop(M, j)
    Wj = set(winners_on_j(M, j))
    tol = M["price_tol"]
    rows = []
    bid_q = M["bid_q"]
    bid_p = M["bid_p"]

    for i in Wj:
        pi_star = float(bid_p[i, j])
        sellers_i = set(active_sellers_for_i(M, i))
        for ell in (sellers_i - {j}) & set(shell):
            if not M["seller_converged"][ell]:
                continue
            p_ell_star = pstar_j(M, ell)
            if p_ell_star >= pj_star - tol:
                pair = (j, ell)
                if pair not in lad:
                    print(f"[ladder] seller {j}: cannot build vs {ell} "
                          f"(p*_ell={p_ell_star:.3f} ≥ p*_j={pj_star:.3f})")
                    lad.add(pair)
                continue
            if p_ell_star <= 0.0 + tol:
                continue

            winners_ell = winners_on_j(M, ell)
            for k in winners_ell:
                if k in Wj:
                    continue
                qk_ell = bid_q[k, ell]
                if qk_ell <= 0.0 + M["tol"]:
                    continue
                pk_ell = bid_p[k, ell]
                ok_left  = (p_ell_star <= pk_ell + tol)
                ok_mid   = (pk_ell <  pj_star - tol)
                ok_right = (pj_star  <= pi_star + tol)
                rows.append({
                    "ell": ell,
                    "k": k,
                    "j": j,
                    "i": i,
                    "p_ell*": p_ell_star,
                    "p_k*": pk_ell,
                    "p_j*": pj_star,
                    "p_i*": pi_star,
                    "ok_left": ok_left,
                    "ok_mid": ok_mid,
                    "ok_right": ok_right,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[ladder] seller {j}: no valid (ell,k,i) tuples")
        return pd.DataFrame({"__no_ladder_reason__": [f"no valid (ell,k,i) tuples found around seller {j}"]})
    df["all_ok"] = df["ok_left"] & df["ok_mid"] & df["ok_right"]
    return df

def check_all_price_ladders(M: Dict, lad: set(), t: int=0.0, hops: int=1):
    frames = []
    reasons = []
    J = M["J"]
    for j in range(J):
        out = _price_ladder_checks(M, j, lad, hops=hops)
        if "__no_ladder_reason__" in out.columns:
            reasons.append((j, out["__no_ladder_reason__"].iloc[0]))
        else:
            out = out.copy()
            out["dt"] = t
            frames.append(out)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for sid, msg in reasons:
        print(f"[ladder] seller {sid}: {msg}")
    if df.empty:
        print("[ladder] No ladder tuples found.")
        return df
    bad = df[~df["all_ok"]]
    good = df[df["all_ok"]]

    if not good.empty:
        margin_left  = (good["p_k*"] - good["p_ell*"]).min()
        margin_mid   = (good["p_j*"] - good["p_k*"]).min()
        margin_right = (good["p_i*"] - good["p_j*"]).min()
        n_pairs = good[["ell", "j"]].drop_duplicates().shape[0]
        print(f"\n[ladder] t={t:.2f} — PRICE LADDER HOLDS on {len(good)} tuples "
              f"across {n_pairs} (j,ℓ) pairs\n"
              f"tightest margins:\n"
              f" (pk−pℓ*): {margin_left:.3g}, (p*j−pk): {margin_mid:.3g}, (pi−p*j): {margin_right:.3g}")
    if not bad.empty:
        print("\n[ladder] WARNING: violations detected (pℓ* ≤ pk < p*j ≤ pi):")
        show = bad.copy()
        for c in ["p_ell*", "p_k*", "p_j*", "p_i*"]:
            show[c] = show[c].round(3)
        print(show.to_string(index=False))
    return df

def _ladder_reason_empty(M: Dict):
    J = M["J"]
    zeros = [j for j in range(J) if pstar_j(M, j) == 0.0]
    if zeros:
        ids = ", ".join(str(x) for x in zeros)
        return f"No ladder: undersubscribed sellers with p* = 0 (sellers: {ids})."

    # Check for equal-clearing-price neighbors (common cause of empty ladders)
    equal_pairs = []
    for j in range(J):
        pj = pstar_j(M, j)
        shell = seller_shell_1hop(M, j)
        for ell in shell:
            if abs(pj - pstar_j(M, ell)) <= M["price_tol"]:
                equal_pairs.append((ell, j))

    if equal_pairs:
        pairs_str = ", ".join(f"({ell},{j})" for ell, j in sorted(set(equal_pairs)))
        return (
            "No ladder: neighboring sellers have equal clearing prices "
            f"(pairs: {pairs_str}); the ladder requires strict p*_ell < p*_j."
        )

    return (
        "No ladder: graph has only trivial shells "
        "(each seller shares ≤1 buyer, so (ℓ,k,j,i) cannot form)."
    )

def _ladder_reason_empty_old(M: Dict):
    J = M["J"]
    zeros = []
    for j in range(J):
        if pstar_j(M, j) == 0.0:
            zeros.append(j)
    if zeros:
        ids = ", ".join(str(x) for x in zeros)
        return f"No ladder: undersubscribed sellers with p* = 0 (sellers: {ids})."
    return ("No ladder: graph has only trivial shells "
            "(each seller shares ≤1 buyer, so (ℓ,k,j,i) cannot form).")

def print_final_ladder_report(M: Dict, out: pd.DataFrame, t_now=0.0, hops=1):
    if out.empty:
        print(f"[Final Ladder] {_ladder_reason_empty(M)}")
        return
    if out["all_ok"].all():
        print(f"\n[Final Ladder] Monotone price ladder verified at t={t_now:.2f} "
              "(pℓ ≤ pk < pj ≤ pi).")
        with pd.option_context("display.max_rows", None, "display.width", 100):
            view = out[["ell","k","j","i","p_ell*","p_k*","p_j*","p_i*"]].drop_duplicates().round(3)
            print(view.to_string(index=False))
    else:
        bad = out[out["all_ok"] == False]
        print("\n[Final Ladder] WARNING: violations remain.")
        with pd.option_context("display.max_rows", None, "display.width", 100):
            print(bad.to_string(index=False))
 
