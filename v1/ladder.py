
import numpy as np
import pandas as pd

EPS_Q = 1e-12

class LadderDict:
    """
    Price-ladder verification (print-based) for a dict-style market M.

    Expected keys in M:
      - "I": int
      - "J": int
      - "Q_max": array-like of length J
      - "bid_q": (I,J)
      - "bid_p": (I,J)
      - optional: "adj" (I,J) bool mask of feasible links
      - optional: "price_tol": float
      - optional: "seller_converged": (J,) bool
    """

    def __init__(self, M):
        self.M = M
        self._ladder_impossible = set()

    # --------------------------- helpers over M -----------------------------

    def _buyers_on(self, j):
        """Buyers i with positive quantity on seller j."""
        qcol = np.asarray(self.M["bid_q"])[:, j]
        active = qcol > 0.0 + EPS_Q
        if "adj" in self.M:
            active &= np.asarray(self.M["adj"])[:, j].astype(bool)
        return np.nonzero(active)[0]

    def _active_sellers_for(self, i):
        """Sellers j with positive quantity for buyer i."""
        qrow = np.asarray(self.M["bid_q"])[i, :]
        mask = qrow > 0.0 + EPS_Q
        if "adj" in self.M:
            mask &= np.asarray(self.M["adj"])[i, :].astype(bool)
        return np.nonzero(mask)[0]

    def _seller_shell_1hop(self, j):
        """1-hop seller shell: sellers ℓ that share ≥1 active buyer with seller j."""
        shell = set()
        buyers = self._buyers_on(j)
        if buyers.size == 0:
            return shell
        bid_q = np.asarray(self.M["bid_q"])
        for i in buyers:
            sellers_i = np.nonzero(bid_q[i, :] > 0.0 + EPS_Q)[0]
            for ell in sellers_i:
                if ell == j:
                    continue
                shell.add(ell)
        return shell

    def _seller_converged(self, j):
        conv = self.M.get("seller_converged", None)
        if conv is None:
            return True
        return bool(np.asarray(conv)[j])

    def _pstar(self, j):
        """Marginal winning price on seller j (0.0 if undersubscribed)."""
        I = int(self.M["I"])
        Qj = float(np.asarray(self.M["Q_max"])[j])
        qcol = np.asarray(self.M["bid_q"])[:, j]
        pcol = np.asarray(self.M["bid_p"])[:, j]
        idx = np.nonzero(qcol > 0.0 + EPS_Q)[0]
        if idx.size == 0:
            return 0.0
        bids = [(int(i), float(qcol[i]), float(pcol[i])) for i in idx]
        bids.sort(key=lambda t: t[2], reverse=True)
        cum = 0.0
        for (i, q, p) in bids:
            if cum + q >= Qj - EPS_Q:
                return float(p)
            cum += q
        return 0.0

    def _winners_on(self, j):
        """Buyers who fill up to Q_max[j] at current bids."""
        Qj = float(np.asarray(self.M["Q_max"])[j])
        qcol = np.asarray(self.M["bid_q"])[:, j]
        pcol = np.asarray(self.M["bid_p"])[:, j]
        idx = np.nonzero(qcol > 0.0 + EPS_Q)[0]
        if idx.size == 0:
            return []
        bids = [(int(i), float(qcol[i]), float(pcol[i])) for i in idx]
        bids.sort(key=lambda t: t[2], reverse=True)
        winners = []
        cum = 0.0
        for i, q, p in bids:
            if cum + EPS_Q >= Qj:
                break
            take = min(q, max(Qj - cum, 0.0))
            if take > 0.0:
                winners.append(i)
                cum += take
        return winners

    # ---------------------- ladder precheck ------------------------------

    def _ladder_precheck(self, j):
        if not self._seller_converged(j):
            return False, f"seller {j} not converged (shell not saturated)"
        pj_star = self._pstar(j)
        if pj_star <= 0.0:
            return False, f"seller {j} undersubscribed / no marginal tier (p*_j=0)"
        shell = self._seller_shell_1hop(j)
        if not shell:
            return False, f"no 1-hop neighbors with shared active buyers for seller {j}"
        if all(not self._seller_converged(ell) for ell in shell):
            return False, f"no converged neighbors with a marginal tier around seller {j}"
        Wj = self._winners_on(j)
        if not Wj:
            return False, f"no winners on seller {j} (empty W^j)"
        has_valid_neighbor = any(self._seller_converged(ell) and self._pstar(ell) > 0.0 for ell in shell)
        if not has_valid_neighbor:
            return False, f"no converged neighbors with a marginal tier around seller {j}"
        return True, ""

    # ---------------------- price ladder core check -------------------------

    def _price_ladder_checks(self, j, hops=1):
        ok, reason = self._ladder_precheck(j)
        if not ok:
            print(f"[ladder] seller {j}: skipped — {reason}")
            return pd.DataFrame({"__no_ladder_reason__": [reason]})

        pj_star = self._pstar(j)
        shell = self._seller_shell_1hop(j)
        Wj = set(self._winners_on(j))
        tol = float(self.M.get("price_tol", 1e-9))
        rows = []
        bid_q = np.asarray(self.M["bid_q"])
        bid_p = np.asarray(self.M["bid_p"])

        for i in Wj:
            pi_star = float(bid_p[i, j])
            sellers_i = set(self._active_sellers_for(i))
            for ell in (sellers_i - {j}) & set(shell):
                if not self._seller_converged(ell):
                    continue
                p_ell_star = self._pstar(ell)
                if p_ell_star >= pj_star - tol:
                    pair = (j, ell)
                    if pair not in self._ladder_impossible:
                        print(f"[ladder] seller {j}: cannot build vs {ell} "
                              f"(p*_ell={p_ell_star:.3f} ≥ p*_j={pj_star:.3f})")
                        self._ladder_impossible.add(pair)
                    continue
                if p_ell_star <= 0.0 + tol:
                    continue

                winners_ell = self._winners_on(ell)
                for k in winners_ell:
                    if k in Wj:
                        continue
                    qk_ell = float(bid_q[k, ell])
                    if qk_ell <= 0.0 + EPS_Q:
                        continue
                    pk_ell = float(bid_p[k, ell])
                    ok_left  = (p_ell_star <= pk_ell + tol)
                    ok_mid   = (pk_ell <  pj_star - tol)
                    ok_right = (pj_star  <= pi_star + tol)
                    rows.append({
                        "ell": int(ell),
                        "k": int(k),
                        "j": int(j),
                        "i": int(i),
                        "p_ell*": float(p_ell_star),
                        "p_k*": float(pk_ell),
                        "p_j*": float(pj_star),
                        "p_i*": float(pi_star),
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

    # --------------------------- public API ---------------------------------

    def check_all_price_ladders(self, t=0.0, hops=1):
        frames = []
        reasons = []
        J = int(self.M["J"])
        for j in range(J):
            out = self._price_ladder_checks(j, hops=hops)
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

    def _ladder_reason_empty(self):
        J = int(self.M["J"])
        zeros = []
        for j in range(J):
            if self._pstar(j) == 0.0:
                zeros.append(j)
        if zeros:
            ids = ", ".join(str(x) for x in zeros)
            return f"No ladder: undersubscribed sellers with p* = 0 (sellers: {ids})."
        return ("No ladder: graph has only trivial shells "
                "(each seller shares ≤1 buyer, so (ℓ,k,j,i) cannot form).")

    def print_final_ladder_report(self, t_now=0.0, hops=1):
        out = self.check_all_price_ladders(t_now, hops=hops)
        if out.empty:
            print(f"[Final Ladder] {self._ladder_reason_empty()}")
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

