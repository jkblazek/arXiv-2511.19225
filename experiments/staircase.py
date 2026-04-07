"""
Aggregate Availability Staircase — H(y; s_{-i}) and P_i(z; s_{-i})

Constructs two small fixed markets (single-auction and two-auction) and
saves the staircase data for analysis/staircase.py to plot.

Writes:
  data/staircase/run.conf
  data/staircase/single_H.dat     H(y) for the single-auction case
  data/staircase/single_P.dat     P_i(z) and theta'_i(z) for single auction
  data/staircase/multi_H.dat      H(y) merged across two sellers
  data/staircase/multi_P.dat      P_i(z) and theta'_i(z) for two auctions
  data/staircase/multi_surface.dat utility surface z0 x z1 for buyer i

Run:
    python -m experiments.staircase
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from psp.market import (make_market_multi, build_ladders, theta_i_prime, theta_i,
                         _avail_lt_price, avail_at_bprice)
from psp.io import write_conf

OUT = "data/staircase"

# ---------------------------------------------------------------------------
# Market: buyer 0 is the focal buyer, opponents set by hand
# Single auction: seller 0 only, 3 opponents
# Two auctions:   sellers 0 and 1, same buyer 0 bridges both
# ---------------------------------------------------------------------------
CONF = {
    "experiment": "staircase",
    "focal_buyer": 0,
    # Single-auction opponents at seller 0: (quantity, price)
    "single_opponents": [[10.0, 8.0], [15.0, 5.0], [8.0, 3.0]],
    "single_Q": 40.0,
    # Two-auction opponents:
    #   seller 0: same as single
    #   seller 1: two opponents at different prices
    "multi_opponents_0": [[10.0, 8.0], [15.0, 5.0], [8.0, 3.0]],
    "multi_opponents_1": [[12.0, 6.0], [10.0, 2.0]],
    "multi_Q": [40.0, 30.0],
    # Focal buyer valuation
    "qbar_i": 60.0,
    "kappa_i": 0.5,
    "epsilon": 2.5,
}


def _build_H(ladders: dict, J: int, y_grid: np.ndarray) -> tuple:
    """Compute H(y) on y_grid and per-seller contributions."""
    per_seller = ladders["per_seller"]
    H = np.zeros_like(y_grid)
    H_per = np.zeros((J, len(y_grid)))
    for j in range(J):
        L = per_seller[j]
        p = L["p"]; suf = L["suf"]
        for yi, y in enumerate(y_grid):
            idx = np.searchsorted(p, y, side="right")
            contrib = float(suf[idx]) if p.size else 0.0
            H_per[j, yi] = contrib
            H[yi] += contrib
    return H, H_per


def _build_Pi(ladders: dict, M: dict, i: int, z_grid: np.ndarray) -> np.ndarray:
    """Compute P_i(z) = inf{y >= 0 : H(y) >= z} on z_grid."""
    steps = ladders["steps"]
    Pi = np.zeros_like(z_grid)
    for zi, z in enumerate(z_grid):
        if z <= 0.0:
            Pi[zi] = 0.0
            continue
        # find smallest y such that H(y) >= z
        found = False
        for y in reversed(steps):   # steps ascending, scan from high to low
            caps = avail_at_bprice(i, y, M, ladders)
            if float(caps.sum()) >= z - M["tol"]:
                Pi[zi] = float(y)
                found = True
                break
        if not found:
            Pi[zi] = float(steps[-1])
    return Pi


def _write_dat(path: str, header: str, data: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in data:
            f.write("  ".join(f"{v:.6g}" for v in row) + "\n")


def run():
    os.makedirs(OUT, exist_ok=True)
    write_conf(f"{OUT}/run.conf", CONF)

    i       = CONF["focal_buyer"]
    qbar_i  = CONF["qbar_i"]
    kappa_i = CONF["kappa_i"]
    eps     = CONF["epsilon"]

    # ------------------------------------------------------------------
    # Single auction
    # ------------------------------------------------------------------
    I_s = 1 + len(CONF["single_opponents"])   # focal + opponents
    J_s = 1
    adj_s = np.ones((I_s, J_s), dtype=bool)

    M_s = make_market_multi(I_s, J_s, Q_max=[CONF["single_Q"]],
                             epsilon=eps, adj=adj_s, seed=1)
    # Override focal buyer valuation
    M_s["qbar"][i]  = qbar_i
    M_s["kappa"][i] = kappa_i
    # Place opponent bids
    for oi, (q, p) in enumerate(CONF["single_opponents"]):
        M_s["bid_q"][oi + 1, 0] = float(q)
        M_s["bid_p"][oi + 1, 0] = float(p)

    ladders_s = build_ladders(i, M_s)
    y_max_s   = float(ladders_s["steps"][-1]) * 1.1
    y_grid_s  = np.linspace(0.0, y_max_s, 500)
    H_s, H_per_s = _build_H(ladders_s, J_s, y_grid_s)

    z_max_s  = float(H_s.max())
    z_grid_s = np.linspace(0.0, z_max_s, 500)
    Pi_s     = _build_Pi(ladders_s, M_s, i, z_grid_s)
    mv_s     = np.array([theta_i_prime(i, float(z), M_s) for z in z_grid_s])

    _write_dat(f"{OUT}/single_H.dat",
               "#y  H  H_0",
               np.column_stack([y_grid_s, H_s, H_per_s[0]]))

    _write_dat(f"{OUT}/single_P.dat",
               "#z  Pi  mv",
               np.column_stack([z_grid_s, Pi_s, mv_s]))

    # ------------------------------------------------------------------
    # Two auctions
    # ------------------------------------------------------------------
    n_opp_0 = len(CONF["multi_opponents_0"])
    n_opp_1 = len(CONF["multi_opponents_1"])
    I_m = 1 + max(n_opp_0, n_opp_1)
    J_m = 2
    adj_m = np.zeros((I_m, J_m), dtype=bool)
    adj_m[i, 0] = True;  adj_m[i, 1] = True   # focal bridges both
    for oi in range(n_opp_0):
        adj_m[oi + 1, 0] = True
    for oi in range(n_opp_1):
        adj_m[oi + 1, 1] = True

    M_m = make_market_multi(I_m, J_m, Q_max=CONF["multi_Q"],
                             epsilon=eps, adj=adj_m, seed=2)
    M_m["qbar"][i]  = qbar_i
    M_m["kappa"][i] = kappa_i
    for oi, (q, p) in enumerate(CONF["multi_opponents_0"]):
        M_m["bid_q"][oi + 1, 0] = float(q)
        M_m["bid_p"][oi + 1, 0] = float(p)
    for oi, (q, p) in enumerate(CONF["multi_opponents_1"]):
        M_m["bid_q"][oi + 1, 1] = float(q)
        M_m["bid_p"][oi + 1, 1] = float(p)

    ladders_m = build_ladders(i, M_m)
    y_max_m   = float(ladders_m["steps"][-1]) * 1.1
    y_grid_m  = np.linspace(0.0, y_max_m, 500)
    H_m, H_per_m = _build_H(ladders_m, J_m, y_grid_m)

    z_max_m  = float(H_m.max())
    z_grid_m = np.linspace(0.0, z_max_m, 500)
    Pi_m     = _build_Pi(ladders_m, M_m, i, z_grid_m)
    mv_m     = np.array([theta_i_prime(i, float(z), M_m) for z in z_grid_m])

    _write_dat(f"{OUT}/multi_H.dat",
               "#y  H  H_0  H_1",
               np.column_stack([y_grid_m, H_m, H_per_m[0], H_per_m[1]]))

    _write_dat(f"{OUT}/multi_P.dat",
               "#z  Pi  mv",
               np.column_stack([z_grid_m, Pi_m, mv_m]))

    # ------------------------------------------------------------------
    # Utility surface z0 x z1 (two-auction buyer)
    # ------------------------------------------------------------------
    steps_z = 60
    Q0, Q1 = CONF["multi_Q"]
    z0s = np.linspace(0.0, float(Q0), steps_z)
    z1s = np.linspace(0.0, float(Q1), steps_z)

    rows = []
    q0_old = float(M_m["bid_q"][i, 0]); q1_old = float(M_m["bid_q"][i, 1])
    p0_old = float(M_m["bid_p"][i, 0]); p1_old = float(M_m["bid_p"][i, 1])

    for z0 in z0s:
        for z1 in z1s:
            z_tot = z0 + z1
            w = theta_i_prime(i, float(z_tot), M_m)
            M_m["bid_q"][i, 0] = float(z0); M_m["bid_q"][i, 1] = float(z1)
            M_m["bid_p"][i, 0] = w;         M_m["bid_p"][i, 1] = w
            # PSP cost approximation: pay p* per seller on allocated quantity
            from psp.helpers import pstar_j
            from psp.market import Qbar_i_j
            a0 = min(z0, Qbar_i_j(i, 0, w, M_m))
            a1 = min(z1, Qbar_i_j(i, 1, w, M_m))
            p0 = pstar_j(M_m, 0); p1 = pstar_j(M_m, 1)
            u = theta_i(i, a0 + a1, M_m) - a0 * p0 - a1 * p1
            rows.append((z0, z1, u))

    M_m["bid_q"][i, 0] = q0_old; M_m["bid_q"][i, 1] = q1_old
    M_m["bid_p"][i, 0] = p0_old; M_m["bid_p"][i, 1] = p1_old

    _write_dat(f"{OUT}/multi_surface.dat",
               "#z0  z1  u",
               np.array(rows))

    print(f"Output written to {OUT}/")


if __name__ == "__main__":
    run()
