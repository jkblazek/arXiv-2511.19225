"""
Analysis — Aggregate Availability Staircase and P_i(z; s_{-i})

Produces three figures:

  Figure 1 — Single auction (two panels)
    Left:  H(y) — availability staircase for one seller
    Right: P_i(z) with θ'_i(z) overlaid; intersection = optimal bid w_i

  Figure 2 — Two auctions (two panels)
    Left:  H(y) — merged staircase, per-seller contributions colored separately
    Right: P_i(z) with θ'_i(z) overlaid; same structure, more steps

  Figure 3 — Utility surface z0 × z1 for the bridging buyer (two auctions)
    Shows the optimization landscape without a misleading "current" point.

Run:
    python -m analysis.staircase
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from psp.io import load_conf

DATA = "data/staircase"


def _load(fname: str) -> tuple[np.ndarray, list[str]]:
    path = f"{DATA}/{fname}"
    with open(path) as f:
        header = f.readline().lstrip("#").split()
    data = np.loadtxt(path, comments="#")
    return data, header


def plot_single(data_H, data_P):
    y  = data_H[:, 0];  H  = data_H[:, 1]
    z  = data_P[:, 0];  Pi = data_P[:, 1];  mv = data_P[:, 2]

    # intersection: where θ'_i(z) crosses P_i(z)
    diff = mv - Pi
    cross = np.where(np.diff(np.sign(diff)))[0]
    z_star = float(z[cross[0]]) if len(cross) else None
    w_star = float(Pi[cross[0]]) if len(cross) else None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: H(y)
    axL.step(y, H, where="post", lw=1.8, color="C0")
    axL.set_xlabel("Price y")
    axL.set_ylabel("Aggregate available quantity H(y)")
    axL.set_title("Single auction — H(y; s₋ᵢ)")
    axL.grid(True, linestyle=":", alpha=0.4)
    axL.set_xlim(left=0)
    axL.set_ylim(bottom=0)

    # Right: P_i(z) and θ'_i(z)
    axR.step(z, Pi, where="post", lw=1.8, color="C0", label="$P_i(z, s_{-i})$")
    axR.plot(z, mv, lw=1.5, color="C1", linestyle="--", label="$\\theta'_i(z)$")
    if z_star is not None:
        axR.axvline(z_star, color="C2", linestyle=":", lw=1.2)
        axR.axhline(w_star, color="C2", linestyle=":", lw=1.2)
        axR.scatter([z_star], [w_star], zorder=5, color="C2", s=60,
                    label=f"$w_i={w_star:.3g}$,  $z^*={z_star:.3g}$")
    axR.set_xlabel("Total quantity z")
    axR.set_ylabel("Price")
    axR.set_title("Single auction — $P_i(z)$ and $\\theta'_i(z)$")
    axR.legend(fontsize=9)
    axR.grid(True, linestyle=":", alpha=0.4)
    axR.set_xlim(left=0)
    axR.set_ylim(bottom=0)

    fig.suptitle("Single-auction staircase (Semret baseline)", y=1.01)
    fig.tight_layout()
    plt.show()


def plot_multi(data_H, data_P):
    y   = data_H[:, 0];  H   = data_H[:, 1]
    H0  = data_H[:, 2];  H1  = data_H[:, 3]
    z   = data_P[:, 0];  Pi  = data_P[:, 1];  mv = data_P[:, 2]

    diff  = mv - Pi
    cross = np.where(np.diff(np.sign(diff)))[0]
    z_star = float(z[cross[0]]) if len(cross) else None
    w_star = float(Pi[cross[0]]) if len(cross) else None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: H(y) with per-seller contributions stacked
    axL.step(y, H,  where="post", lw=2.0, color="C0", label="H(y) total")
    axL.step(y, H0, where="post", lw=1.2, color="C1", linestyle="--",
             label="seller 0 contribution")
    axL.step(y, H1, where="post", lw=1.2, color="C2", linestyle="-.",
             label="seller 1 contribution")
    axL.set_xlabel("Price y")
    axL.set_ylabel("Aggregate available quantity H(y)")
    axL.set_title("Two auctions — H(y; s₋ᵢ) merged")
    axL.legend(fontsize=9)
    axL.grid(True, linestyle=":", alpha=0.4)
    axL.set_xlim(left=0)
    axL.set_ylim(bottom=0)

    # Right: P_i(z) and θ'_i(z)
    axR.step(z, Pi, where="post", lw=1.8, color="C0", label="$P_i(z, s_{-i})$")
    axR.plot(z, mv, lw=1.5, color="C1", linestyle="--", label="$\\theta'_i(z)$")
    if z_star is not None:
        axR.axvline(z_star, color="C2", linestyle=":", lw=1.2)
        axR.axhline(w_star, color="C2", linestyle=":", lw=1.2)
        axR.scatter([z_star], [w_star], zorder=5, color="C2", s=60,
                    label=f"$w_i={w_star:.3g}$,  $z^*={z_star:.3g}$")
    axR.set_xlabel("Total quantity z")
    axR.set_ylabel("Price")
    axR.set_title("Two auctions — $P_i(z)$ and $\\theta'_i(z)$")
    axR.legend(fontsize=9)
    axR.grid(True, linestyle=":", alpha=0.4)
    axR.set_xlim(left=0)
    axR.set_ylim(bottom=0)

    fig.suptitle("Two-auction staircase — local ladders merge into one global H(y)", y=1.01)
    fig.tight_layout()
    plt.show()


def plot_surface(data_S, conf):
    z0 = data_S[:, 0];  z1 = data_S[:, 1];  u = data_S[:, 2]
    Q0, Q1 = conf["multi_Q"]
    steps = int(round(len(z0) ** 0.5))
    Z0 = z0.reshape(steps, steps)
    Z1 = z1.reshape(steps, steps)
    U  = u.reshape(steps, steps)

    fig = plt.figure(figsize=(7, 5.5))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Z0, Z1, U, rstride=1, cstride=1,
                           linewidth=0, alpha=0.88)
    ax.contour(Z0, Z1, U, zdir="z", offset=float(U.min()),
               levels=12, linewidths=0.8, alpha=0.6)
    ax.set_xlabel("z₀  (allocation from seller 0)")
    ax.set_ylabel("z₁  (allocation from seller 1)")
    ax.set_zlabel("uᵢ")
    ax.set_title("Two-auction utility surface  uᵢ(z₀, z₁)\n"
                 "w = θ′ᵢ(z₀+z₁)")
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.12, label="uᵢ")
    fig.tight_layout()
    plt.show()


def main():
    for fname in ["single_H.dat", "single_P.dat", "multi_H.dat",
                  "multi_P.dat", "multi_surface.dat"]:
        if not os.path.exists(f"{DATA}/{fname}"):
            print(f"Missing {fname} — run experiments/staircase.py first.")
            return

    conf    = load_conf(f"{DATA}/run.conf")
    data_sH, _ = _load("single_H.dat")
    data_sP, _ = _load("single_P.dat")
    data_mH, _ = _load("multi_H.dat")
    data_mP, _ = _load("multi_P.dat")
    data_S,  _ = _load("multi_surface.dat")

    plot_single(data_sH, data_sP)
    plot_multi(data_mH, data_mP)
    plot_surface(data_S, conf)


if __name__ == "__main__":
    main()
