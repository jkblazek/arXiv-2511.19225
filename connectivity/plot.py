
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from market import theta_i_prime, theta_i, a_row, u_i_current, integral_P_i_j, joint_best_response_plan
from helpers import eval_u_i_two_sellers, pstar_j
from market import a_row, cost_row, u_i_current
from metrics import compute_market_metrics, MarketMetrics

def plot_buyer_diagnostics_broken(M: Dict,
                           i: int,
                           *,
                           num_points: int = 200):
    # --- Current state for buyer i ---
    J = M["J"]
    metrics = compute_market_metrics(M)

    p_bids = metrics.bid_p[i,:]
    p_star = metrics.p_star_j
    v_cur = metrics.buyer_value[i]
    u_cur = metrics.buyer_util[i]
    w_cur = metrics.buyer_marg[i]
    z_cur = metrics.buyer_alloc[i]
    cost_cur = metrics.buyer_cost[i]


    # --- Curves θ_i(z) and θ'_i(z) ---
    zmax = max(M["qbar"][i], M["tol"])
    Z = np.linspace(0.0, zmax, num_points)
    fig, (axV, axM) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Valuation plot ---
    Theta = np.array([theta_i(i, z, M) for z in Z])
    axV.plot(Z, Theta, label=r"$\theta_i(z)$")

    axV.axvline(z_cur, linestyle="--", linewidth=0.8)
    axV.scatter([z_cur], [v_cur], zorder=5, label="(z, θ(z))")
    
    
    axV.set_xlabel("Total quantity z")
    axV.set_ylabel(r"$\theta_i(z)$")
    axV.set_title(f"Buyer {i}: valuation")
    axV.legend(loc="best")
    axV.set_xlim(0.0, zmax * 1.1 if zmax > 0 else 1.0)

    # --- Marginal vs prices ---
    MTheta = np.array([theta_i_prime(i, z, M) for z in Z])
    axM.plot(Z, MTheta, label=r"$\theta'_i(z)$")
    axM.axvline(z_cur, linestyle="--", linewidth=0.8)
    axM.scatter([z_cur], [w_cur], zorder=5, label="(z, θ'(z))")

    axM.plot([], [], linestyle="-", linewidth=0.8, alpha=0.5, label="p*(seller)")
    axM.axhline(p_star[0], linestyle="-", linewidth=0.8, alpha=0.5)

    axM.set_xlabel("Total quantity z")
    axM.set_ylabel("Marginal Value / Price")
    axM.set_title(f"Buyer {i}: marginal vs. bids and p*")
    axM.legend(loc="best")

    y_top = max(MTheta.max() if MTheta.size else 1.0,
                np.max(p_bids) if J else 0.0,
                np.max(p_star)) * 1.1
    axM.set_xlim(0.0, zmax * 1.1 if zmax > 0 else 1.0)
    axM.set_ylim(0.0, y_top if y_top > 0 else 1.0)
    lines = [
        f"z={z_cur:.3f}\n",
        f"θ'(z)={w_cur:.3f}\n",
        f"value={v_cur:.3f}, \ncost={cost_cur:.3f}, \nutil={u_cur:.3f}\n",
    ]
    lines.append(f"best-response z={z_cur:.3f}, w={w_cur:.3f}")

    axM.text(0.98, 0.02,"".join(lines),
            ha="right",
            va="bottom",
            transform=axM.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.1),
    )

    fig.suptitle(f"Buyer {i}: Valuation and Marginal Prices", y=0.95)
    fig.tight_layout()
    plt.show()

    plot_shared_buyer_surface_z0z1(M, 0)

def plot_buyer_diagnostics(M: Dict, i: int, *, show_jbr: bool = True, num_points: int = 200):
    # snapshot
    a_vec, z_cur = a_row(i, M)
    val_cur = theta_i(i, float(z_cur), M)
    w_cur  = theta_i_prime(i, float(z_cur), M)

    J = M["J"]
    p_row = M["bid_p"][i]
    cost_cur = sum(integral_P_i_j(i, j, float(a_vec[j]), M) for j in range(J))
    util_cur = val_cur - cost_cur

    # curves
    zmax = float(M["qbar"][i])
    Zs = np.linspace(0.0, max(zmax, 1e-12), num_points)
    vals  = np.array([theta_i(i, float(z), M)       for z in Zs])
    mvals = np.array([theta_i_prime(i, float(z), M) for z in Zs])

    Z_hat, w_hat = None, None
    if show_jbr:
        q_hat, p_hat, feasible, _ = joint_best_response_plan(i, M)
        if feasible:
            Z_hat = float(np.sum(q_hat))
            w_hat = float(p_hat[0]) if len(p_hat) else None

    # plotting
    fig, (axV, axM) = plt.subplots(1, 2, figsize=(10, 4))

    # valuation
    axV.plot(Zs, vals)
    axV.axvline(z_cur, linestyle="--", linewidth=0.8)
    axV.scatter([z_cur], [val_cur], zorder=5)
    if show_jbr and (Z_hat is not None):
        axV.axvline(Z_hat, linestyle=":", linewidth=0.8)
        axV.scatter([Z_hat], [theta_i(i, Z_hat, M)], marker="x", zorder=6)
    axV.set_xlabel("Total quantity z")
    axV.set_ylabel(r"$\theta_i(z)$")
    axV.set_title(f"Buyer {i}: valuation")
    axV.set_xlim(0.0, zmax * 1.1 if zmax > 0 else 1.0)

    # marginal & prices
    axM.plot(Zs, mvals)
    axM.scatter([z_cur], [w_cur], zorder=5)
    if show_jbr and (Z_hat is not None) and (w_hat is not None):
        axM.scatter([Z_hat], [w_hat], marker="x", zorder=6)
    for j in range(J):
        pj = float(p_row[j])
        axM.axhline(pj, linestyle="--", linewidth=0.8, alpha=0.5)
        axM.text(zmax * 1.01 if zmax > 0 else 0.01, pj, f"p[{j}]={pj:.2f}", va="center", fontsize=8)
    axM.set_xlabel("Total quantity z")
    axM.set_ylabel("Marginal value / Price")
    axM.set_title(f"Buyer {i}: marginal vs. prices")
    ylim_max = max(mvals.max() if mvals.size else 1.0, float(np.max(p_row)) if J else 0.0) * 1.1
    axM.set_xlim(0.0, zmax * 1.1 if zmax > 0 else 1.0)
    axM.set_ylim(0.0, ylim_max if ylim_max > 0 else 1.0)

    # info box
    info = [
        f"z={z_cur:.3f}",
        f"θ'={w_cur:.3f}",
        f"val={val_cur:.3f}",
        f"cost={cost_cur:.3f}",
        f"util={util_cur:.3f}",
    ]
    if show_jbr and (Z_hat is not None) and (w_hat is not None):
        info.append(f"joint-best-response ẑ={Z_hat:.3f}, ŵ={w_hat:.3f}")
    axM.text(0.98, 0.02, "\n".join(info), ha="right", va="bottom",
             transform=axM.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))

    fig.suptitle(f"Buyer {i}: valuation & marginal diagnostics", y=0.95)
    fig.tight_layout()
    plt.show()

def plot_connectivity(M: Dict, *, title: Optional[str] = None, show_labels: bool = True):
    adj = M["adj"]
    I, J = adj.shape
    ii, jj = np.nonzero(adj)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axL, axR = axes
    axL.scatter(jj, ii, s=36, marker='s')
    axL.set_xlabel("seller j")
    axL.set_ylabel("buyer i")
    axL.set_xlim(-0.5, J - 0.5)
    axL.set_ylim(I - 0.5, -0.5)  # buyer 0 at top
    axL.set_xticks(range(J))
    axL.set_yticks(range(I))
    axL.set_title("Adjacency (matrix view)")
    axL.grid(True, linestyle=":", linewidth=0.6, alpha=0.4)
    x_b = np.arange(I)
    y_b = np.zeros(I)
    x_s = np.arange(J)
    y_s = np.ones(J)

    buyers_sc = axR.scatter(x_b, y_b, s=52, label="buyers")
    sellers_sc = axR.scatter(x_s, y_s, s=72, marker="^", label="sellers")

    if ii.size:
        segments = [((x_b[i], 0.0), (x_s[j], 1.0)) for i, j in zip(ii, jj)]
        lc = LineCollection(segments, linewidths=1.0, alpha=0.35)
        axR.add_collection(lc)

    if show_labels:
        for i in range(I):
            axR.text(x_b[i], -0.08, f"b{i}", ha="center", va="top", fontsize=8)
        for j in range(J):
            axR.text(x_s[j], 1.08, f"s{j}", ha="center", va="bottom", fontsize=8)
    xmax = max(I, J) - 0.5
    axR.set_xlim(-0.5, xmax)
    axR.set_ylim(-0.4, 1.4)
    axR.set_yticks([0, 1])
    axR.set_yticklabels(["buyers", "sellers"])
    axR.set_xticks([])
    axR.set_title("Adjacency (bipartite view)")
    axR.legend(loc="upper right", fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, y=.95)
    fig.tight_layout()
    plt.show()

def plot_seller_price_ladder(M, j):
    qcol = M["bid_q"][:, j]
    pcol = M["bid_p"][:, j]
    bids = [(i, float(qcol[i]), float(pcol[i])) for i in range(M["I"]) if qcol[i] > 0.0]
    if not bids:
        print(f"Seller {j}: no bids.")
        return

    bids.sort(key=lambda t: t[2], reverse=True)
    Qj = float(M["Q_max"][j])
    cum, p_star = 0.0, 0.0
    xs, heights, labels = [], [], []
    for idx, (i, q, p) in enumerate(bids):
        xs.append(idx)
        heights.append(q)
        labels.append(f"B{i}\n({q:.2g},{p:.2g})")
        if p_star == 0.0 and cum + q >= Qj:
            p_star = p
        cum += q

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(xs, heights, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("requested q_i^j")
    ax.set_xlabel("buyers\n (quantity, price)")
    ax2 = ax.twinx()
    ax2.step([x - 0.5 for x in xs] + [xs[-1] + 0.5], np.cumsum([0.0] + heights), where='post', linewidth=1.25)
    ax2.set_ylabel("cumulative quantity")

    ax.axhline(Qj, linestyle='--', linewidth=1, label=f"Q_j={Qj:g}")
    ax2.axhline(Qj, linestyle='--', linewidth=1)
    if p_star > 0.0:
        ax.text(0.02, 0.95, f"p*={p_star:.3g}", transform=ax.transAxes, ha='left', va='top', bbox=dict(boxstyle='round', alpha=0.2))
    ax.set_title(f"Seller {j}: price ladder and p*")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_threshold_cross_section(M, j):
    qcol = M["bid_q"][:, j]
    pcol = M["bid_p"][:, j]
    bids = [(float(pcol[i]), float(qcol[i])) for i in range(M["I"]) if qcol[i] > 0.0]
    if not bids:
        print(f"Seller {j}: no bids.")
        return
    bids.sort(reverse=True)
    prices = np.array([b[0] for b in bids])
    qtys = np.array([b[1] for b in bids])
    cum = np.cumsum(qtys)
    Qj = float(M["Q_max"][j])
    p_star = 0.0
    for k in range(len(cum)):
        if cum[k] >= Qj:
            p_star = prices[k]
            break
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.step(prices, cum, where="post")
    ax.axhline(Qj, linestyle="--", linewidth=1, label=f"Q_j={Qj:g}")
    if p_star > 0:
        ax.axvline(p_star, linestyle=":", linewidth=1)
        ax.text(p_star, 0, f"  p*={p_star:.3g}", va="bottom", ha="left")
    ax.set_xlabel("price")
    ax.set_ylabel("cumulative quantity ≥ price")
    ax.set_title(f"Seller {j}: threshold cross-section")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

def plot_shared_buyer_surface_z0z1(M, i, sellers=(0,1), steps=60,
                                   elev=25, azim=-120):
    j0, j1 = sellers
    z0s = np.linspace(0.0, float(M["Q_max"][j0]), steps)
    z1s = np.linspace(0.0, float(M["Q_max"][j1]), steps)
    Z0, Z1 = np.meshgrid(z0s, z1s)
    U = np.zeros_like(Z0)
    q0_old, q1_old = float(M["bid_q"][i, j0]), float(M["bid_q"][i, j1])
    p0_old, p1_old = float(M["bid_p"][i, j0]), float(M["bid_p"][i, j1])
    def w_at(z0, z1): return float(theta_i_prime(i, float(z0+z1), M))
    w_label = r"w = θ′(z₀+z₁)"
    for r, z1 in enumerate(z1s):
        for c, z0 in enumerate(z0s):
            w = w_at(z0, z1)
            M["bid_q"][i, j0] = float(z0); M["bid_q"][i, j1] = float(z1)
            M["bid_p"][i, j0] = w;         M["bid_p"][i, j1] = w
            #U[r, c] = u_i_current(i, M)
            U[r, c] = eval_u_i_two_sellers(M, i, j0, j1, z0=Z0[r,c], z1=Z1[r,c])
    M["bid_q"][i, j0] = q0_old; M["bid_q"][i, j1] = q1_old
    M["bid_p"][i, j0] = p0_old; M["bid_p"][i, j1] = p1_old
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Z0, Z1, U, rstride=1, cstride=1, linewidth=0, alpha=0.9)
    ax.set_xlabel(f"allocation from seller {j0} (z₀)")
    ax.set_ylabel(f"allocation from seller {j1} (z₁)")
    ax.set_zlabel("uᵢ")
    ax.set_title(f"Buyer {i}: utility surface in (z₀, z₁) — {w_label}")
    ax.contour(Z0, Z1, U, zdir='z', offset=U.min(), levels=12, linewidths=0.8, alpha=0.7)
    a_vec, _ = a_row(i, M)
    ax.scatter([a_vec[j0]], [a_vec[j1]], [u_i_current(i, M)],
                s=50, c="k", marker="o", label="current")
    ax.legend(loc="best")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.65, aspect=12, pad=0.12, label="uᵢ")
    plt.tight_layout(); plt.show()

def plot_shared_buyer_surface_split(M, i, sellers=(0,1), q_steps=60, a_steps=60):
    j0, j1 = sellers
    qmax = float(M["Q_max"][j0] + M["Q_max"][j1])
    zs   = np.linspace(0.0, max(qmax, 1e-12), q_steps)
    alps = np.linspace(0.0, 1.0, a_steps)
    Z, A = np.meshgrid(zs, alps)
    U    = np.zeros_like(Z)
    # snapshot
    q0_old, q1_old = float(M["bid_q"][i, j0]), float(M["bid_q"][i, j1])
    p0_old, p1_old = float(M["bid_p"][i, j0]), float(M["bid_p"][i, j1])
    def w_at(z): return float(theta_i_prime(i, float(z), M))
    # evaluate u on grid (others held fixed)
    for r in range(a_steps):
        for c in range(q_steps):
            z_tot = float(Z[r, c]); alpha = float(A[r, c])
            z0 = alpha * z_tot
            z1 = (1.0 - alpha) * z_tot
            w  = w_at(z_tot)
            M["bid_p"][i, j0] = w;  M["bid_p"][i, j1] = w
            z0 = alpha * z_tot
            z1 = (1 - alpha) * z_tot
            M["bid_q"][i, j0] = z0; M["bid_q"][i, j1] = z1
            ##U[r, c] = u_i_current(i, M)
            U[r, c] = eval_u_i_two_sellers(M, i, j0, j1, z0=z0, z1=z1)
    # restore
    M["bid_q"][i, j0] = q0_old; M["bid_q"][i, j1] = q1_old
    M["bid_p"][i, j0] = p0_old; M["bid_p"][i, j1] = p1_old
    # plot
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Z, A, U, rstride=1, cstride=1, linewidth=0, alpha=0.85)
    ax.set_xlabel("total quantity z_total")
    ax.set_ylabel(r"split $\alpha$  (z$_{j0}$ = α·z,  z$_{j1}$ = (1−α)·z)")
    ax.set_zlabel("u_i")
    title = f"Buyer {i}: utility vs total & split (sellers {j0},{j1})"
    title += " — uniform w = θ′(z_total)"
    ax.set_title(title)

    a_vec, z_now = a_row(i, M)
    alpha_now = float(a_vec[j0] / max(z_now, 1e-12)) if z_now > 0 else 0.0
    u_now     = u_i_current(i, M)
    ax.scatter([z_now], [alpha_now], [u_now], s=50, c="k", marker="o", label="current")
    ax.legend(loc="best")
    plt.tight_layout(); plt.show()


def plot_prices_vs_percent(P, E, V=None, T=None, labels=None,
                          title="Marginal Value vs % Multi-Auction Buyers",
                          config=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    L, J = E.shape
    if labels is None:
        labels = [f"Seller {j}" for j in range(J)]
    x = 100.0 * P

    for j in range(J):
        ax.plot(x, E[:, j], lw=2, label=f"{labels[j]} – E(p_i)")
        if V is not None:
            std = np.sqrt(np.maximum(V[:, j], 0.0))
            ax.fill_between(x, E[:, j]-std, E[:, j]+std, alpha=0.18)
        if T is not None:
            ax.plot(x, T[:, j], linestyle=":", lw=2.0, label=f"{labels[j]} – p*")

    ax.set_xlabel("% of buyers bidding in multiple auctions")
    ax.set_ylabel("Marginal Value/ Price")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    if config is not None:
        info = []
        for key in config:
            val = config[key]
            info.append(f"{key}={val}")
        if info:
            ax.text(1.02, 0.05, "\n".join(info), transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))

    fig.tight_layout()
    plt.show()

def plot_utility_surface(M: Dict, i: int, j: int, q_steps: int = 50, p_steps: int = 50):
    """Plot u_i under PSP cost while varying (q,p) at seller j.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # noqa: F401 (needed for 3D projection)

    zs = np.linspace(0.0, float(M["Q_max"][j]), q_steps)
    w_max = theta_i_prime(i, 0.0, M)
    ws = np.linspace(0.0, max(w_max, 1e-12), p_steps)
    Z, W = np.meshgrid(zs, ws)
    U = np.zeros_like(Z)

    # snapshot original cell and compute
    q_old = float(M["bid_q"][i, j]); p_old = float(M["bid_p"][i, j])
    for r in range(p_steps):
      for c in range(q_steps):
        M["bid_q"][i, j] = float(Z[r, c])
        M["bid_p"][i, j] = float(W[r, c])
        U[r, c] = u_i_current(i, M)
    M["bid_q"][i, j] = q_old
    M["bid_p"][i, j] = p_old

    print(f"Utility range (buyer {i}, seller {j}): {U.min():.6f} .. {U.max():.6f}")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(Z, W, U, rstride=1, cstride=1)
    ax.set_xlabel('quantity z to seller j')
    ax.set_ylabel('price w to seller j')
    ax.set_zlabel('u_i (PSP)')
    ax.set_title(f'Utility Surface (buyer {i}, seller {j})')
    plt.tight_layout(); plt.show()

def plot_equilibrium_transition(P, p_star_avg, theta_prime_avg, z_star_avg=None, title=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if z_star_avg is not None:
        ax.plot(P, z_star_avg, "o-", label="avg $z^*$ (per buyer)")
    ax.plot(P, theta_prime_avg, "s-", label=r"avg $\theta_i'(z_i^*)$")
    ax.plot(P, p_star_avg, "^-", label=r"avg $p^*$")

    ax.set_xlabel("% shared buyers")
    ax.set_ylabel("Value / Price / Quantity")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Equilibrium transition vs. market connectivity")
    ax.grid(True, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

