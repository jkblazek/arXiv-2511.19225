
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from market import theta_i_prime, theta_i, a_row, u_i_current, integral_P_i_j, joint_best_response_plan
from helpers import eval_u_i_two_sellers

# ------------------------------
# Comparison/ averaging plots
# ------------------------------

def plot_shared_buyer_surface_z0z1(M, i, sellers=(0,1), steps=60, 
                                   mark_current=True, with_contours=True, elev=25, azim=-120):
    """
    3D surface: u_i over (z0, z1) where z0 = alloc from seller j0, z1 = alloc from seller j1.
    Uniform price across the two sellers; price θ'(z0+z1).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    j0, j1 = sellers
    z0s = np.linspace(0.0, float(M["Q_max"][j0]), steps)
    z1s = np.linspace(0.0, float(M["Q_max"][j1]), steps)
    Z0, Z1 = np.meshgrid(z0s, z1s)
    U = np.zeros_like(Z0)

    # snapshot
    q0_old, q1_old = float(M["bid_q"][i, j0]), float(M["bid_q"][i, j1])
    p0_old, p1_old = float(M["bid_p"][i, j0]), float(M["bid_p"][i, j1])

    def w_at(z0, z1): return float(theta_i_prime(i, float(z0+z1), M))
    w_label = r"w = θ′(z₀+z₁)"

    # evaluate
    for r, z1 in enumerate(z1s):
        for c, z0 in enumerate(z0s):
            w = w_at(z0, z1)
            M["bid_q"][i, j0] = float(z0); M["bid_q"][i, j1] = float(z1)
            M["bid_p"][i, j0] = w;         M["bid_p"][i, j1] = w
            #U[r, c] = u_i_current(i, M)
            U[r, c] = eval_u_i_two_sellers(M, i, j0, j1, z0=Z0[r,c], z1=Z1[r,c])


    # restore
    M["bid_q"][i, j0] = q0_old; M["bid_q"][i, j1] = q1_old
    M["bid_p"][i, j0] = p0_old; M["bid_p"][i, j1] = p1_old

    # plot
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Z0, Z1, U, rstride=1, cstride=1, linewidth=0, alpha=0.9)
    ax.set_xlabel(f"allocation from seller {j0} (z₀)")
    ax.set_ylabel(f"allocation from seller {j1} (z₁)")
    ax.set_zlabel("uᵢ")
    ax.set_title(f"Buyer {i}: utility surface in (z₀, z₁) — {w_label}")

    if with_contours:
        # add projections to the "floor" for readability
        ax.contour(Z0, Z1, U, zdir='z', offset=U.min(), levels=12, linewidths=0.8, alpha=0.7)

    if mark_current:
        a_vec, _ = a_row(i, M)
        ax.scatter([a_vec[j0]], [a_vec[j1]], [u_i_current(i, M)],
                   s=50, c="k", marker="o", label="current")
        ax.legend(loc="best")

    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.65, aspect=12, pad=0.12, label="uᵢ")
    plt.tight_layout(); plt.show()

def plot_shared_buyer_surface_split(M, i, sellers=(0,1), q_steps=60, a_steps=60,
                                    mark_current=True):
    """
    Utility surface for a shared buyer i across two sellers (j0, j1) with *uniform price*.
    Axes: (z_total, alpha) where z0=alpha*z_total and z1=(1-alpha)*z_total.
    w := θ'_i(z_total) on each grid point
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

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

            U[r, c] = eval_u_i_two_sellers(M, i, j0, j1, z0=z0, z1=z1,
                               price_mode="theta_prime")


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

    if mark_current:
        a_vec, z_now = a_row(i, M)
        alpha_now = float(a_vec[j0] / max(z_now, 1e-12)) if z_now > 0 else 0.0
        u_now     = u_i_current(i, M)
        ax.scatter([z_now], [alpha_now], [u_now], s=50, c="k", marker="o", label="current")
        ax.legend(loc="best")

    plt.tight_layout(); plt.show()

def plot_prices_vs_percent(
    P: np.ndarray,
    E: np.ndarray,
    V: np.ndarray | None = None,
    labels: list[str] | None = None,
    title: str = "Marginal value vs % multi-auction buyers",
    config: dict | None = None,    # optional: small config box
):
    """
    Plot E (marginal value) vs percent overlap, with optional:
      - V: ±sqrt(V) shading around E
      - B: average price (e.g., avg posted bid) overlay (dashed)
      - D: average quantity on a secondary y-axis (dotted)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    L, J = E.shape
    if labels is None:
        labels = [f"Seller {j}" for j in range(J)]
    x = 100.0 * P

    # Secondary y-axis for quantities if provided
    ax_q = ax.twinx() if D is not None else None

    for j in range(J):
        color = plt.cm.tab10(j % 10)

        # E_j (marginal value)
        ax.plot(x, E[:, j], color=color, lw=2, label=f"{labels[j]} – E")

        # ±√V shading
        if V is not None:
            std = np.sqrt(np.maximum(V[:, j], 0.0))
            ax.fill_between(x, E[:, j]-std, E[:, j]+std, color=color, alpha=0.18)

    ax.set_xlabel("% of buyers bidding in multiple auctions")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Legends: combine from both axes if needed
    handles, labels_ = ax.get_legend_handles_labels()
    if ax_q is not None:
        h2, l2 = ax_q.get_legend_handles_labels()
        handles += h2; labels_ += l2
        ax_q.set_ylabel("Quantity (avg allocation)")
    ax.legend(handles, labels_, loc="best", fontsize=9)

    # Optional config box
    if config is not None:
        info_lines = []
        for key in ["I", "J", "epsilon", "base_seed", "Q_max", "reserve"]:
            if key in config:
                val = config[key]
                if isinstance(val, (list, np.ndarray)):
                    # compact summary if array-like
                    try:
                        val = np.round(np.mean(val), 3)
                    except Exception:
                        pass
                info_lines.append(f"{key}={val}")
        if info_lines:
            ax.text(1.02, 0.05, "\n".join(info_lines),
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))

    fig.tight_layout()
    plt.show()

# ------------------------------
# Valuation and utility surface plots
# ------------------------------

def plot_buyer_diagnostics(M: Dict, i: int, *, show_jbr: bool = True, num_points: int = 200):
    """
    One-stop diagnostic for buyer i:
      - Left: valuation θ_i(z) with current total allocation Z_cur marked.
      - Right: marginal θ'_i(z) with (Z_cur, θ'_i(Z_cur)), seller price lines p[i,j],
               and (optionally) the JBR target point (Z_hat, w_hat) from a dry-run.
      - Prints deviation metrics: max |p[i,j] - θ'_i(Z_cur)| on active sellers, spread of p’s, etc.
    """
    # ----- current snapshot totals -----
    a_vec, Z_cur = a_row(i, M)                              # allocations under current bids
    val_cur = theta_i(i, float(Z_cur), M)
    w_eff   = theta_i_prime(i, float(Z_cur), M)

    # Per-seller PSP cost (for info box)
    J = int(M["J"])
    cost_cur = sum(float(integral_P_i_j(i, j, float(a_vec[j]), M)) for j in range(J))
    util_cur = val_cur - cost_cur

    # Active sellers for this buyer
    active = (a_vec > 1e-12)
    p_row = M["bid_p"][i, :]
    p_active = p_row[active] if np.any(active) else np.array([], dtype=float)
    max_dev = float(np.max(np.abs(p_active - w_eff))) if p_active.size else 0.0
    spread  = float(np.ptp(p_active)) if p_active.size > 1 else 0.0

    # ----- optional JBR target (dry run; not applied) -----
    Z_hat, w_hat = None, None
    if show_jbr:
        q_hat, p_hat, feasible, u_hat = joint_best_response_plan(i, M)
        if feasible:
            Z_hat = float(np.sum(q_hat))
            w_hat = float(p_hat[0]) if len(p_hat) else 0.0

    # ----- curves -----
    zmax = float(M["qbar"][i])
    Zs = np.linspace(0.0, max(zmax, 1e-12), num_points)
    vals  = np.array([theta_i(i, float(z), M)       for z in Zs], dtype=float)
    mvals = np.array([theta_i_prime(i, float(z), M) for z in Zs], dtype=float)

    # ----- plotting -----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axV, axM = axes

    # Valuation
    axV.plot(Zs, vals, label=r"$\theta_i(z)$")
    axV.axvline(float(Z_cur), linestyle="--", linewidth=0.8)
    axV.scatter([Z_cur], [val_cur], zorder=5, label="current")
    if show_jbr and (Z_hat is not None):
        axV.axvline(Z_hat, linestyle=":", linewidth=0.8)
        axV.scatter([Z_hat], [theta_i(i, Z_hat, M)], marker="x", zorder=6, label="JBR target")

    axV.set_xlabel("Total quantity z")
    axV.set_ylabel(r"$\theta_i(z)$")
    axV.set_title(f"Buyer {i}: valuation")
    axV.legend(loc="best")
    axV.set_xlim(0.0, zmax * 1.1 if zmax > 0 else 1.0)

    # Marginal
    axM.plot(Zs, mvals, label=r"$\theta'_i(z)$")
    axM.scatter([Z_cur], [w_eff], zorder=5, label="current (z, θ')")
    if show_jbr and (w_hat is not None):
        axM.scatter([Z_hat], [w_hat], marker="x", zorder=6, label="JBR (ẑ, ŵ)")

    # Seller price lines for this buyer
    for j in range(J):
        pj = float(p_row[j])
        axM.axhline(pj, linestyle="--", linewidth=0.8, alpha=0.5)
        # annotate on the right margin
        axM.text(zmax * 1.01 if zmax > 0 else 0.01, pj, f" p[{j}]={pj:.2f}",
                 va="center", fontsize=8)

    axM.set_xlabel("Total quantity z")
    axM.set_ylabel("Marginal value / Price")
    axM.set_title(f"Buyer {i}: marginal vs. prices")
    axM.legend(loc="best")
    ylim_max = max(
        (mvals.max() if mvals.size else 1.0),
        (float(np.max(p_row)) if J > 0 else 0.0)
    ) * 1.1
    if ylim_max <= 0: ylim_max = 1.0
    axM.set_xlim(0.0, zmax * 1.1 if zmax > 0 else 1.0)
    axM.set_ylim(0.0, ylim_max)

    # Info box (right)
    txt = [
        f"Z_cur={Z_cur:.3f}",
        f"θ'(Z_cur)={w_eff:.3f}",
        f"val={val_cur:.3f}, cost={cost_cur:.3f}, util={util_cur:.3f}",
        f"max|p-θ'| (active)={max_dev:.3e}",
    ]
    if show_jbr and (Z_hat is not None):
        txt += [f"JBR: ẑ={Z_hat:.3f}, ŵ={w_hat:.3f}"]
    axM.text(0.98, 0.02, "\n".join(txt), ha="right", va="bottom",
             transform=axM.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))

    fig.suptitle(f"Buyer {i}: valuation & marginal diagnostics", y=1.02)
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

# ------------------------------
# Connectivity plots
# ------------------------------

def plot_connectivity(M: Dict, *, title: Optional[str] = None, show_labels: bool = True):
    """
    Visualize market connectivity M["adj"] (I x J boolean) with:
      - Left: matrix scatter (buyer index vs seller index)
      - Right: bipartite layout (buyers at y=0, sellers at y=1, edges as line segments)
    """
    if "adj" not in M:
        raise ValueError("M['adj'] not found. Pass an (I x J) boolean adjacency matrix in make_market_multi(...).")

    adj = np.asarray(M["adj"], dtype=bool)
    I, J = adj.shape
    ii, jj = np.nonzero(adj)  # edge endpoints (buyer i, seller j)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axL, axR = axes

    # --- Left: matrix scatter view ---
    axL.scatter(jj, ii, s=25, marker='s')
    axL.set_xlabel("seller j")
    axL.set_ylabel("buyer i")
    axL.set_xlim(-0.5, J - 0.5)
    axL.set_ylim(I - 0.5, -0.5)  # buyer 0 at top
    axL.set_xticks(range(J))
    axL.set_yticks(range(I))
    axL.set_title("Adjacency (matrix scatter)")
    axL.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)

    # --- Right: bipartite scatter view ---
    # buyers at y=0, sellers at y=1
    x_b = np.arange(I); y_b = np.zeros(I)
    x_s = np.arange(J); y_s = np.ones(J)

    axR.scatter(x_b, y_b, s=40, label="buyers")
    axR.scatter(x_s, y_s, s=60, marker="^", label="sellers")

    # Build edge segments efficiently
    if ii.size:
        segments = [((x_b[i], 0.0), (x_s[j], 1.0)) for i, j in zip(ii, jj)]
        lc = LineCollection(segments, linewidths=0.8, alpha=0.35)
        axR.add_collection(lc)

    if show_labels:
        for i in range(I):
            axR.text(x_b[i], -0.06, f"b{i}", ha="center", va="top", fontsize=8)
        for j in range(J):
            axR.text(x_s[j], 1.06, f"s{j}", ha="center", va="bottom", fontsize=8)

    axR.set_xlim(-0.5, max(I, J) - 0.5)
    axR.set_ylim(-0.4, 1.4)
    axR.set_yticks([0, 1]); axR.set_yticklabels(["buyers", "sellers"])
    axR.set_xticks([])
    axR.set_title("Adjacency (bipartite scatter)")
    axR.legend(loc="upper right", fontsize=8, frameon=False)

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()


