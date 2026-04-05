
from typing import Dict, Tuple, List, Optional
from market import *
from helpers import *
from ladder import *
from network import *
from init import *
from plot import *
from metrics import *

# =====================================================
# 1) Sanity (Price-Ladder)
# =====================================================

def seed_bids_multi(M, bids):
    """
    bids: iterable of (i, j, q, p)
    """
    for (i, j, q, p) in bids:
        M["bid_q"][i, j] = float(q)
        M["bid_p"][i, j] = float(p)

def run_sanity_ladder():
    """
    Construction:
      - Two sellers: j=1 (Q=8), ℓ=0 (Q=15).
      - Buyer i (idx 0): bids high on BOTH sellers (q=8 on j & ℓ at price 40).
      - Buyer k (idx 1): bids only on ℓ (q=2 at price 4).
      - Buyer z2 (idx 2): bids only on ℓ (q=6 at very low price 1).
      - Buyer z1 (idx 3): inactive (kept for indexing symmetry).

    Outcomes:
      - Seller j (1): winners {i}; p*_j ≈ 40 (filled by i alone).
      - Seller ℓ (0): winners {i, k, z2}; p*_ℓ ≈ 1 (marginal tier is low).
      - Ladder tuple (ℓ,k,j,i) should satisfy p*_ℓ < p_k < p*_j ≤ p_i.
    """
    from math import isclose
    I=4; J=2
    ell, j = 0, 1     # two sellers
    i, k   = 0, 1    # pick any two distinct buyers

    adj = np.ones((I, J), dtype=bool)
    adj[i, ell] = True
    adj[i, j]   = True
    adj[k, ell] = True
    adj[k, j]   = False

    M = make_market_multi(I=I, J=J, Q_max=[15.0, 8.0], epsilon=2.5, adj=adj, price_tol=5e-3)

    # (i=0) high price on both sellers; (k=1) only ℓ; (z2=2) only ℓ; (z1=3) inactive
    seed_bids_multi(M, [
        (0, 1, 8.0, 40.0),  # i on j
        (0, 0, 8.0, 40.0),  # i on ℓ
        (1, 0, 2.0,  4.0),  # k on ℓ
        (2, 0, 6.0,  1.0),  # z2 on ℓ
        (3, 1, 2.0, 1.2)  # z1 inactive to keep p*_j high; uncomment to test variants
    ])

    # Ladder check
    df = make_ladder_report(M)
    metrics = compute_market_metrics(M)
    rep = market_report_from(metrics)
    print_df(rep)
    plot_connectivity(M, title="Market connectivity", show_labels=True)
    plot_buyer_diagnostics(M, 0)
    plot_buyer_diagnostics(M, 1)
    #plot_buyer_diagnostics(M, 2)
    #plot_buyer_diagnostics(M, 3)

    #plot_shared_buyer_surface_z0z1(M, 0, sellers=(0,1))
    #plot_shared_buyer_surface_z0z1(M, 1, sellers=(0,1))
    #plot_shared_buyer_surface_z0z1(M, 2, sellers=(0,1))
    #plot_shared_buyer_surface_z0z1(M, 3, sellers=(0,1))
    #plot_seller_price_ladder(M, 0)
    #plot_seller_price_ladder(M, 1)


def seed_bids(M, bids):

    for (i, q, p) in bids:
        M["bid_q"][i] = float(q)
        M["bid_p"][i] = float(p)

def run_demo():

    from math import isclose
    I=4; J=1
    adj = np.ones((I,J), dtype=bool)
    M = make_market_multi(I=I, J=J, Q_max=[15.0], adj=adj, epsilon=2.5, price_tol=5e-3)

    seed_bids(M, [
        (0, 8.0, 40.0),
        (0, 8.0, 40.0),
        (1, 2.0,  4.0),
        (2, 6.0,  1.0),
    ])

    schedule_all_buyers_stable(M, t0=0.0, seed_order=42)

    it = run(M, steps=2000, verbose=True)
    metrics = compute_market_metrics(M)
    rep = market_report_from(metrics)
    print_round_from_metrics(metrics)
    print_df(rep)

    plot_buyer_diagnostics(M, 1)
    plot_buyer_diagnostics(M, 2)


def run_experiment_trials(I, J, percents, Q_max, epsilon, reserve,
                          steps, seed, jitter, trials=100):
    # Fix primitives once
    Q = (np.full(J, float(Q_max)) if np.isscalar(Q_max)
         else np.asarray(Q_max, float))
    R = (np.full(J, float(reserve)) if np.isscalar(reserve)
         else np.asarray(reserve, float))

    P = np.asarray(percents, float)
    nP = len(P)

    E = np.zeros((nP, J))
    V = np.zeros((nP, J))
    T = np.zeros((nP, J))

    for trial in range(trials):
        # --- 1) draw ONE market (valuations) for this trial
        seed_base = seed + 1000 * trial
        M = make_market_multi(
            I, J,
            Q_max=Q,
            epsilon=epsilon,
            reserve=R,
            seed=seed_base,
            adj=np.ones((I, J), dtype=bool),
        )

        # --- 2) sweep connectivity for THIS fixed population
        for li, pm in enumerate(P):
            rng_adj = np.random.default_rng(seed_base + 1009 + li)
            adj = make_membership_adj_banded(I, J, pm)
            #adj = make_membership_adj(I, J, pm, rng=rng_adj)

            reset_market_for_new_adj(
                M,
                adj,
                seed_bids=seed_base + 17 + 50 * li,
                seed_sched=seed_base + 313 + 50 * li,
                jitter=jitter,
            )

            schedule_all_buyers_stable(M, t0=0.0, seed_order=42)
            run(M, steps=steps, verbose=False)

            metrics = compute_market_metrics(M)
            rep = market_report_from(metrics)

            E[li] += rep["E_j"].to_numpy()
            V[li] += rep["V_j"].to_numpy()
            T[li] += rep["p_star_j"].to_numpy()
            print("Percentage ", pm, " done")

    # average over trials
    E /= trials
    V /= trials
    T /= trials
    seeds = record_seeds(
        base_seed=base_seed,
        I=I, J=J, percents=percents,
        verbose=True
    )

    tbl_E = pd.DataFrame({
        f"Seller {j}": [f"{E[li, j]:.3f} ± {np.sqrt(V[li, j]):.3f}" for li in range(len(P))]
        for j in range(J)
    }, index=[f"{int(100*p)}%" for p in P])

    print_df(tbl_E, header="E_j ± √V_j by percent (mean ± std)")

    plot_prices_vs_percent(P, E, V, T,
                           labels=[f"Seller {j}" for j in range(J)],
                           config={"epsilon": epsilon,
                                    "base_seed": base_seed,
                                    "Q_max": Q_max,
                                    "jitter": jitter,
                                    "I": I}
                           )

def experiment1():
    I, J = 300, 3
    seed=20405008
    Q_max=[500,1000,2000]
    epsilon=5.0
    jitter=0

    percents = [x / 100.0 for x in range(0, 101, 10)]  # 0 → 100
    P, E, V, T = run_experiment_trials(
        I=I,
        J=J,
        percents=percents,
        Q_max=Q_max,
        epsilon=epsilon,
        reserve=[6,6,6],
        steps=2500,
        seed=seed,
        jitter=jitter,
        trials=25,
    )

# -----------------
# Demo (minimal)
# -----------------
if __name__ == "__main__":
    #run_demo()
    #run_sanity_ladder()
    experiment1()


