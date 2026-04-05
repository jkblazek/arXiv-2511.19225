
from typing import Dict, Tuple, List, Optional
from market import *
from helpers import *
from ladder import *
from network import *
from init import *
from plot import *
from metrics import *

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

def run_noise_experiment(I, J, percents, Q_max, epsilon, reserve, steps, seed, jitter):
    # Fix primitives once
    Q = np.full(J, float(Q_max)) if np.isscalar(Q_max) else np.asarray(Q_max, float)
    R = np.full(J, float(reserve)) if np.isscalar(reserve) else np.asarray(reserve, float)

    M = make_market_multi(I, J, Q_max=Q, epsilon=epsilon, reserve=R,
                          seed=seed, adj=np.ones((I, J), dtype=bool))

    P = np.asarray(percents, float)
    E = np.zeros((len(P), J)); V = np.zeros((len(P), J))
    T = np.zeros((len(P), J))

    seed_bids  = seed + 17    # same bids across levels
    seed_sched0 = seed + 313  # deterministic schedule per level

    for li, pm in enumerate(P):
        # new adjacency for this level (percent-specific but reproducible)
        rng_adj = np.random.default_rng(seed + 1009*li)
        adj = make_membership_adj(I, J, pm, rng=rng_adj)

        # reuse the same market: swap adj + reset state/bids
        reset_market_for_new_adj(M, adj, seed_bids=seed_bids,
                                 seed_sched=seed_sched0, jitter=0.0)

        #M["instant_post"] = True            # apply updates on compute, no POST_BID
        M["deterministic_sched"] = True     # schedule computes at t0+i, no jitter

        schedule_all_buyers_stable(M, t0=0.0, seed_order=42)
        #schedule_all_buyers(M, t0=0.0)

        run(M, steps=2500, verbose=False)

        df = make_ladder_report(M)

        #plot_connectivity(M, title="Market connectivity "+str(pm), show_labels=True)
        print(adj)
        metrics = compute_market_metrics(M)
        rep = market_report_from(metrics)
        rep["% shared buyers"] = pm
        print_df(rep)
        print_round_from_metrics(metrics)
        #plot_shared_buyer_surface_z0z1(M, 6, sellers=(0,1))
        #plot_buyer_diagnostics(M, 6, show_jbr=True)
        #plot_buyer_diagnostics(M, 3, show_jbr=True)
        #plot_seller_price_ladder(M, 0)
        #schedule_all_buyers(M)

        # 1) Global classification (one row per buyer)
        df_bstat = classify_buyers(M)
        n_zero_alloc_pos_bid = int(df_bstat["zero_alloc_pos_bid"].sum())
        n_zero_bid_no_gain   = int(df_bstat["zero_bid_no_gain"].sum())
        # 2) Per-seller loser counts
        losers_per_seller = per_seller_outbid_losers(M)
        rep["losers"] = losers_per_seller  # per seller
        # Add global counts (same value on each seller row for convenience)
        rep["zero_alloc_pos_bid_total"] = n_zero_alloc_pos_bid
        rep["zero_bid_no_gain_total"]   = n_zero_bid_no_gain

        # metrics & reports
        E[li] = rep["E_j"].to_numpy()
        V[li] = rep["V_j"].to_numpy()
        T[li] = rep["p_star_j"].to_numpy()


    return M, P, E, V, T


def run_connectivity_experiment(I, J, percents, Q_max, epsilon, reserve,
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
            rep["% shared buyers"] = pm
            print_df(rep)

            E[li] += rep["E_j"].to_numpy()
            V[li] += rep["V_j"].to_numpy()
            T[li] += rep["p_star_j"].to_numpy()
            #print("Percentage ", pm, " done")

    # average over trials
    E /= trials
    V /= trials
    T /= trials
    return P, E, V, T

def experiment1():
    I, J = 300, 3
    seed=20405008
    Q_max=[500,1000,2000]
    epsilon=5.0
    jitter=0

    percents = [x / 100.0 for x in range(0, 101, 10)]  # 0 → 100
    P, E, V, T = run_connectivity_experiment(
        I=I,
        J=J,
        percents=percents,
        Q_max=Q_max,
        epsilon=epsilon,
        reserve=[6,6,6],
        steps=2500,
        seed=seed,
        jitter=jitter,
        trials=3,
    )
    seeds = record_seeds(
        base_seed=seed,
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
                           config={#"epsilon": epsilon,
                                    #"seed": seed,
                                    #"Q_max": Q_max,
                                    #"jitter": jitter,
                                    #"I": I
                                   }
                           )

# -----------------
# Demo (minimal)
# -----------------
if __name__ == "__main__":
    #run_demo()
    experiment1()


