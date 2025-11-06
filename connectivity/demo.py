
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


def run_experiment_trials2(I, J, percents, Q_max, epsilon, reserve, steps, base_seed, jitter, trials=100):
    # Fix primitives once
    Q = np.full(J, float(Q_max)) if np.isscalar(Q_max) else np.asarray(Q_max, float)
    R = np.full(J, float(reserve)) if np.isscalar(reserve) else np.asarray(reserve, float)

    P = np.asarray(percents, float)
    E = np.zeros((len(P), J))
    V = np.zeros((len(P), J))
    T = np.zeros((len(P), J))
    metrics_list = []

    for li, pm in enumerate(P):
        trial_E = np.zeros((trials, J))
        trial_V = np.zeros((trials, J))
        trial_T = np.zeros((trials, J))

        for trial in range(trials):
            # new market per trial for reproducibility
            seed_base = base_seed + 10000 * li + 1000 * trial
            M = make_market_multi(I, J, Q_max=Q, epsilon=epsilon, reserve=R,
                                  seed=seed_base, adj=np.ones((I, J), dtype=bool))

            rng_adj = np.random.default_rng(seed_base + 1009)
            adj = make_membership_adj(I, J, pm, rng=rng_adj)

            reset_market_for_new_adj(M, adj, seed_bids=seed_base + 17,
                                     seed_sched=seed_base + 313, jitter=jitter)

            schedule_all_buyers_stable(M, t0=0.0, seed_order=42)
            run(M, steps=steps, verbose=False)

            metrics = compute_market_metrics(M)
            rep = market_report_from(metrics)
            metrics_list.append(metrics)

            trial_E[trial] = rep["E_j"].to_numpy()
            trial_V[trial] = rep["V_j"].to_numpy()
            trial_T[trial] = rep["p_star_j"].to_numpy()

        # Mean across trials
        E[li] = trial_E.mean(axis=0)
        V[li] = trial_V.mean(axis=0)
        T[li] = trial_T.mean(axis=0)

    return P, E, V, T

def run_experiment_trials(I, J, percents, Q_max, epsilon, reserve, steps, base_seed, jitter, trials=100):
    # Fix primitives once
    Q = np.full(J, float(Q_max)) if np.isscalar(Q_max) else np.asarray(Q_max, float)
    R = np.full(J, float(reserve)) if np.isscalar(reserve) else np.asarray(reserve, float)

    P = np.asarray(percents, float)
    E = np.zeros((len(P), J))
    V = np.zeros((len(P), J))
    T = np.zeros((len(P), J))
    metrics_list = []
    M = make_market_multi(I, J, Q_max=Q, epsilon=epsilon, reserve=R,
                              seed=base_seed, adj=np.ones((I, J), dtype=bool))
  
    for li, pm in enumerate(P):
        trial_E = np.zeros((trials, J))
        trial_V = np.zeros((trials, J))
        trial_T = np.zeros((trials, J))
        seed_base = base_seed + 100*li
        rng_adj = np.random.default_rng(seed_base + 1009)
        adj = make_membership_adj(I, J, pm, rng=rng_adj)

        for trial in range(trials):
            reset_market_for_new_adj(M, adj, seed_bids=seed_base + 17,
                                 seed_sched=seed_base + 313, jitter=jitter)
        
            schedule_all_buyers_stable(M, t0=0.0, seed_order=42)
            run(M, steps=steps, verbose=False)

            metrics = compute_market_metrics(M)
            rep = market_report_from(metrics)

            trial_E[trial] = rep["E_j"].to_numpy()
            trial_V[trial] = rep["V_j"].to_numpy()
            trial_T[trial] = rep["p_star_j"].to_numpy()

        metrics_list.append(metrics)

        # Mean across trials
        E[li] = trial_E.mean(axis=0)
        V[li] = trial_V.mean(axis=0)
        T[li] = trial_T.mean(axis=0)

    return P, E, V, T, metrics_list

def run_experiment(I, J, percents, Q_max, epsilon, reserve, steps, base_seed, jitter):
    # Fix primitives once
    Q = np.full(J, float(Q_max)) if np.isscalar(Q_max) else np.asarray(Q_max, float)
    R = np.full(J, float(reserve)) if np.isscalar(reserve) else np.asarray(reserve, float)

    M = make_market_multi(I, J, Q_max=Q, epsilon=epsilon, reserve=R,
                          seed=base_seed, adj=np.ones((I, J), dtype=bool))

    P = np.asarray(percents, float)
    E = np.zeros((len(P), J)); V = np.zeros((len(P), J))
    T = np.zeros((len(P), J))

    seed_bids  = base_seed + 17    # same bids across levels
    seed_sched0 = base_seed + 313  # deterministic schedule per level

    for li, pm in enumerate(P):
        # new adjacency for this level (percent-specific but reproducible)
        rng_adj = np.random.default_rng(base_seed + 1009*li)
        adj = make_membership_adj(I, J, pm, rng=rng_adj)

        # reuse the same market: swap adj + reset state/bids
        reset_market_for_new_adj(M, adj, seed_bids=seed_bids,
                                 seed_sched=seed_sched0 + li, jitter=0.0)

        #M["instant_post"] = True            # apply updates on compute, no POST_BID
        #M["deterministic_sched"] = True     # schedule computes at t0+i, no jitter

        schedule_all_buyers_stable(M, t0=0.0, seed_order=42)
        #schedule_all_buyers(M, t0=0.0)

        run(M, steps=2000, verbose=False)

        df = make_ladder_report(M)

        plot_connectivity(M, title="Market connectivity "+str(pm), show_labels=True)
        print(adj)
        metrics = compute_market_metrics(M)
        rep = market_report_from(metrics)
        rep["% shared buyers"] = pm
        print_df(rep)
        print_round_from_metrics(metrics)
        #plot_shared_buyer_surface_z0z1(M, 6, sellers=(0,1))
        plot_buyer_diagnostics(M, 6, show_jbr=True)
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

def experiment1(stats: bool = False):
    I, J = 8, 2
    base_seed=20405008
    Q_max=[60.0, 40.0]
    epsilon=2.5

    if stats:
        percents = [x / 100.0 for x in range(0, 101, 5)]  # 0 → 100
        P, E, V, T, metrics_list = run_experiment_trials2(
            I=I,
            J=J,
            percents=percents,
            Q_max=Q_max,
            epsilon=epsilon,
            reserve=[0.0, 0.0],
            steps=1500,
            base_seed=base_seed,
            jitter=0.0
        )
        # metrics_list must be aligned with P (same order, one per level)
        p_star_avg, theta_prime_avg, z_star_avg = summarize_transition_from_arrays(P, T, metrics_list)
        plot_equilibrium_transition(P, p_star_avg, theta_prime_avg, z_star_avg,
                            title="With MarketMetrics (true θ′ and z*)")
        plot_prices_vs_percent(P, E, V, T,
                           labels=[f"Seller {j}" for j in range(J)])
    else:
        percents = [x / 100.0 for x in range(0, 101, 10)]  # 0 → 100
        M, P, E, V, T = run_experiment(
            I=I,
            J=J,
            percents=percents,
            Q_max=Q_max,
            epsilon=epsilon,
            reserve=[0.0, 0.0],
            steps=1500,
            base_seed=base_seed,
            jitter=0.0
        )
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
                                    "I": I}
                           )


# -----------------
# Demo (minimal)
# -----------------
if __name__ == "__main__":
    #run_demo()
    experiment1(stats=True)


