
from typing import Dict, Tuple, List, Optional
from market import *
from helpers import *
from ladder import *
from network import *
from init import *
from plot import *
from metrics import *

# ======================================================
# 2) Sanity (Price-Ladder)
# ======================================================
def seed_bids(M, bids):
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
    i, k, z2, z1   = 0, 1, 2, 3     # two distinct buyers

    #adj = np.ones((I, J), dtype=bool)
    adj = np.zeros((I, J), dtype=bool)
    adj[i, ell] = True
    adj[i, j]   = True
    adj[k, ell] = True
    adj[k, j]   = False
    M = make_market_multi(I=I, J=J, Q_max=[15.0, 8.0], epsilon=2.5, adj=adj, price_tol=5e-3)

    # (i=0) high price on both sellers; (k=1) only ℓ; (z2=2) only ℓ; (z1=3) inactive
    seed_bids(M, [
        (i, j, 8.0, 40.0),  # i on j
        (i, ell, 8.0, 40.0),  # i on ℓ
        (k, ell, 2.0,  4.0),  # k on ℓ
        (z2, ell, 6.0,  1.0),  # z2 on ℓ
        (z1, j, 2.0, 1.2)  # z1 inactive to keep p*_j high; uncomment to test variants
    ])

    #M["instant_post"] = True            # apply updates on compute, no POST_BID
    #M["deterministic_sched"] = True     # schedule computes at t0+i, no jitter

    set_stable_order(M, seed_order=42)
    schedule_all_buyers_stable(M, t0=0.0)
    #schedule_all_buyers(M, t0=0.0)

    #t = run(M, steps=2000, verbose=True)
    metrics = compute_market_metrics(M)
    rep = market_report_from(metrics)
    print_df(rep)

    p_l = rep["p_star_j"][0]
    p_j = rep["p_star_j"][1]
    #assert isclose(p_j, 40, rel_tol=0.0, abs_tol=0.5), f"p*_j should be 40, got {p_j}"
    #assert isclose(p_l,  1.0, rel_tol=0.0, abs_tol=0.5), f"p*_ℓ should be 1, got {p_l}"

    # Ladder check
    lad = LadderDict(M)
    df = lad.check_all_price_ladders(t=M["t"])

    if df.empty:
        print("No ladder tuples found.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 100):
            print("\nLADDER TUPLES:")
            print(df.round(3).to_string(index=False))
            plot_connectivity(M, title="Market connectivity", show_labels=True)
            plot_buyer_diagnostics(M, i=2, show_jbr=True)
            plot_shared_buyer_surface_split(M, 2, sellers=(0,1))
            plot_shared_buyer_surface_z0z1(M, 2, sellers=(0,1))

    lad.print_final_ladder_report(M["t"])


def run_experiment(I, J, percents, Q_max, epsilon, reserve, steps, base_seed, jitter):
    # Fix primitives once
    Q = np.full(J, float(Q_max)) if np.isscalar(Q_max) else np.asarray(Q_max, float)
    R = np.full(J, float(reserve)) if np.isscalar(reserve) else np.asarray(reserve, float)

    M = make_market_multi(I, J, Q_max=Q, epsilon=epsilon, reserve=R,
                          seed=base_seed, adj=np.ones((I, J), dtype=bool))

    P = np.asarray(percents, float)
    E = np.zeros((len(P), J)); V = np.zeros((len(P), J))
    T = np.zeros((len(P), J))
    history_buyers = pd.DataFrame(columns=['interval','Seller','Buyer','q_i','p_i','a_i','z_i','p_marg','v_i','u_i','c_i','diff'])
    history_sellers = pd.DataFrame()

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

        set_stable_order(M, seed_order=42)
        schedule_all_buyers_stable(M, t0=0.0)
        #schedule_all_buyers(M, t0=0.0)

        run(M, steps=2000, verbose=False)

        lad = LadderDict(M)
        df = lad.check_all_price_ladders(t=pm)
        if df.empty:
            print("No ladder tuples found — reasons logged.")
        else:
            with pd.option_context("display.max_rows", None, "display.width", 100):
                print("\nLADDER TUPLES:")
                print(df.round(3).to_string(index=False))
        lad.print_final_ladder_report(pm)

        plot_shared_buyer_surface_split(M, 6, sellers=(0,1), mode="theta_prime")
        plot_shared_buyer_surface_z0z1(M, 6, sellers=(0,1), mode="theta_prime")
        #schedule_all_buyers(M)

        metrics = compute_market_metrics(M)
        #print_round_from_metrics(metrics)
        rep = market_report_from(metrics)
        rep["interval"] = pm

        # 1) Global classification (one row per buyer)
        df_bstat = classify_buyers(M)
        n_zero_alloc_pos_bid = int(df_bstat["zero_alloc_pos_bid"].sum())
        n_zero_bid_no_gain   = int(df_bstat["zero_bid_no_gain"].sum())

        # 2) Per-seller loser counts
        losers_per_seller = per_seller_outbid_losers(M)

        rep["losers"] = losers_per_seller  # per seller
        print_df(rep)
        # Add global counts (same value on each seller row for convenience)
        rep["zero_alloc_pos_bid_total"] = n_zero_alloc_pos_bid
        rep["zero_bid_no_gain_total"]   = n_zero_bid_no_gain

        # metrics & reports
        E[li] = rep["E_j"].to_numpy()
        V[li] = rep["V_j"].to_numpy()
        T[li] = rep["p_star_j"].to_numpy()


    return M, P, E, V, T, history_buyers, history_sellers

def experiment1():
    I, J = 8, 2
    percents = [x / 100.0 for x in range(0, 101, 10)]  # 0 → 100
    base_seed=20405008
    Q_max=[60.0, 40.0]
    epsilon=2.5
    jitter=0.0

    M, P, E, V, T, H_buyers, H_sellers = run_experiment(
        I=I,
        J=J,
        percents=percents,
        Q_max=Q_max,
        epsilon=epsilon,
        reserve=[0.0, 0.0],
        steps=1500,
        base_seed=base_seed,
        jitter=jitter
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

    tbl_B = pd.DataFrame({
        f"Seller {j}": [f"{B[li, j]:.3f}" for li in range(len(P))]
        for j in range(J)
    }, index=[f"{int(100*p)}%" for p in P])

    print_df(tbl_E, header="E_j ± √V_j by percent (mean ± std)")
    print_df(tbl_B, header="Average Bid Price by percent (avg_p_bid)")

    tbl_res = pd.DataFrame({
        f"Seller {j}": [
            f"{E[li, j]:.3f} ± {np.sqrt(V[li, j]):.3f}  |  {B[li, j]:.3f}"
            for li in range(len(P))
        ]
        for j in range(J)
    }, index=[f"{int(100*p)}%" for p in P])

    print_df(tbl_res, header="E_j ± √V_j  |  avg marginal value (mean ± std  | avg bid price)")

    plot_prices_vs_percent(P, E, V, B, T,
                           labels=[f"Seller {j}" for j in range(J)],
                           config={"epsilon": epsilon,
                                    "base_seed": base_seed,
                                    "Q_max": Q_max,
                                    "jitter": jitter,
                                    "#buyers": I}
                           )


# -----------------
# Demo (minimal)
# -----------------
if __name__ == "__main__":
    run_sanity_ladder()
    experiment1()


