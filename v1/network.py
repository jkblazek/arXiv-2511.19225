
import heapq
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from market import u_i_current, joint_best_response_plan

# ------------------------------
# Event types
# ------------------------------
BUYER_COMPUTE = 1
POST_BID = 2

# ------------------------------
# Priority queue helpers
# ------------------------------

def push(M: Dict, t: float, etype: int, payload: Tuple):
    M["seq"] += 1
    heapq.heappush(M["pq"], (float(t), int(M["seq"]), int(etype), payload))

def pop(M: Dict):
    return heapq.heappop(M["pq"]) if M["pq"] else None

def schedule_all_buyers(M: Dict, t0: float = 0.0):
    """Schedule one BUYER_COMPUTE per buyer at start time t0.
    If M["deterministic_sched"] is True, schedule at integer offsets (t0 + i).
    Otherwise, use jittered times for async interleaving.
    """
    det = bool(M.get("deterministic_sched", False))
    rng = M["rng"]
    for i in range(M["I"]):
        if det:
            t_i = t0 + float(i)
        else:
            t_i = t0 + float(rng.random() * M.get("jitter", 0.0))
        push(M, t_i, BUYER_COMPUTE, (i,))

def set_stable_order(M: dict, seed_order: int) -> None:
    """Install a stable buyer visitation order into the market."""
    I = int(M["I"])
    M["perm"] = np.random.default_rng(seed_order).permutation(I).astype(int)
    # Position map for micro-ordering on ties:
    pos = np.empty(I, dtype=int)
    pos[M["perm"]] = np.arange(I)
    M["perm_pos"] = pos
    M["jitter"] = 0.0  # make sure no randomization sneaks in

def schedule_all_buyers_stable(M: dict, t0: float = 0.0):
    """Schedule BUYER_COMPUTE once in the stable order, no jitter."""
    assert "perm" in M, "call set_stable_order(M, seed_order) first"
    M["pq"].clear()
    for k, i in enumerate(M["perm"]):
        # strictly increasing times guarantee deterministic processing order
        push(M, t0 + float(k), BUYER_COMPUTE, (int(i),))



# ------------------------------
# Event handlers
# ------------------------------

def handle_buyer_compute(M: Dict, i: int, verbose: bool = False, debug: bool = False):
    base_t = M["t"]
    old_u   = u_i_current(i, M)
    q_row, p_row, feasible, u_new = joint_best_response_plan(i, M)

    # --- schedule next compute exactly like before ---
    det = bool(M.get("deterministic_sched", False))
    if det:
        next_t = base_t + 1.0
        push(M, next_t, BUYER_COMPUTE, (i,))
    elif "perm_pos" in M:  # stable mode
        slot = 1e-9 * float(int(M["perm_pos"][i]))  # tiny, deterministic tie-break
        push(M, float(np.floor(M["t"])) + 1.0 + slot, BUYER_COMPUTE, (i,))
    else:
        next_t = base_t + 1.0 + float(M["rng"].random() * M.get("jitter", 0.0))
        push(M, next_t, BUYER_COMPUTE, (i,))

    # --- thresholds ---
    eta_u = float(M.get("eta_u", M.get("epsilon", 1e-4)))  # require > eta_u utility gain
    qtol  = float(M.get("qtol", 1e-9))
    ptol  = float(M.get("ptol", 1e-9))

    # Quick exit if plan is infeasible or utility gain too small
    if (not feasible) or (u_new <= old_u + eta_u):
        if verbose:
            print(f"BUYER_COMPUTE i={i}: no accepted update "
                  f"(feasible={feasible}, Δu={u_new-old_u:.6f}, eta_u={eta_u:.1e})")
        return False

    J = M["J"]
    adj = M["adj"]

    # Compute max bid move (∞-norm over connected auctions)
    max_dq = 0.0
    max_dp = 0.0
    for j in range(J):
        if not adj[i, j]:
            continue
        dq = abs(float(M["bid_q"][i, j]) - float(q_row[j]))
        dp = abs(float(M["bid_p"][i, j]) - float(p_row[j]))
        if dq > max_dq: max_dq = dq
        if dp > max_dp: max_dp = dp
    bid_move = max(max_dq, max_dp)

    # Require a material bid movement too
    if bid_move <= max(qtol, ptol):
        if verbose:
            print(f"BUYER_COMPUTE i={i}: plan equals current bids within tol "
                  f"(Δu={u_new-old_u:.6f}, bid_move={bid_move:.3e}, qtol={qtol:.1e}, ptol={ptol:.1e})")
        return False

    # --- Accept plan ---
    M["gen"][i] += 1
    gen_now = int(M["gen"][i])

    changed = False
    instant = bool(M.get("instant_post", False))

    if instant:
        # Apply changes immediately; no POST_BID events.
        for j in range(J):
            if not adj[i, j]:
                continue
            dq = abs(float(M["bid_q"][i, j]) - float(q_row[j]))
            dp = abs(float(M["bid_p"][i, j]) - float(p_row[j]))
            if (dq > qtol) or (dp > ptol):
                M["bid_q"][i, j] = float(q_row[j])
                M["bid_p"][i, j] = float(p_row[j])
                changed = True

        if changed:
            # Reset apply idle counter just like POST_BID would
            M["events_since_apply"] = 0
            if verbose:
                print(f"BUYER_COMPUTE i={i}: applied immediately "
                      f"(gen={gen_now}, Δu={u_new-old_u:.6f}, bid_move={bid_move:.3e})")
        else:
            if verbose:
                print(f"BUYER_COMPUTE i={i}: nothing to apply after tol check "
                      f"(bid_move={bid_move:.3e})")
        return changed

    # --- Asynchronous mode: schedule POST_BID events for changed auctions only ---
    num_posts = 0
    for j in range(J):
        if not adj[i, j]:
            continue
        dq = abs(float(M["bid_q"][i, j]) - float(q_row[j]))
        dp = abs(float(M["bid_p"][i, j]) - float(p_row[j]))
        if (dq > qtol) or (dp > ptol):
            changed = True
            num_posts += 1
            t_post = base_t + float(M["rng"].random() * M.get("jitter", 0.0))
            push(M, t_post, POST_BID, (i, j, float(q_row[j]), float(p_row[j]), gen_now))

    if num_posts:
        M["pending_posts"] = M.get("pending_posts", 0) + num_posts

    if verbose:
        if changed:
            print(f"BUYER_COMPUTE i={i}: accepted plan (gen={gen_now}), "
                  f"posts scheduled: {num_posts}, Δu={u_new-old_u:.6f}, bid_move={bid_move:.3e}")
        else:
            print(f"BUYER_COMPUTE i={i}: plan equals current bids after tol check, nothing to post "
                  f"(bid_move={bid_move:.3e})")

    return changed

def handle_post_bid(M: Dict, i: int, j: int, q: float, p: float, gen: int, verbose: bool = False, debug: bool = False):
    # Block if (i,j) is not connected
    if not M["adj"][i, j]:
        if M.get("pending_posts", 0) > 0:
            M["pending_posts"] -= 1
        if debug:
            print(f"POST_BID blocked (no edge): i={i}, j={j}")
        return False

    # Generation guard
    if (int(M["gen"][i]) != int(gen)):
        if verbose:
            print(f"POST_BID stale drop: i={i}, j={j}, gen={gen} (current gen={int(M['gen'][i])})")
        return False
    # fresh -> apply
    M["bid_q"][i,j] = float(q); M["bid_p"][i,j] = float(p)

    if M.get("pending_posts", 0) > 0:
        M["pending_posts"] -= 1
    M["events_since_apply"] = 0
    if verbose:
        print(f"POST_BID applied: i={i}, j={j}, (q,p)=({q:.4f},{p:.4f})")
    return True


# ------------------------------
# Engine run loop
# ------------------------------

def run(M: Dict, steps: int = 1000, verbose: bool = False, *, idle_event_limit: int | None = None):
    """Run the engine.

    If M["instant_post"] is True, only BUYER_COMPUTE events will be present
    (POST_BID is unused). If M["deterministic_sched"] is True, scheduling uses
    deterministic times (no jitter), improving reproducibility.

    break_on_convergence: stop when no POST_BID has applied for
      idle_event_limit events and there are no pending POST_BIDs.
    idle_event_limit: default = I (set at runtime if None).
    """
    if idle_event_limit is None:
        idle_event_limit = int(M["I"]) if M["I"] else 4

    for _ in range(steps):
        item = pop(M)
        if item is None:
            break
        t, _, etype, payload = item
        M["t"] = float(t)
        if etype == BUYER_COMPUTE:
            (i,) = payload
            handle_buyer_compute(M, int(i), verbose=verbose)
            # count toward idle if nothing applied elsewhere
            M["events_since_apply"] = M.get("events_since_apply", 0) + 1
        elif etype == POST_BID:
            i, j, q, p, gen = payload
            handle_post_bid(M, int(i), int(j), float(q), float(p), int(gen), verbose=verbose)
            # events_since_apply reset inside handler on apply
        else:
            raise RuntimeError(f"Unknown event type {etype}")

        # Convergence check: no pending posts and long idle since last apply
        pend = M.get("pending_posts", 0)
        if pend == 0 and M.get("events_since_apply", 0) >= idle_event_limit:
            if verbose:
                print(f"Converged: no price updates for {M['events_since_apply']} events; pending_posts=0")
            break

