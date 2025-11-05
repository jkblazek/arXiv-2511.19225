
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
    det = M["deterministic_sched"]
    perm = M["perm"]
    rng = M["rng"]
    for i in range(M["I"]):
        if det:
            t_i = t0 + float(i)
        else:
            t_i = t0 + float(rng.random() * M.get("jitter", 0.0))
        push(M, t_i, BUYER_COMPUTE, (i,))

def schedule_all_buyers_stable(M: dict, seed_order: int, t0: float = 0.0):
    I = int(M["I"])
    M["perm"] = np.random.default_rng(seed_order).permutation(I).astype(int)
    pos = np.empty(I, dtype=int)
    pos[M["perm"]] = np.arange(I)
    M["perm_pos"] = pos
    M["jitter"] = 0.0
    M["pq"].clear()
    for k, i in enumerate(M["perm"]):
        push(M, t0 + float(k), BUYER_COMPUTE, (int(i),))

# ------------------------------
# Events
# ------------------------------

def handle_buyer_compute(M: Dict, i: int, verbose: bool = False, debug: bool = False):
    t0 = M["t"]
    u0 = u_i_current(i, M)
    q_row, p_row, feasible, u1 = joint_best_response_plan(i, M)
    if "deterministic_sched" in M and M["deterministic_sched"]:
        push(M, t0 + 1.0, BUYER_COMPUTE, (i,))
    elif "perm_pos" in M:
        slot = 1e-9 * int(M["perm_pos"][i])
        push(M, float(np.floor(t0)) + 1.0 + slot, BUYER_COMPUTE, (i,))
    else:
        push(M, t0 + 1.0 + M["rng"].random() * M["jitter"], BUYER_COMPUTE, (i,))
    tol = M["tol"]
    if (not feasible) or (u1 <= u0 + tol):
        if verbose:
            print(f"[event] BUYER_COMPUTE i={i}: no update (feasible={feasible}, Δu={u1-u0:.6f})")
        return False
    dq = np.abs(M["bid_q"][i] - q_row)
    dp = np.abs(M["bid_p"][i] - p_row)
    bid_move = max(dq.max(initial=0.0), dp.max(initial=0.0))
    if bid_move <= tol:
        if verbose:
            print(f"[event] BUYER_COMPUTE i={i}: within tol (Δu={u1-u0:.6f}, move={bid_move:.3e})")
        return False
    M["gen"][i] += 1
    gen = int(M["gen"][i])
    J = M["J"]
    if "instant_post" in M and M["instant_post"]:
        changed = False
        for j in range(J):
            if (dq[j] > tol) or (dp[j] > tol):
                M["bid_q"][i, j] = q_row[j]
                M["bid_p"][i, j] = p_row[j]
                changed = True
        if changed:
            M["events_since_apply"] = 0
            if verbose:
                print(f"[event] BUYER_COMPUTE i={i}: applied (gen={gen}, Δu={u1-u0:.6f}, move={bid_move:.3e})")
        else:
            if verbose:
                print(f"[event] BUYER_COMPUTE i={i}: nothing to apply after tol check (move={bid_move:.3e})")
        return changed
    num_posts = 0
    for j in range(J):
        if (dq[j] > tol) or (dp[j] > tol):
            num_posts += 1
            t_post = t0 + M["rng"].random() * M["jitter"]
            push(M, t_post, POST_BID, (i, j, q_row[j], p_row[j], gen))
    if num_posts:
        M["pending_posts"] += num_posts
        if verbose:
            print(f"[event] BUYER_COMPUTE i={i}: accepted (gen={gen}), posts={num_posts}, Δu={u1-u0:.6f}, move={bid_move:.3e}")
        return True
    if verbose:
        print(f"[event] BUYER_COMPUTE i={i}: nothing to post (move={bid_move:.3e})")
    return False


def handle_post_bid(M: Dict, i: int, j: int, q: float, p: float, gen: int, verbose: bool = False, debug: bool = False):
    if not M["adj"][i, j]:
        if M.get("pending_posts", 0) > 0:
            M["pending_posts"] -= 1
        if debug:
            print(f"[event] POST_BID blocked (no edge): i={i}, j={j}")
        return False
    if (M["gen"][i] != gen):
        if verbose:
            print(f"[event] POST_BID stale drop: i={i}, j={j}, gen={gen} (current gen={M['gen'][i]})")
        return False
    # fresh -> apply
    M["bid_q"][i,j] = q; M["bid_p"][i,j] = p
    if M.get("pending_posts", 0) > 0:
        M["pending_posts"] -= 1
    M["events_since_apply"] = 0
    if verbose:
        print(f"[event] POST_BID applied: i={i}, j={j}, (q,p)=({q:.4f},{p:.4f})")
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

    for it in range(steps):
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

    return it
