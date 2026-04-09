"""
File I/O for PSP simulation output.

Trajectory format (traj_NNN.dat)
---------------------------------
One row per recorded step. Columns:

  t  a{i}_{j}...  v{i} c{i} u{i}  ...  plo_{j} phi_{j} ...

where for each buyer i:
  a{i}_{j}  allocation from seller j
  v{i}      total valuation θ_i(z_i)
  c{i}      total PSP cost
  u{i}      utility  v{i} - c{i}

and for each seller j (appended after all buyer blocks):
  plo_{j}   lowest winning bid  p_j(t)  — lower bound of margin interval
  phi_{j}   highest losing bid  p̄_j(t) — upper bound of margin interval

The margin interval is (phi_j, plo_j); its width plo_j - phi_j measures
market tightness at seller j.

Prices / summary format (prices.dat)
--------------------------------------
One row per connectivity level (or experiment condition). Columns:

  e  pavg_{j} pstd_{j} plo_{j} phi_{j} ...  vtot ctot utot az

where:
  e           connectivity level (fraction, 0–1)
  pavg_{j}    allocation-weighted mean bid price at seller j
  pstd_{j}    allocation-weighted std of bid price at seller j
  plo_{j}     lowest winning bid (clearing price)
  phi_{j}     highest losing bid
  vtot        total market valuation (sum over buyers)
  ctot        total market cost
  utot        total market utility
  az          average per-buyer allocation (z_i mean)

Configuration format (run.conf)
---------------------------------
JSON, one file per experiment run.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from typing import Dict


# ---------------------------------------------------------------------------
# Header builders
# ---------------------------------------------------------------------------

def _traj_header(I: int, J: int) -> str:
    cols = ["t"]
    for i in range(I):
        for j in range(J):
            cols.append(f"a{i}_{j}")
        cols += [f"v{i}", f"c{i}", f"u{i}"]
    for j in range(J):
        cols += [f"plo_{j}", f"phi_{j}"]
    return "#" + "  ".join(cols)


def _prices_header(J: int) -> str:
    cols = ["e"]
    for j in range(J):
        cols += [f"pavg_{j}", f"pstd_{j}", f"plo_{j}", f"phi_{j}"]
    cols += ["vtot", "ctot", "utot", "az"]
    return "#" + "  ".join(cols)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def write_traj_header(path: str, M: Dict) -> None:
    """Create (or overwrite) a trajectory file and write the header."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(_traj_header(M["I"], M["J"]) + "\n")


def append_traj_row(path: str, t: float, metrics) -> None:
    """Append one row to an open trajectory file from a MarketMetrics snapshot."""
    I, J = metrics.a_mat.shape
    row = [f"{t:.6g}"]
    for i in range(I):
        for j in range(J):
            row.append(f"{metrics.a_mat[i, j]:.6g}")
        row += [
            f"{metrics.buyer_value[i]:.6g}",
            f"{metrics.buyer_cost[i]:.6g}",
            f"{metrics.buyer_util[i]:.6g}",
        ]
    for j in range(J):
        row += [f"{metrics.plo_j[j]:.6g}", f"{metrics.phi_j[j]:.6g}"]
    with open(path, "a") as f:
        f.write("  ".join(row) + "\n")


def load_traj(path: str) -> pd.DataFrame:
    """Load a trajectory file into a DataFrame."""
    with open(path) as f:
        header = f.readline().lstrip("#").split()
    df = pd.read_csv(path, comment="#", sep=r"\s+", names=header)
    return df


# ---------------------------------------------------------------------------
# Prices / summary
# ---------------------------------------------------------------------------

def write_prices(path: str, rows: list[dict]) -> None:
    """Write the prices summary file from a list of row dicts."""
    if not rows:
        return
    J = sum(1 for k in rows[0] if k.startswith("pavg_"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(_prices_header(J) + "\n")
        for row in rows:
            cols = ["e"]
            for j in range(J):
                cols += [f"pavg_{j}", f"pstd_{j}", f"plo_{j}", f"phi_{j}"]
            cols += ["vtot", "ctot", "utot", "az"]
            vals = [f"{row[c]:.6g}" for c in cols]
            f.write("  ".join(vals) + "\n")


def load_prices(path: str) -> pd.DataFrame:
    """Load a prices summary file into a DataFrame."""
    with open(path) as f:
        header = f.readline().lstrip("#").split()
    df = pd.read_csv(path, comment="#", sep=r"\s+", names=header)
    return df


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def write_conf(path: str, conf: dict) -> None:
    """Write experiment configuration as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(conf, f, indent=2, default=_json_default)
        f.write("\n")


def load_conf(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Prices row builder (convenience)
# ---------------------------------------------------------------------------

def prices_row_from(e: float, metrics) -> dict:
    """Build a prices row dict from a MarketMetrics snapshot and level e."""
    I, J = metrics.a_mat.shape
    row = {"e": e}
    for j in range(J):
        a_col = metrics.a_mat[:, j]
        p_col = metrics.bid_p[:, j]
        A = float(a_col.sum())
        if A > 0:
            pavg = float((a_col * p_col).sum() / A)
            pstd = float(np.sqrt((a_col * (p_col - pavg) ** 2).sum() / A))
        else:
            pavg = pstd = 0.0
        row[f"pavg_{j}"] = pavg
        row[f"pstd_{j}"] = pstd
        row[f"plo_{j}"]  = float(metrics.plo_j[j])
        row[f"phi_{j}"]  = float(metrics.phi_j[j])
    row["vtot"] = float(metrics.buyer_value.sum())
    row["ctot"] = float(metrics.buyer_cost.sum())
    row["utot"] = float(metrics.buyer_util.sum())
    row["az"]   = float(metrics.buyer_alloc.mean())
    return row
