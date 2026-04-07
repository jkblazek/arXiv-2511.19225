"""
Free-Lunch Experiment  (placeholder)

Investigates the effect of the "fair" (equal count-split) tie-breaking rule
versus the correct QJC (quantity-proportional) rule on PSP auction outcomes.

The "fair" split gives buyers at a tie price equal shares of residual capacity
regardless of their requested quantity. This breaks the PSP/VCG externality
calculation and can allow buyers to obtain more resource than their bid warrants
— a "free lunch" relative to the truthful QJC equilibrium.

To activate the fair split, set M["tie_policy"] = "fair" after market creation.
The default (M["tie_policy"] = "qjc") preserves incentive-compatibility.

TODO:
  - Compare equilibrium prices, allocations, and utilities under "qjc" vs "fair"
    across a range of market sizes and connectivity levels.
  - Measure the utility gain available to a deviating buyer under "fair" split
    (the "free lunch" magnitude).
  - Relate findings to the Julia free-lunch experiments.
"""

# Placeholder — implementation pending.
raise NotImplementedError("free_lunch experiment not yet implemented.")
