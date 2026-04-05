# Bipartiteness in Progressive Second-Price Multi-Auction Networks with Perfect Substitute

Simulation code supporting the paper:

> Blazek, J. and Harris, F. C. Jr. "Bipartiteness in Progressive Second-Price Multi-Auction Networks with Perfect Substitute." arXiv:2511.19225v2, January 2026.

Google Colab notebook: https://colab.research.google.com/drive/1ie2qiMakIYibcY3UXx13irCagkhHsN9A

---

## Repository Structure

```
v3/          Canonical modular implementation (use this)
  market.py    PSP auction mechanics: valuations, allocations, costs, best-response
  network.py   Async event-driven simulation engine (priority-queue scheduler)
  ladder.py    Price-ladder verification (Lemma 5.2 / Section 6.2)
  metrics.py   Market state snapshots, buyer/seller reporting
  helpers.py   Seller-side graph utilities (shells, p*, winners)
  init.py      Market initialization, adjacency generation, seed management
  plot.py      Visualization: diagnostics, connectivity, utility surfaces, price plots
  demo.py      Entry point: sanity check, single demo, connectivity experiment

colab/
  arxiv.py     Self-contained standalone script for Google Colab

connectivity/  Variant focused on connectivity sweep experiments
ladder_check/  Variant focused on price-ladder unit tests
v1/, v2/       Earlier versions (kept for reference)
```

## Quickstart

```bash
pip install -r requirements.txt
cd v3
python demo.py
```

`demo.py` runs `experiment1()` by default, which reproduces the connectivity sweep
(Figure 8 in the paper). To run the price-ladder sanity check (Section 6.2), call
`run_sanity_ladder()` instead.

## Experiments

### Price-Ladder Verification (Section 6.2)

```python
from demo import run_sanity_ladder
run_sanity_ladder()
```

Initializes a 4-buyer × 2-seller market with hand-crafted bids and verifies the
monotone price-ladder condition `p*_ℓ ≤ p_k < p*_j ≤ p_i` (Lemma 5.2).

### Connectivity Sweep (Section 6.3)

```python
from demo import experiment1
experiment1()
```

Sweeps buyer–seller overlap from 0 % to 100 % in 5 % increments (I=8 buyers,
J=2 sellers, base_seed=20405008) and plots marginal value vs. connectivity
percentage (Figure 8).

## Reproducibility

All random draws are seeded. The base seed for the connectivity experiment is
`20405008`; per-level adjacency seeds are derived as `base_seed + 1009 * level_index`.
Buyer valuations use independent per-buyer RNG streams keyed on `seed + i * 1_000_003`.

## Dependencies

See `requirements.txt`. Requires Python ≥ 3.10 and NumPy ≥ 1.25 (uses
`np.trapezoid`).

## Citation

```bibtex
@article{blazek2026bipartiteness,
  title   = {Bipartiteness in Progressive Second-Price Multi-Auction Networks
             with Perfect Substitute},
  author  = {Blazek, Jordana and Harris, Frederick C., Jr.},
  journal = {arXiv preprint arXiv:2511.19225},
  year    = {2026}
}
```
