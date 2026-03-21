# Phase 10c: Mechanism Attribution

## Status: PLANNED

## Objective

Determine which parts of the market mechanism are responsible for each observed
behaviour in the current synthetic futures market.

## Target Mechanisms

- sign-based trading rule
- total-profit scoring across the full window
- positive-profit truncation
- linear weight normalisation
- weighted-average forecast aggregation
- EMA dampening via `ema_alpha`
- initialization from `last_settlement_price`

## Key Questions

1. Which mechanism is necessary for active-strategy collapse?
2. Which mechanism creates or amplifies oscillation?
3. Does dampening reduce oscillation by changing the attractor, or only by
   slowing movement?
4. How sensitive are outcomes to initialization versus update dynamics?

## Planned Methods

- `ema_alpha` sweeps under the current strategy pool
- initialization counterfactuals
- strategy-family ablations
- extreme-day masking
- top-cluster removal experiments

## Checklist

- [ ] Define mechanism-isolation experiments with clear hypotheses
- [ ] Run `ema_alpha` sensitivity sweep: {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
  - Record: convergence (Y/N), iterations, final delta, MAE, active count
- [ ] Run initialization sensitivity tests:
  - default (last_settlement), forecast mean, constant, random
- [ ] Run strategy-family ablations:
  - remove all ML strategies, remove all mean-reversion, remove all momentum, etc.
  - measure change in convergence and MAE
- [ ] Run extreme-day masking:
  - identify top-5 most volatile days, re-run with them excluded
- [ ] For each mechanism, answer: does disabling/modifying it change the
  convergence class (e.g., from non-converged to converged)?
- [ ] Write `scripts/phase10c_mechanism_attribution.py` to automate sweeps
- [ ] Save results to `data/results/phase10/mechanism_attribution.csv`
- [ ] Summarise which mechanisms explain which behaviours in a table
