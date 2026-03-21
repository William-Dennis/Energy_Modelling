# Phase 10b: Behaviour Inventory

## Status: COMPLETE

## Objective

Catalogue the full set of behaviours exhibited by the current synthetic futures
market before attempting deeper causal explanation or intervention.

## Why This Matters

The project can no longer treat market behaviour as a single binary outcome such
as "converges" or "oscillates". The current artifacts suggest several distinct
behaviour classes that need separate explanation.

## Candidate Behaviour Classes

- fast convergence
- slow convergence
- damped non-convergence within the current iteration budget
- absorbing carry-forward convergence (all weights go to zero)
- active-strategy collapse
- early-iteration accuracy improvement followed by later degradation
- cluster switching / regime switching

## Core Metrics

Per-iteration metrics extracted by `scripts/phase10b_behaviour_inventory.py`:

- **convergence_delta**: max|P_k - P_{k-1}|
- **mae**: mean|P_market - P_real|
- **rmse**: sqrt(mean((P_market - P_real)^2))
- **bias**: mean(P_market - P_real)
- **active_count**: number of strategies with positive weight
- **weight_entropy**: -sum(w * ln(w)) for w > 0
- **top1_weight**: max strategy weight
- **top5_concentration**: sum of top 5 weights
- **total_profit_spread**: max(profit) - min(profit)
- **median_profit**: median of all strategy profits
- **max_profit / min_profit**: extreme strategy profit values

## Findings

### Year 2024: Oscillating Non-Convergence

| Metric | Value |
|--------|-------|
| Converged | No |
| Iterations | 500 |
| Final delta | 1.7729 |
| Initial MAE | 21.26 EUR/MWh |
| Best MAE | 10.27 EUR/MWh (iter 90) |
| Final MAE | 10.30 EUR/MWh |
| MAE degraded after best | No (within 5% tolerance) |
| Initial active strategies | 60 |
| Final active strategies | 8 |
| Min active strategies | 4 |
| Active collapse (<=2) | No |
| Monotone damped | No |
| Oscillating | Yes |
| Initial weight entropy | 3.80 |
| Final weight entropy | 1.85 |
| Max top-1 weight | 0.84 |

**Narrative**: The 2024 market runs for the full 500-iteration budget without
converging. The convergence delta oscillates around ~1.8 in later iterations,
never settling below the 0.01 threshold. The active strategy set collapses
rapidly from 60 to ~5-8 within the first 50 iterations and remains in that
range, but never fully collapses to zero. MAE improves from 21.3 to ~10.3 in
the first ~90 iterations, then plateaus — the market finds a reasonable price
level but cannot stabilise it precisely. Weight concentration increases
dramatically: top-1 weight peaks at 0.84 and entropy drops from 3.8 to 1.85.

**Behaviour classification**: `oscillating_non_convergence`

### Year 2025: Absorbing Collapse

| Metric | Value |
|--------|-------|
| Converged | Yes |
| Iterations | 327 |
| Final delta | ~0 (2.8e-14) |
| Initial MAE | 19.88 EUR/MWh |
| Best MAE | 8.89 EUR/MWh (iter 230) |
| Final MAE | 8.91 EUR/MWh |
| MAE degraded after best | No (within 5% tolerance) |
| Initial active strategies | 60 |
| Final active strategies | 0 |
| Min active strategies | 0 |
| Active collapse (<=2) | Yes (at iter 61) |
| Monotone damped | No |
| Oscillating | Yes |
| Absorbing state | Yes |
| Initial weight entropy | 3.81 |
| Final weight entropy | 0.00 |
| Max top-1 weight | 1.00 |

**Narrative**: The 2025 market converges at iteration 327, but convergence
occurs because all strategies are eliminated (zero active). The engine enters
an absorbing state: with no active strategies, the carry-forward rule
`P_{k+1} = P_k` trivially satisfies the convergence criterion. The active
count drops rapidly: from 60 to ~7 by iter 50, to <=2 by iter 61, and
oscillates between 0-7 active strategies through the middle iterations before
reaching all-zero at iter 326. MAE improves to ~8.9 (best at iter 230) and
the final price is essentially frozen at the last non-trivial iteration's
price. Weight entropy drops to 0 and top-1 weight reaches 1.0 in several
intermediate iterations before complete collapse.

**Behaviour classification**: `absorbing_collapse`

### Side-by-Side Comparison

| Dimension | 2024 | 2025 |
|-----------|------|------|
| Converged | No | Yes (absorbing) |
| Final MAE | 10.30 | 8.91 |
| Best MAE iter | 90 | 230 |
| Active at end | 8 | 0 |
| Collapse to <=2 | No | Yes (iter 61) |
| Delta pattern | Oscillating ~1.8 | Oscillating then zero |
| Weight entropy trend | 3.8 -> 1.85 | 3.8 -> 0.0 |
| Behaviour label | oscillating_non_convergence | absorbing_collapse |

### Common Dynamics

Both years share:
1. **Rapid initial active-count drop**: 60 -> ~5-8 within 50 iterations
2. **Oscillating delta pattern**: significant sign changes in delta differences
3. **Strong weight concentration**: top-1 weight reaches 0.8-1.0
4. **MAE improvement then plateau**: accuracy improves significantly in early
   iterations, then flattens; degradation after best is minimal

Key differences:
1. 2024 stabilises at ~5-8 active strategies; 2025 eventually collapses to 0
2. 2025's collapse to zero triggers the absorbing convergence condition
3. 2024 has a higher final MAE (10.3 vs 8.9)

## High-Priority Behaviours for Phase 10c-10e

1. **Active-strategy elimination dynamics**: Why do strategies lose profitability
   across iterations? What triggers the cascade from 60 down to single digits?
   This is the dominant dynamic in both years.

2. **Oscillating non-convergence (2024)**: The delta oscillates around ~1.8
   without damping. What drives this persistent oscillation? Is it a small group
   of strategies alternating between profitable and unprofitable?

3. **Absorbing-state convergence (2025)**: The market "converges" not via
   consensus but via elimination of all participants. Is this a meaningful
   equilibrium or a degenerate outcome?

4. **Early accuracy improvement then plateau**: Both years show MAE cutting in
   half within ~100 iterations. What mechanism drives the early improvement and
   why does it stall?

5. **Weight concentration to near-monopoly**: Top-1 weight reaching 0.84 (2024)
   and 1.00 (2025) suggests the market may be dominated by one strategy in late
   iterations. Which strategy dominates, and does rotation occur?

## Outputs

- `scripts/phase10b_behaviour_inventory.py` — automated extraction and classification
- `data/results/phase10/behaviour_inventory.csv` — 827 rows, per-iteration metrics for both years
- `data/results/phase10/behaviour_summary.csv` — run-level classification summary
- `tests/backtest/test_behaviour_inventory.py` — 17 tests covering metric extraction and classification

## Checklist

- [x] Load `market_2024.pkl` and `market_2025.pkl`, extract per-iteration arrays
- [x] Compute per-iteration metric panels for 2024 and 2025:
  - convergence delta, MAE, RMSE, bias vs real prices
  - active strategy count, top-1 weight, top-5 concentration, weight entropy
- [x] Classify each run into one or more behaviour types:
  - converged / non-converged
  - monotone damped / oscillating / regime-switching
  - active-strategy collapse vs stable active set
- [x] Compare 2024 and 2025 behaviour side by side
- [x] Identify the 3-5 highest-information behaviours to explain in Phase 10c-10e
- [x] Write `scripts/phase10b_behaviour_inventory.py` to automate the above
- [x] Save results to `data/results/phase10/behaviour_inventory.csv`
- [x] Save behaviour summary to `data/results/phase10/behaviour_summary.csv`
- [x] Write unit tests (`tests/backtest/test_behaviour_inventory.py` — 17 tests)
