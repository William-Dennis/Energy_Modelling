# Phase 10e: Sentinel Case Studies

## Status: COMPLETE

## Objective

Produce a small set of iteration-by-iteration case studies that make the market
behaviour understandable in concrete causal terms.

## Purpose

Aggregate metrics alone will not explain the system clearly. This sub-phase is
for narrative-quality evidence: a few cases that show exactly how prices,
profits, weights, and strategy clusters interact.

## Candidate Case Types

- a high-volatility 2024 window with persistent non-convergence
- a 2025 window approaching zero-active convergence
- a day where iter-0 is better than the final iteration
- a day dominated by a single cluster shift

## Results

### 5 Sentinel Cases Analysed

| Case ID | Type | Year | Trace Rows | Key Finding |
|---------|------|------|------------|-------------|
| hvnc_2024 | High-volatility non-convergence | 2024 | 100 | MAE improves from 50.2 to 31.5 but delta oscillates (4.65 to 1.40); active count collapses 60 to 8; 57 leadership changes across 7 leaders |
| clsw_2024 | Cluster switching | 2024 | 30 | 3 leadership transitions in 30 iters: SVMDirection -> Ridge -> Lasso Cal -> Stacked Ridge Meta; MAE improves 21.3 to 10.4; active collapses 60 to 5 |
| ealost_2024 | Early accuracy lost | 2024 | 100 | On 5 focus dates: iter-0 MAE = 3.30, final MAE = 23.75 -- a 7x degradation; iterative repricing optimises aggregate profit, not per-date accuracy |
| zact_2025 | Zero-active convergence | 2025 | 26 | 4 active strategies at iter 301 to 0 at iter 326; MAE stable (8.92 to 8.91); 17 leadership changes in death spiral; absorbing state reached |
| ealost_2025 | Early accuracy lost | 2025 | 100 | On 5 focus dates: iter-0 MAE = 3.11, final MAE = 18.48; 60 active strategies collapse to 8; same aggregate-vs-local accuracy tension as 2024 |

### Key Causal Mechanisms Identified

1. **Oscillation from correlated ML forecasts** (hvnc_2024): The ML regression
   cluster issues highly correlated forecasts that pull the market price in the
   same direction simultaneously. Combined with EMA dampening, this creates
   persistent overshooting/undershooting that prevents convergence.

2. **Leadership instability from profit truncation** (clsw_2024): Small profit
   differences between strategies are amplified by the positive-profit truncation
   rule into large weight shifts. The top position flips 3 times in 30 iterations,
   each time altering the weighted forecast average.

3. **Aggregate vs local accuracy trade-off** (ealost_2024, ealost_2025): The
   market optimises for aggregate profitability across all dates, not per-date
   accuracy. Strategies that are profitable overall can still have poor forecasts
   on specific dates, and the weighting scheme cannot down-weight them day-by-day.
   This produces counter-intuitive 7x MAE degradation on specific dates.

4. **Absorbing collapse via one-way ratchet** (zact_2025): Positive-profit
   truncation creates a one-way ratchet. Once all strategies accumulate net-
   negative total profit, all weights become zero and the market price freezes.
   No strategy can recover because frozen prices produce zero net profit.

### Leadership Trajectory (2024 Cluster Switching)

```
Iter  0: SVMDirection       (weight 0.030)
Iter  1: Ridge Regression   (weight 0.056)
Iter 10: Lasso Cal Augment  (weight ~0.06)
Iter 19: Stacked Ridge Meta (weight 0.413)
```

All leaders are from the ML regression / ensemble cluster, confirming Phase 10d's
finding that this cluster captures >90% of market weight.

## Checklist

- [x] Select 3-5 sentinel windows based on Phase 10b/10d findings:
  - [x] at least one high-volatility 2024 window with persistent non-convergence
  - [x] at least one 2025 window approaching zero-active convergence
  - [x] at least one window where iter-0 MAE < final MAE (early accuracy lost)
  - [x] at least one cluster-switching episode from Phase 10d
- [x] For each sentinel window, build iteration-level trace showing:
  - [x] market price trajectory vs real price
  - [x] active strategy count and top-3 strategy weights
  - [x] convergence delta per iteration
  - [x] which strategy cluster is dominant at each step
- [x] Write plain-language causal explanation for each case:
  - [x] what triggered the behaviour
  - [x] which mechanism (from Phase 10c) is responsible
  - [x] whether the outcome is desirable or pathological
- [x] Produce 1-page narrative summary per sentinel case
- [x] Write `scripts/phase10e_sentinel_traces.py`
- [x] Save trace data to `data/results/phase10/sentinel_traces/`

## Test Coverage

23 tests in `tests/backtest/test_sentinel_traces.py`:
- `TestFindHighVolatilityWindow` (3 tests)
- `TestFindActiveCollapseWindow` (2 tests)
- `TestFindEarlyAccuracyDates` (2 tests)
- `TestFindClusterSwitchingWindow` (2 tests)
- `TestBuildIterationTrace` (4 tests)
- `TestGenerateNarrative` (5 tests)
- `TestBuildCaseSummary` (2 tests)
- `TestSelectSentinelCases` (3 tests)

## Artifacts

- `scripts/phase10e_sentinel_traces.py` — analysis script
- `tests/backtest/test_sentinel_traces.py` — 23 tests
- `data/results/phase10/sentinel_summaries.csv` — 5-row case summary
- `data/results/phase10/sentinel_traces/hvnc_2024.csv` — 100-row trace
- `data/results/phase10/sentinel_traces/clsw_2024.csv` — 30-row trace
- `data/results/phase10/sentinel_traces/ealost_2024.csv` — 100-row trace
- `data/results/phase10/sentinel_traces/zact_2025.csv` — 26-row trace
- `data/results/phase10/sentinel_traces/ealost_2025.csv` — 100-row trace
