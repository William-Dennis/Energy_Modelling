# Phase 10f: Strategy Robustness Analysis

## Status: COMPLETE

## Objective

Distinguish strategies that are merely strong in standalone backtests from those
that remain useful inside the synthetic futures market.

## Working Definition

A robust strategy should do more than earn standalone PnL. It should also be
informative after market repricing and ideally improve the aggregate market.

## Results

### Classification Summary

| Classification | 2024 | 2025 | Description |
|---------------|------|------|-------------|
| robust | 2 | 1 | Strong standalone + strong market-adjusted + positive market contribution |
| standalone_only | 0 | 1 | Strong standalone but weak market-adjusted |
| redundant | 33 | 45 | High forecast correlation (>0.95) with another strategy |
| destabilising | 25 | 2 | Removing strategy improves market MAE |
| weak | 7 | 18 | Below median in both standalone and market-adjusted |

### Key Finding: Massive Redundancy

The dominant finding is that **most strategies are redundant**. In 2024, 33/67
(49%) strategies have max forecast correlation >0.95 with at least one other
strategy. In 2025, this rises to 45/67 (67%). The strategy pool has extreme
internal correlation, confirming Phase 10d's finding that the ML regression
cluster dominates.

### Top Robust Strategies

| Strategy | Year | SA PnL | MA PnL | LOO delta-MAE | Note |
|----------|------|--------|--------|---------------|------|
| Wind Forecast Error | 2024 | 5608.50 | -92.24 | -0.0047 | Unique supply-side signal |
| Weekly Cycle | 2024 | 3026.68 | -64.13 | +0.0035 | Mean-reversion, helps MAE |
| Multi Spread | 2025 | 2728.34 | -64.87 | +0.0268 | Cross-border spread, strongest positive contribution |

### Top Destabilising Strategies

| Strategy | Year | LOO delta-MAE | MaxCorr | Note |
|----------|------|---------------|---------|------|
| PLSRegression | 2024 | -0.0858 | 0.9957 | Worst: removing it improves MAE by 0.09 EUR/MWh |
| Carbon Trend | 2024 | -0.0271 | 0.9006 | Momentum strategy, destabilises in volatile 2024 |
| Top KEnsemble | 2024 | -0.0224 | 1.0000 | Perfectly correlated with another; actively harmful |
| Gradient Boosting | 2024 | -0.0204 | 1.0000 | Perfectly correlated with another |
| Decision Tree | 2025 | -0.0155 | 0.9998 | Harmful in 2025 |

### Biggest Rank Shifts (standalone -> market-adjusted)

In 2024:
- Solar Forecast: SA rank 53 -> MA rank 7 (+46 places) -- gains from market repricing
- Carbon Trend: SA rank 51 -> MA rank 9 (+42 places)
- Composite Signal: SA rank 28 -> MA rank 65 (-37 places) -- loses badly

In 2025:
- Always Short: SA rank 62 -> MA rank 6 (+56 places) -- extreme gainer
- Neural Net: SA rank 8 -> MA rank 54 (-46 places) -- strong standalone, weak market

### Implications

1. **Pruning opportunity**: Removing the ~25 destabilising strategies from 2024
   would likely improve market MAE. Many are ML strategies with near-perfect
   forecast correlation to another strategy.

2. **Redundancy is structural**: The 11-strategy ML regression cluster produces
   nearly identical forecasts, so having all 11 adds noise rather than
   information.

3. **Cross-border and supply-side signals are underrepresented but valuable**:
   Multi Spread, Wind Forecast Error, and Solar Forecast gain ranks after
   market repricing, suggesting they carry unique information.

4. **Standalone PnL is a poor proxy for market contribution**: Many top-ranked
   standalone strategies (Neural Net, Gradient Boosting) become weak or harmful
   in the market context due to forecast crowding.

## Checklist

- [x] Build standalone vs market-adjusted PnL comparison table for all 67 strategies
- [x] Define and compute market-contribution metrics:
  - [x] MAE improvement when strategy is included vs excluded (leave-one-out)
  - [x] weight stability across iterations (low variance = more stable)
  - [x] correlation with the strategy's closest cluster peer (redundancy measure)
- [x] Classify each strategy into one of:
  - **robust**: strong standalone + strong market-adjusted + positive market contribution
  - **standalone-only**: strong standalone but weak market-adjusted
  - **redundant**: similar forecasts to another strategy already in the pool
  - **destabilising**: negative market contribution (MAE worsens when included)
- [x] Identify the top-5 most robust and top-5 most problematic strategies
- [x] Write `scripts/phase10f_strategy_robustness.py`
- [x] Save results to `data/results/phase10/strategy_robustness.csv`
- [x] Produce a shortlist of keep / revise / replace recommendations for Phase 10g

## Recommendations for Phase 10g

### Keep (robust, unique signal)
- Wind Forecast Error, Weekly Cycle, Multi Spread, Solar Forecast

### Revise (potentially valuable but needs de-correlation)
- ML regression cluster: keep 2-3 best representatives, remove the rest
- Ensemble strategies: rebuild to reward diversity, not average skill

### Replace or Prune
- PLSRegression (worst destabiliser in 2024)
- Top KEnsemble, Gradient Boosting, Decision Tree (perfectly correlated + harmful)
- Most momentum strategies destabilise in volatile periods

### New Strategy Gaps
- More cross-border/spread strategies (underrepresented, high market contribution)
- Regime-aware forecasters that adapt to volatility
- Balanced long/short strategies (Always Short gained 56 ranks in 2025)

## Test Coverage

18 tests in `tests/backtest/test_strategy_robustness.py`:
- `TestComputeStandalonePnl` (3 tests)
- `TestComputeWeightStability` (3 tests)
- `TestComputeForecastRedundancy` (3 tests)
- `TestClassifyStrategy` (6 tests)
- `TestRunMarketFast` (3 tests)

## Artifacts

- `scripts/phase10f_strategy_robustness.py` — analysis script (LOO in ~7s)
- `tests/backtest/test_strategy_robustness.py` — 18 tests
- `data/results/phase10/strategy_robustness.csv` — 134-row full results (67 strategies x 2 years)
