# Phase G: Feedback Loop Infrastructure

> [ROADMAP](../phases/ROADMAP.md) · [Expansion index](README.md)

## Status: ✅ Complete

## Objective

Automate the feedback loop described in `docs/phases/phase_6_feedback_loop.md`.
The manual Phase 6 process (run → analyze → revisit EDA → refine) is replaced
by automated infrastructure in `src/energy_modelling/backtest/feedback.py`.

---

## Components

### 1. Strategy Correlation Matrix

After running backtest for all strategies, collect daily `act()` predictions
(+1/-1/None) for each strategy. Compute the pairwise Pearson correlation
between prediction series.

Output: `data/results/strategy_correlations.csv`

Uses:
- Identify **redundant strategies** (corr > 0.8): candidates for pruning
- Identify **diverse strategies** (corr < 0.2): valuable for ensembles
- Identify **natural hedges** (corr < -0.3): anti-correlated pairs

### 2. Feature Contribution Analysis

For each raw and derived feature:
- Count how many strategies use it as a primary input
- Compute mean PnL on days where that feature's signal was correct
- Compute signal stability across years (does corr with direction hold?)

Output: `data/results/feature_contributions.csv`

### 3. Walk-Forward Validation

Instead of the static 2019-2023 train / 2024 validation split, run:

```
Year 2020: train=2019,       eval=2020
Year 2021: train=2019-2020,  eval=2021
Year 2022: train=2019-2021,  eval=2022
Year 2023: train=2019-2022,  eval=2023
Year 2024: train=2019-2023,  eval=2024
```

Output: `data/results/walk_forward_results.csv` with PnL/Sharpe per strategy
per year. Reveals:
- Strategies that only work in specific regimes (e.g. 2022 crisis)
- Strategies with consistent edge across all years
- Degradation/improvement trends

### 4. Strategy Performance Report

`StrategyReport` dataclass (persisted alongside `BacktestResult`):
- `name`: strategy class name
- `total_pnl`, `sharpe`, `win_rate`: from standard backtest
- `daily_predictions`: Series of +1/-1/0/None per day
- `regime_performance`: PnL breakdown by low/mid/high volatility regime
- `yearly_pnl`: PnL per calendar year
- `feature_usage`: list of features this strategy depends on

### 5. Automated Strategy Candidate Generation (optional)

A meta-algorithm that:
1. Reads the correlation matrix and feature contributions
2. For each underexploited feature (used by <3 strategies):
   - Generate threshold strategy variants (P25/P50/P75 threshold)
   - Generate trend strategy variants (3d/7d/14d diff)
3. For each pair of low-correlation strategies:
   - Generate an equal-weight ensemble candidate
4. Backtest all candidates
5. Register those that are (a) profitable AND (b) corr < 0.5 with all existing

This is not required for the initial implementation but provides a path
to automated strategy discovery.

---

## Files

| File | Contents |
|------|----------|
| `src/energy_modelling/backtest/feedback.py` | Core feedback loop functions |
| `src/energy_modelling/backtest/walk_forward.py` | Walk-forward validation runner |
| `tests/backtest/test_feedback.py` | Tests for feedback functions |

---

## Completion Criteria

- [x] `feedback.py` implements correlation matrix computation
- [x] `feedback.py` implements feature contribution analysis
- [x] `walk_forward.py` implements 5-window walk-forward validation
- [ ] Results persisted to `data/results/` (deferred — requires live data)
- [ ] Dashboard has a "Feedback" tab showing correlation heatmap and WF results (deferred — dashboard scope)
- [x] 19 tests covering all feedback functions (exceeds 10+ target)
