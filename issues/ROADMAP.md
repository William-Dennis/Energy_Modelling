# Issue Roadmap

## Overview

Six issues covering benchmarks, dashboard improvements, new strategies, refactoring, and CLI tooling. Four can run in parallel immediately; two have dependencies.

## Dependency Graph

```
Issue 1 (Benchmarks)       ──┐
                             ├──> Issue 4 (Dashboard benchmark views)
Issue 2 (Saved results)    ──┤
                             └──> Issue 6 (CLI + docs)

Issue 3 (New strategies)   ───── standalone
Issue 5 (Refactor)         ───── standalone
```

## Parallel Execution Plan

### Wave 1 (Start Immediately — All Independent)

| Issue | Title | Priority | Est. Effort |
|-------|-------|----------|-------------|
| 1 | [Configurable Entry Price Benchmarks](issue_1_configurable_entry_price_benchmarks.md) | High | Medium |
| 2 | [Dashboard Loads Saved Results](issue_2_dashboard_saved_results.md) | High | Medium |
| 3 | [New Strategies (5-7)](issue_3_new_strategies.md) | High | Large |
| 5 | [Refactor Large Files](issue_5_refactor_large_files.md) | Medium | Medium |

These four issues have **zero file overlap** and can be assigned to four parallel agents.

### Wave 2 (After Issues 1 + 2 Merge)

| Issue | Title | Priority | Est. Effort |
|-------|-------|----------|-------------|
| 4 | [Dashboard Benchmark Views](issue_4_dashboard_benchmark_views.md) | Medium | Small |
| 6 | [Recompute CLI + Docs](issue_6_recompute_cli_and_docs.md) | Medium | Small |

Both depend on the benchmark factories (Issue 1) and result storage (Issue 2). Issues 4 and 6 can run in parallel with each other.

## Issue Summary

### Issue 1: Configurable Entry Price Benchmarks
Add `entry_prices` parameter to `run_backtest()` and factory functions for alternative entry prices (noisy, biased, oracle). Creates `backtest/benchmarks.py`.

### Issue 2: Dashboard Loads Saved Results
Dashboard loads pre-computed results from disk instead of recomputing live. Adds `ResultStore`, result CSVs in `data/results/`, and a "Recompute" button.

### Issue 3: New Strategies (5-7)
Implement 5-7 new strategies exploiting the 20 unused features: solar, commodities, temperature, cross-border spreads, volatility regimes, nuclear, renewables surplus.

### Issue 4: Dashboard Benchmark Views
Add benchmark comparison tabs/charts to the dashboard. Strategy performance across different entry price scenarios, side-by-side comparison, robustness heatmaps.

### Issue 5: Refactor Large Files
Break 6 files that exceed 300 lines into smaller modules. Split 28 functions that exceed 40 lines into sub-functions. Pure refactoring, no behavioral changes.

### Issue 6: Recompute CLI + Docs
Single `recompute-all` CLI command to regenerate all results. Update README and hackathon docs with current terminology and regeneration instructions.

## Current State (Pre-Issues)

- **276 tests passing** (excluding `data_collection/` — needs `pytest-mock`)
- **Full rename complete**: `challenge` → `backtest`, `market_simulation` → `futures_market`
- **9 strategies** implemented (2 baseline + 7 signal-based)
- **Dashboard** has 4 tabs: EDA, Backtest, Futures Market, Accuracy
- **All 7 original phases** (0-7) complete

## Constraints

- **TDD**: Write tests first, then implement
- **DRY/YAGNI**: No duplicated logic; only build what's needed
- **File limits**: No `.py` over 300 lines, no function over 40 lines (enforced in Issue 5)
- **BacktestStrategy interface**: `fit()` → `reset()` → `act()`. `reset()` must NOT clear fitted params
- **Test command**: `pytest --ignore=tests/data_collection -q`
- **11 data_collection tests** fail due to missing `pytest-mock` — always exclude
