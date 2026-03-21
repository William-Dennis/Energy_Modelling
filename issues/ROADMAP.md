# Issue Roadmap

## Overview

Six issues covering benchmarks, dashboard improvements, new strategies, refactoring, and CLI tooling. Four can run in parallel immediately; two have dependencies.

## Current Status

| Issue | Title | Status | Tests Added |
|-------|-------|--------|-------------|
| 1 | Configurable Entry Price Benchmarks | ✅ Complete | 9 |
| 2 | Dashboard Loads Saved Results | ✅ Complete | 6 |
| 3 | New Strategies (5-7) | ✅ Complete (Expansion Phase B) | 70+ |
| 4 | Dashboard Benchmark Views | ✅ Complete | — |
| 5 | Refactor Large Files | ✅ Complete | — |
| 6 | Recompute CLI + Docs | ✅ Complete | 5 |

**Total tests**: 948 passing (excluding system-Python `data_collection/` — needs `uv run`)

## Dependency Graph

```
Issue 1 (Benchmarks)  ✅  ──┐
                             ├──> Issue 4 (Dashboard benchmark views)  ✅
Issue 2 (Saved results) ✅ ──┤
                             └──> Issue 6 (CLI + docs)  ✅

Issue 3 (New strategies)   ───── standalone  ✅ (completed via Expansion Phase B)
Issue 5 (Refactor)  ✅     ───── standalone
```

## Parallel Execution Plan

### Wave 1 (Start Immediately — All Independent)

| Issue | Title | Priority | Est. Effort | Status |
|-------|-------|----------|-------------|--------|
| 1 | [Configurable Entry Price Benchmarks](issue_1_configurable_entry_price_benchmarks.md) | High | Medium | ✅ Done |
| 2 | [Dashboard Loads Saved Results](issue_2_dashboard_saved_results.md) | High | Medium | ✅ Done |
| 3 | [New Strategies (5-7)](issue_3_new_strategies.md) | High | Large | ✅ Done (Expansion Phase B) |
| 5 | [Refactor Large Files](issue_5_refactor_large_files.md) | Medium | Medium | ✅ Done |

### Wave 2 (After Issues 1 + 2 Merge)

| Issue | Title | Priority | Est. Effort | Status |
|-------|-------|----------|-------------|--------|
| 4 | [Dashboard Benchmark Views](issue_4_dashboard_benchmark_views.md) | Medium | Small | ✅ Done |
| 6 | [Recompute CLI + Docs](issue_6_recompute_cli_and_docs.md) | Medium | Small | ✅ Done |

## Issue Summary

### Issue 1: Configurable Entry Price Benchmarks ✅
Add `entry_prices` parameter to `run_backtest()` and factory functions for alternative entry prices (noisy, biased, oracle). Creates `backtest/benchmarks.py`.

### Issue 2: Dashboard Loads Saved Results ✅
Dashboard loads pre-computed results from disk instead of recomputing live. Adds `io.py` with pickle-based save/load, auto-loads from `data/results/` on startup.

### Issue 3: New Strategies (5-7) ✅
Implement 5-7 new strategies exploiting the 20 unused features: solar, commodities, temperature, cross-border spreads, volatility regimes, nuclear, renewables surplus. **Completed during Expansion Phase B** -- all 7 strategies implemented and registered. See `docs/expansion/strategy_registry.md`.

### Issue 4: Dashboard Benchmark Views ✅
Added "Benchmark Comparison" tab to the dashboard with strategy × benchmark comparison table, heatmap visualization, and metric selector (PnL, Sharpe, Max Drawdown, Win Rate).

### Issue 5: Refactor Large Files ✅
All 6 target files refactored to ≤300 lines. `_eda.py` (1788 lines) split into 11 modules. All functions ≤40 lines. No behavioral changes, backward-compatible.

### Issue 6: Recompute CLI + Docs ✅
`recompute-all` CLI command registered in `pyproject.toml`. README updated with "Regenerating Results" section. `docs/hackathon_challenge.md` terminology fixed.

## Current State (Post-Issues)

- **948 tests passing** (all pass via `uv run pytest`)
- **Full rename complete**: `challenge` → `backtest`, `market_simulation` → `futures_market`
- **67 strategies** implemented (2 baseline + 7 Phase 4 + 58 expansion)
- **Dashboard** has 5 tabs: EDA, Backtest, Futures Market, Accuracy, Benchmark Comparison
- **All 9 phases** (0-9) complete; Phase 10 planned
- **8 benchmark configurations**: baseline, noise (1/5/10/20 EUR), bias (±5 EUR), oracle
- **CLI**: `collect-data`, `build-backtest-data`, `recompute-all`
- **Result persistence**: `data/results/` with pickle-based save/load

## Constraints

- **TDD**: Write tests first, then implement
- **DRY/YAGNI**: No duplicated logic; only build what's needed
- **File limits**: No `.py` over 300 lines, no function over 40 lines (enforced in Issue 5)
- **BacktestStrategy interface**: `fit()` → `reset()` → `act()`. `reset()` must NOT clear fitted params
- **Test command**: `pytest --ignore=tests/data_collection -q`
- **11 data_collection tests** fail due to missing `pytest-mock` — always exclude
