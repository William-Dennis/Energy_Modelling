# Synthetic Futures Market - Implementation Status

## Status: COMPLETE

## WARNING — Historical Document (2026-03-21)

> **This document describes the pre-Phase 7 engine implementation and is
> substantially out of date.** It should be read as a build log for the
> initial market engine, not as a description of the current system.
>
> Key changes since this document was written:
>
> - **Engine rewrite**: The engine was rewritten for Phase 7 (spec-compliant,
>   undampened) and then modified again with EMA dampening (`ema_alpha=0.1`).
>   The `forecast_spread` parameter no longer exists.
> - **Convergence**: The engine now converges on both 2024 and 2025 data
>   (the "did not converge" note in the integration results is obsolete).
> - **Strategy count**: 67 strategies (not 12). The expansion work (Phases
>   A-G) added 55 strategies after this document was written.
> - **Test count**: 948 tests pass (not 185 or 150 as stated in various
>   sections below).
> - **Dampening**: `ema_alpha=0.1` (not "dampening alpha: 0.5" as stated
>   in the Decisions section). The old `alpha` parameter was removed.
> - **Dashboard**: Expanded to ~20 modules (not 4 tabs).
>
> For the current engine behaviour, see:
> - `docs/phases/phase_9_ema_price_update.md` (EMA experiments)
> - `docs/phases/ROADMAP.md` (overall project state)
> - `src/energy_modelling/backtest/futures_market_engine.py` (source of truth)

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `backtest/futures_market_engine.py` | DONE | 23/23 pass | Core engine: types, iteration, convergence |
| `backtest/futures_market_runner.py` | DONE | 11/11 pass | Orchestrator: collects strategies, runs market |
| `backtest/scoring.py` ext | DONE | 9/9 pass | Market-adjusted metrics |
| `backtest/__init__.py` | DONE | - | Export new public API |
| `dashboard/_futures_market.py` | DONE | - | Market view tabs + sidebar toggle |
| `tests/backtest/test_market.py` | DONE | 23/23 pass | Unit tests for market engine |
| `tests/backtest/test_market_runner.py` | DONE | 11/11 pass | Integration tests |
| `tests/backtest/test_scoring.py` | DONE | 9/9 pass | Scoring extension tests |

**Full test suite: 185/185 pass** (excluding pre-existing data_collection failures unrelated to market work)

## Build Log

### Loop 1: market.py core engine
- [x] Write types (FuturesMarketIteration, FuturesMarketEquilibrium)
- [x] Write compute_strategy_profits()
- [x] Write compute_weights()
- [x] Write compute_market_prices()
- [x] Write run_futures_market_iteration()
- [x] Write run_futures_market()
- [x] Write test_market.py - 23 tests
- [x] Run tests: 23/23 pass (0.54s)
- [x] Fix 3 keyword-arg mismatches (spread -> forecast_spread)

### Loop 2: market_runner.py orchestrator
- [x] Write FuturesMarketResult type
- [x] Write _recompute_pnl_against_market()
- [x] Write run_futures_market_evaluation()
- [x] Write test_market_runner.py - 11 tests
- [x] Run tests: 11/11 pass (0.62s)
- [x] Fixed 1 test assertion (market overshoot is correct economics)

### Loop 3: scoring.py extension + __init__.py
- [x] Add compute_market_adjusted_metrics() with alpha_pnl and original_total_pnl
- [x] Add market_leaderboard_score()
- [x] Write test_scoring.py - 9 tests
- [x] Run tests: 9/9 pass (0.51s)
- [x] Update __init__.py with new exports
- [x] Full regression: 332/332 pass (17.40s)

### Loop 4: dashboard integration
- [x] Add helper functions (_run_market_for_period, _market_leaderboard_frame, _format_market_leaderboard)
- [x] Add _render_market_section() with convergence info, rank changes, weight evolution, price/PnL charts
- [x] Wire market into main(): sidebar toggle, market evaluation after backtest, Market tabs
- [x] Full integration test: 12 strategies on daily_public.csv 2024 period
- [x] Final regression: 185/185 pass (3.66s)

## Integration Test Results (2024 period, 12 strategies)

The market model ran successfully on all 12 submission strategies:
- **Convergence**: Did not converge in 20 iterations (oscillation at delta=46.96 EUR/MWh)
- **Oscillation cause**: Strongly directional strategy pool creates a 2-cycle where short-biased and long-biased dominance alternates. This is economically correct behavior.
- **Rank changes**: Significant — TinyMLStrategy drops from #1 to #10 (forecast-driven alpha absorbed by market), StudentStrategy rises from #9 to #1 (alpha against market consensus)
- **Key insight**: Forecast-driven strategies that dominate the original leaderboard lose alpha when the market aggregates their signals, rewarding strategies with genuine contrarian edge

### Market-adjusted rankings vs original (2024)

| Market Rank | Strategy | Original PnL | Market PnL | Orig Rank | Change |
|:-----------:|----------|-------------:|-----------:|:---------:|:------:|
| 1 | StudentStrategy | 1,241 | 13,859 | 9 | +8 |
| 2 | YesterdayMeanReversion | 12,818 | 13,202 | 6 | +4 |
| 3 | SkipAll | 0 | 0 | 10 | +7 |
| 4 | GasTrend | 3,057 | -8,376 | 8 | +4 |
| 5 | YesterdayMomentum | -12,818 | -13,202 | 12 | +7 |
| 6 | AlwaysShort | -1,241 | -13,859 | 11 | +5 |
| 7 | WindForecastContrarian | 56,542 | -20,951 | 3 | -4 |
| 8 | PriceLevelMeanReversion | 79,577 | -26,099 | 2 | -6 |
| 9 | LoadForecastMedian | 31,467 | -29,931 | 4 | -5 |
| 10 | TinyML | 170,908 | -39,869 | 1 | -9 |
| 11 | SolarForecastContrarian | 8,801 | -40,882 | 7 | -4 |
| 12 | DEFranceSpread | 17,696 | -55,314 | 5 | -7 |

## Decisions
- Dampening alpha: 0.5
- Forecast spread: auto-calibrated from training std (~99.6 EUR/MWh for 2024)
- Market view: alongside original leaderboard (not replacing)
- Students see real last_settlement_price (market price is scoring-only)
- Non-convergence handled gracefully: dashboard shows convergence status, iteration count, and delta

---

## DRY Cleanup (Post-Market)

### Completed

| Task | Details |
|------|---------|
| Extract `_year_range()` | Moved from 7 `entsoe_*.py` files to `data_collection/utils.py` |
| Extract `_normalise_name()` | Moved from `entsoe_generation.py` + `entsoe_forecasts.py` to `data_collection/utils.py` |
| Extract `_class_display_name()` | Moved from `backtest.py` + `challenge_submissions.py` to `dashboard/__init__.py` |
| DRY excluded columns | `submission/common.py` now imports `_STATE_EXCLUDE_COLUMNS` from `backtest/runner.py` |
| Consolidate Kaggle docs | `docs/kaggle_description.md` + `docs/kaggle_upload.md` merged into `docs/kaggle.md` |
| Delete `docs/code_review.md` | One-time review artifact removed |
| Fix `annualized_return_pct` | Renamed to `annualized_pnl_eur` in `backtest/scoring.py` (was EUR, not a percentage) |
| Replace stale `README.md` | Replaced template placeholder with real project description |

## Dashboard Consolidation

### Status: COMPLETE

Consolidated three separate Streamlit dashboards (`app.py`, `backtest.py`,
`challenge_submissions.py`) into one modular dashboard with 4 tabs.

| File | Purpose | Status |
|------|---------|--------|
| `dashboard/app.py` | Thin orchestrator (page config + 4 tabs) | CURRENT |
| `dashboard/__init__.py` | Shared helpers: `class_display_name`, `monthly_pnl_heatmap`, `render_metric_cards` | EXTENDED |
| `dashboard/_eda.py` | Tab 1: EDA (12 sections) | CURRENT |
| `dashboard/_backtest.py` | Tab 2: Backtest leaderboard, yesterday-settlement pricing | CURRENT |
| `dashboard/_futures_market.py` | Tab 3: Synthetic futures market model | CURRENT |
| `dashboard/_accuracy.py` | Tab 4: Futures Market Simulation accuracy (converged vs real settlement) | CURRENT |
| `dashboard/_backtest.py` | DELETED — single-strategy backtest tab removed during framework consolidation | REMOVED |
| `dashboard/backtest.py` | DELETED — replaced by `_backtest.py`, then both removed | REMOVED |
| `dashboard/challenge_submissions.py` | DELETED — replaced by `_backtest.py` + `_futures_market.py` + `_accuracy.py` | REMOVED |

### Architecture

- Each tab is a module with a `render()` entry point
- `app.py` is a thin orchestrator: page config, title, `st.tabs()`, calls each `render()`
- Data flows via `st.session_state`: Backtest stores results, Futures Market reads them, Accuracy reads market results
- Shared chart/metric helpers in `dashboard/__init__.py` avoid duplication
- Single strategy framework: all strategies use `BacktestStrategy` ABC (the `strategy.*` framework was deleted)

### Verification

- All 150 tests pass (excluding pre-existing `data_collection` failures due to missing `pytest-mock`)
- All 4 tab modules import without errors
- Test import updated: `test_submission_strategies.py` now imports from `_backtest.py`

---

## Framework Consolidation (Phase 0)

### Status: COMPLETE

Deleted the duplicate `strategy.*` framework, keeping only `backtest.*` as the single strategy framework.

| Deleted | Reason |
|---------|--------|
| `src/energy_modelling/strategy/` (7 files) | Duplicate of `backtest.*` — identical PnL formula, same feature engineering |
| `src/energy_modelling/market_simulation/market.py` | `MarketEnvironment` was only used by the deleted backtest tab |
| `tests/strategy/` (4 test files) | Tests for deleted framework |
| `tests/futures_market/test_market.py` | Tests for deleted `MarketEnvironment` |
| `dashboard/_backtest.py` | Only consumer of the `strategy.*` framework |
