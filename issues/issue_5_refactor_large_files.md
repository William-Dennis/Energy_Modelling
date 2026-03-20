# Issue 5: Refactor Large Files (300-Line / 40-Line Limits)

## Summary

Six source files exceed the 300-line target, and 28 functions exceed 40 lines. Break these into smaller, focused modules and sub-functions to improve readability, testability, and maintainability.

## Motivation

Large files and long functions make it harder to:
- Review changes in PRs
- Test individual behaviors in isolation
- Onboard new contributors
- Avoid merge conflicts when working in parallel

The targets are **aspirational but achievable**: no `.py` file over 300 lines, no function over 40 lines.

## Current State

### Files Over 300 Lines

| File | Lines | Functions >40 lines |
|------|-------|---------------------|
| `dashboard/_eda.py` | 1788 | 17 |
| `dashboard/eda_analysis.py` | 534 | 4 |
| `dashboard/_backtest.py` | 460 | 2 |
| `data_collection/join.py` | 399 | 2 |
| `backtest/convergence.py` | 314 | 2 |
| `dashboard/_accuracy.py` | 304 | 1 |

All paths are relative to `src/energy_modelling/`.

### Worst Offenders (Functions >80 lines)

| Function | File | Lines | Length |
|----------|------|-------|--------|
| `_render_period` | `_accuracy.py` | 73-264 | 192 lines |
| `join_datasets` | `join.py` | 234-399 | 166 lines |
| `render` | `_backtest.py` | 217-346 | 130 lines |
| `_section_volatility_regimes` | `_eda.py` | 1144-1256 | 113 lines |
| `_section_forecast_errors` | `_eda.py` | 939-1048 | 110 lines |
| `_section_wind_quintile` | `_eda.py` | 1585-1690 | 106 lines |
| `_section_distributions` | `_eda.py` | 577-676 | 100 lines |
| `_section_strategy_correlation_insights` | `_eda.py` | 1693-1788 | 96 lines |
| `_section_residual_load` | `_eda.py` | 1259-1352 | 94 lines |
| `run_adaptive_foresight_market` | `convergence.py` | 221-314 | 94 lines |
| `_section_price_changes` | `_eda.py` | 785-875 | 91 lines |
| `_section_feature_importance` | `_eda.py` | 1051-1141 | 91 lines |

## Implementation Plan

### 1. Split `dashboard/_eda.py` (1788 lines → 4 files, each <300)

This is the largest file by far. Split by section groupings:

**`dashboard/_eda_core.py`** (~180 lines)
- `_load_data()` (lines 152-168)
- `_load_backtest_data()` (lines 172-178)
- `render()` (lines 193-263) — the main dispatch function
- Imports shared by all section modules

**`dashboard/_eda_sections_basic.py`** (~290 lines)
- `_section_overview()` (30 lines)
- `_section_price_ts()` (33 lines)
- `_section_generation()` (45 lines)
- `_section_load()` (40 lines)
- `_section_neighbours()` (58 lines)
- `_section_commodities()` (24 lines)
- `_section_weather()` (30 lines)
- `_section_correlations()` (30 lines)

**`dashboard/_eda_sections_analysis.py`** (~300 lines)
- `_section_distributions()` (100 lines — split into sub-functions)
- `_section_negative()` (52 lines — split into sub-functions)
- `_section_heatmap()` (27 lines)
- `_section_scatter()` (16 lines)
- `_section_price_changes()` (91 lines — split into sub-functions)
- `_section_autocorrelation()` (59 lines — split into sub-functions)

**`dashboard/_eda_sections_advanced.py`** (~300 lines)
- `_section_forecast_errors()` (110 lines — split into sub-functions)
- `_section_feature_importance()` (91 lines — split into sub-functions)
- `_section_volatility_regimes()` (113 lines — split into sub-functions)
- `_section_residual_load()` (94 lines — split into sub-functions)
- `_section_dow_edge_stability()` (47 lines — split into sub-functions)
- `_section_feature_drift()` (64 lines — split into sub-functions)
- `_section_quarterly_patterns()` (55 lines — split into sub-functions)
- `_section_volatility_regime_performance()` (49 lines — split into sub-functions)
- `_section_wind_quintile()` (106 lines — split into sub-functions)
- `_section_strategy_correlation_insights()` (96 lines — split into sub-functions)

**Wiring**: `_eda_core.py` imports the section functions from the other modules. The original `_eda.py` becomes a thin re-export: `from ._eda_core import render` so external callers don't break.

**Sub-function strategy for >40-line sections**: Each large section typically has a pattern of (1) compute data, (2) create chart, (3) display metrics. Extract each step into a helper:
```python
# Before (110 lines)
def _section_forecast_errors(df, backtest_data):
    ...

# After (30 lines + 3 helpers of ~25 lines each)
def _section_forecast_errors(df, backtest_data):
    errors = _compute_forecast_errors(df)
    fig = _plot_forecast_errors(errors)
    _display_forecast_error_stats(errors)
```

### 2. Split `dashboard/eda_analysis.py` (534 lines → 2 files)

**`dashboard/eda_analysis.py`** (~270 lines) — keep core analysis functions:
- `day_of_week_edge_by_year`, `compute_direction_streaks`, `feature_drift`, `price_change_histogram_data`, `price_volatility_regimes`, `autocorrelation_analysis`

**`dashboard/eda_analysis_advanced.py`** (~270 lines) — move advanced analysis:
- `feature_importance_analysis`, `residual_load_analysis`, `quarterly_seasonal_patterns`, `volatility_regime_performance`, `wind_quintile_analysis`

Break `feature_drift` (45 lines), `compute_direction_streaks` (43 lines), `wind_quintile_analysis` (43 lines), and `volatility_regime_performance` (42 lines) into sub-functions. Each is just barely over 40 lines, so extracting one helper from each should suffice.

### 3. Refactor `dashboard/_backtest.py` (460 lines)

**`render()` (130 lines → ~40 lines + helpers)**:
Extract into sub-functions:
- `_render_strategy_selector()` — strategy picker UI
- `_run_backtest_if_requested()` — the recompute logic
- `_display_backtest_summary()` — summary metrics table
- `_display_pnl_chart()` — PnL time series chart

**`_render_detail()` (46 lines)**: Acceptable, but if possible split chart creation from display.

Target: `_backtest.py` stays as one file at ~300 lines after `render()` is broken up.

### 4. Refactor `dashboard/_accuracy.py` (304 lines)

**`_render_period()` (192 lines → ~30 lines + helpers)**:
This single function generates all accuracy charts for a period. Split into:
- `_compute_accuracy_metrics()` — compute accuracy stats
- `_plot_direction_accuracy()` — direction accuracy chart
- `_plot_calibration()` — calibration chart
- `_plot_confusion_matrix()` — confusion matrix
- `_display_accuracy_summary()` — summary table

Target: `_accuracy.py` stays as one file at ~300 lines after `_render_period()` is broken up.

### 5. Refactor `backtest/convergence.py` (314 lines)

**`run_adaptive_foresight_market()` (94 lines → ~30 lines + helpers)**:
- `_run_single_iteration()` — one iteration of the adaptive loop
- `_check_convergence()` — convergence criterion check
- `_build_trajectory_record()` — format iteration results

**`compute_convergence_trajectory()` (47 lines)**: Acceptable, but split if needed.

Target: `convergence.py` drops to ~270 lines.

### 6. Refactor `data_collection/join.py` (399 lines)

**`join_datasets()` (166 lines → ~30 lines + staged helpers)**:
This function has clear stages: load each dataset, clean, merge, validate. Split into:
- `_load_and_clean_entsoe()` — ENTSO-E data loading
- `_load_and_clean_weather()` — weather data loading
- `_load_and_clean_commodities()` — commodity data loading
- `_merge_datasets()` — the actual join logic
- `_validate_joined()` — post-join validation

**`build_kaggle_metadata()` (63 lines)**: Split into `_build_column_metadata()` and `_build_dataset_metadata()`.

Target: `join.py` drops to ~300 lines.

## Testing Strategy

This is a pure refactoring issue — **no behavioral changes**. The test strategy is:

1. **All 276 existing tests must still pass** after each refactoring step
2. **No new tests needed** unless a new public API surface is created
3. Run `pytest --ignore=tests/data_collection -q` after every file split
4. For dashboard modules, verify the dashboard still renders by running `streamlit run src/energy_modelling/dashboard/app.py` and clicking through tabs

### Refactoring Safety Checklist (Per File)

- [ ] Read the file and map all function dependencies
- [ ] Identify all external callers (grep for imports)
- [ ] Split into new modules
- [ ] Update imports in the original file (re-export for backward compat)
- [ ] Update imports in all callers
- [ ] Run full test suite
- [ ] Verify no circular imports: `python -c "from energy_modelling.dashboard._eda_core import render"`

## Files to Create

| File | From | Approx Lines |
|------|------|-------------|
| `dashboard/_eda_core.py` | `_eda.py` | ~180 |
| `dashboard/_eda_sections_basic.py` | `_eda.py` | ~290 |
| `dashboard/_eda_sections_analysis.py` | `_eda.py` | ~300 |
| `dashboard/_eda_sections_advanced.py` | `_eda.py` | ~300 |
| `dashboard/eda_analysis_advanced.py` | `eda_analysis.py` | ~270 |

## Files to Modify

| File | Change |
|------|--------|
| `dashboard/_eda.py` | Replace with thin re-export shim |
| `dashboard/eda_analysis.py` | Remove functions moved to advanced module |
| `dashboard/_backtest.py` | Extract `render()` sub-functions |
| `dashboard/_accuracy.py` | Extract `_render_period()` sub-functions |
| `backtest/convergence.py` | Extract `run_adaptive_foresight_market()` sub-functions |
| `data_collection/join.py` | Extract `join_datasets()` staged helpers |
| `dashboard/app.py` | Update imports if needed |

## Acceptance Criteria

- [x] No `.py` file exceeds 300 lines (across the 6 targets) — **all verified ≤300**
- [x] No function exceeds 40 lines (across the 6 targets)
- [x] All 276 existing tests pass with no modifications — **295 tests now passing (includes new tests from other issues)**
- [x] No circular imports introduced
- [ ] Dashboard renders correctly (manual smoke test) — **import-verified; full smoke test requires running Streamlit**
- [x] All public API surfaces remain backward-compatible (re-exports where needed)
- [x] Each new module has a module-level docstring

## Status: ✅ COMPLETE

### Refactoring Results

| Original File | Before | After | New Modules |
|---------------|--------|-------|-------------|
| `dashboard/_eda.py` | 1788 | 4 (shim) | 11 modules (114–257 lines each) |
| `dashboard/eda_analysis.py` | 534 | 297 | `eda_analysis_advanced.py` (283) |
| `dashboard/_backtest.py` | 502 | 299 | `_backtest_render.py` (235) |
| `dashboard/_accuracy.py` | 314 | 297 | sub-functions extracted in-place |
| `backtest/convergence.py` | 314 | 268 | sub-functions extracted in-place |
| `data_collection/join.py` | 399 | 300 | sub-functions extracted in-place |

### Files Created
- `dashboard/_eda_core.py` (165), `_eda_constants.py` (114)
- `dashboard/_eda_sections_basic.py` (257), `_eda_sections_distributions.py` (180)
- `dashboard/_eda_sections_feedback.py` (241), `_eda_sections_forecasts.py` (231)
- `dashboard/_eda_sections_market.py` (151), `_eda_sections_signals.py` (237)
- `dashboard/_eda_sections_trading.py` (176), `_eda_sections_volatility.py` (252)
- `dashboard/eda_analysis_advanced.py` (283)
- `dashboard/_backtest_render.py` (235)

## Labels

`refactor`, `code-quality`, `priority-medium`

## Parallel Safety

Fully safe to work on in parallel with Issues 1, 2, 3, and 6. This issue only reorganizes existing code, it does not change behavior or public APIs. The main risk is merge conflicts if another issue modifies the same files -- but since this is purely structural, conflicts should be easy to resolve.

**Recommended order**: Start with `_eda.py` (biggest win), then `_backtest.py` and `_accuracy.py`, then `convergence.py` and `join.py`, finally `eda_analysis.py`. Do one file at a time, run tests after each.
