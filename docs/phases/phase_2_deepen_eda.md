# Phase 2: Deepen EDA Analysis

## Status: COMPLETE

**Test results: 145 passed (121 existing + 24 new), 0 failed**

## Objective

Add richer EDA sections to `_eda.py` that go beyond surface-level descriptive stats.
Focus on analyses that reveal patterns useful for trading strategy development.
Each new section should naturally lead to a testable hypothesis (feeding Phase 3).

## Prerequisites

- Phase 0 complete (consolidated codebase)
- Phase 1 complete (audit findings and prioritized gap list)

## Checklist

### New EDA sections to add (priority order from Phase 1 audit)

- [x] P1: Price Change Distribution — histogram of daily settlement changes, skewness, kurtosis, base rates, direction by month/DOW
- [x] P2: Autocorrelation Analysis — ACF of price changes with 95% CI bands, direction streaks, transition probability matrix
- [x] P3: Forecast Error Analysis — load/solar/wind forecast errors, scatter, monthly bias, MAE/RMSE metrics
- [x] P4: Lagged Feature Importance — D-1 feature correlation with D direction, ranked bar chart, top features table
- [x] P5: Volatility & Regime Detection — rolling volatility, high-vol vs low-vol regime comparison, annual volatility
- [x] P6: Residual Load — residual load vs price scatter, change-vs-change, weekly time series overlay

### Implementation approach
- [x] Each section is a private function `_section_*()` called from `render()`
- [x] Follow existing pattern: charts + stat cards
- [x] All new sections added AFTER existing 12 (non-breaking), under "Trading Signal Analysis" divider
- [x] Unit tests for all pure computation functions

### Testing
- [x] Extract pure computation functions into `dashboard/eda_analysis.py` (no Streamlit dependency)
- [x] Write 24 tests across 11 test classes in `tests/dashboard/test_eda_analysis.py`
- [x] Verify dashboard module imports without errors
- [x] Full regression: 145 passed, 0 failed

## Architecture

```
dashboard/eda_analysis.py          # Pure computation (10 functions, fully tested)
  compute_daily_settlement()       # Hourly → daily mean
  compute_price_changes()          # Day-over-day settlement changes
  direction_base_rates()           # Up/down/zero counts, percentages, moments
  autocorrelation()                # ACF at specified lags
  compute_direction_streaks()      # Consecutive same-direction runs
  compute_forecast_errors()        # error, abs_error, pct_error
  lagged_direction_correlation()   # Feature-direction Pearson correlation
  rolling_volatility()             # Rolling std of price changes
  compute_residual_load()          # Load - renewable generation
  direction_by_group()             # Direction win rates by category

dashboard/_eda.py                  # Streamlit rendering (18 sections)
  Sections 1-12: Original (unchanged)
  Section 13: Price Change Distribution (P1)
  Section 14: Autocorrelation & Direction Persistence (P2)
  Section 15: Forecast Error Analysis (P3)
  Section 16: Feature Importance for Price Direction (P4)
  Section 17: Volatility & Regime Analysis (P5)
  Section 18: Residual Load Analysis (P6)
```

## Sections Added

| Section | Function | Tests | Hypothesis Generated |
|---------|----------|-------|---------------------|
| 13. Price Change Distribution | `_section_price_changes` | 8 | Base rate asymmetry → always-long/short viability; monthly/DOW patterns → calendar strategies |
| 14. Autocorrelation & Persistence | `_section_autocorrelation` | 5 | ACF sign at lag 1 → momentum vs mean-reversion; transition probs → conditional direction |
| 15. Forecast Error Analysis | `_section_forecast_errors` | 3 | Systematic forecast bias → contrarian signal; seasonal error patterns → conditional strategies |
| 16. Feature Importance | `_section_feature_importance` | 2 | Top correlated features → feature-based direction strategies |
| 17. Volatility & Regime | `_section_volatility_regimes` | 2 | Regime-dependent base rates → adaptive strategy switching |
| 18. Residual Load | `_section_residual_load` | 4 | Residual load change → price direction predictor |
