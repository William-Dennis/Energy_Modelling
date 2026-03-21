# Phase A: Feature Engineering Foundation

> [ROADMAP](../phases/ROADMAP.md) · [Expansion index](README.md)

## Status: ✅ Complete

## Objective

Add 18 derived features to `build_daily_backtest_frame()` in
`src/energy_modelling/backtest/data.py`. These features are computed
from the existing 29 raw features and serve as inputs to the new
strategies in Phases B–F.

A dedicated module `src/energy_modelling/backtest/feature_engineering.py`
contains pure functions for each derived feature. The data pipeline calls
these functions after building the raw daily frame.

---

## Design Decisions

### Where to compute derived features

**Decision**: Pipeline level — added to `build_daily_backtest_frame()`.

**Rationale**:
- Computed once, reused by all strategies.
- Consistent across all strategies (no drift from re-implementation).
- Strategies can use derived features directly from `state.features` without
  needing to access `state.history` for computations already done.
- Makes the glossary and feature inventory accurate and complete.

**Trade-off**: The data pipeline becomes slightly more complex. Mitigated by
isolating all derived feature logic in `feature_engineering.py`.

### Look-ahead safety

All derived features must be look-ahead-safe:
- Features derived from **same-day forecast** columns (e.g. `net_demand_mw`)
  are aligned to the delivery date — safe, same timing group.
- Features derived from **lagged realised** columns (e.g. `de_fr_spread`) are
  already lagged by 1 day — safe.
- Rolling statistics (e.g. `rolling_vol_7d`) are computed over the lagged
  `price_change_eur_mwh` series — safe.
- `price_zscore_20d` uses the 20-day rolling mean/std of `price_mean` (lagged)
  — safe.

---

## Derived Features Added

### Group 1: Supply/Demand Balance (same-day forecast)

| Column | Formula | Timing | Correlation (approx) |
|--------|---------|--------|----------------------|
| `net_demand_mw` | `load_forecast - wind_onshore_fcast - wind_offshore_fcast - solar_fcast` | same-day | ~+0.30 |
| `renewable_penetration_pct` | `(wind_onshore + wind_offshore + solar_fcast) / load_forecast` | same-day | ~−0.25 |

### Group 2: Price Spreads (lagged realised)

| Column | Formula | Timing | Correlation (approx) |
|--------|---------|--------|----------------------|
| `de_fr_spread` | `price_mean − price_fr_eur_mwh_mean` | lagged | −0.132 |
| `de_nl_spread` | `price_mean − price_nl_eur_mwh_mean` | lagged | ~−0.10 |
| `de_avg_neighbour_spread` | `price_mean − mean(FR, NL, AT, CZ, PL, DK1)` | lagged | ~−0.12 |

### Group 3: Price Mean-Reversion (lagged realised, rolling)

| Column | Formula | Timing | Correlation (approx) |
|--------|---------|--------|----------------------|
| `price_zscore_20d` | `(price_mean − MA20) / std20` | lagged | −0.173 |
| `price_range` | `price_max − price_min` | lagged | weak |

### Group 4: Commodity Trends (lagged realised, rolling)

| Column | Formula | Timing | Correlation (approx) |
|--------|---------|--------|----------------------|
| `gas_trend_3d` | `gas_price_usd_mean.diff(3)` | lagged | ~+0.07 |
| `carbon_trend_3d` | `carbon_price_usd_mean.diff(3)` | lagged | ~+0.05 |
| `fuel_cost_index` | `gas_price * 7.5 + carbon_price * 0.37` | lagged | weak |

### Group 5: Surprise / Error Signals (lagged realised)

| Column | Formula | Timing | Correlation (approx) |
|--------|---------|--------|----------------------|
| `wind_forecast_error` | `(wind_onshore_fcast + wind_offshore_fcast) − (gen_wind_onshore + gen_wind_offshore)_{t−1}` | mixed | TBD |
| `load_surprise` | `load_forecast − load_actual_mw_mean_{t−1}` | mixed | TBD |

### Group 6: Volatility Regime (lagged realised, rolling)

| Column | Formula | Timing | Correlation (approx) |
|--------|---------|--------|----------------------|
| `rolling_vol_7d` | `rolling 7d std(price_change_eur_mwh)` | lagged | regime indicator |
| `rolling_vol_14d` | `rolling 14d std(price_change_eur_mwh)` | lagged | regime indicator |

### Group 7: Aggregated Generation (lagged realised)

| Column | Formula | Timing | Correlation (approx) |
|--------|---------|--------|----------------------|
| `total_fossil_mw` | `gen_fossil_gas + gen_fossil_hard_coal + gen_fossil_brown_coal_lignite` | lagged | ~−0.20 |
| `net_flow_mw` | `flow_fr_net_import + flow_nl_net_import` | lagged | ~−0.15 |

### Group 8: Calendar Encodings (reference)

| Column | Formula | Notes |
|--------|---------|-------|
| `dow_int` | `delivery_date.weekday() + 1` (1=Mon, 7=Sun) | ML-usable integer |
| `is_weekend` | `dow_int >= 6` | Binary flag |

---

## Files Changed

| File | Change |
|------|--------|
| `src/energy_modelling/backtest/feature_engineering.py` | **New** — pure functions for each derived feature group |
| `src/energy_modelling/backtest/data.py` | Modified — calls `add_derived_features()` in `build_daily_backtest_frame()` |
| `tests/backtest/test_feature_engineering.py` | **New** — tests for all derived feature functions |

---

## Tests

34 tests covering:
- Correct computation of each group
- Look-ahead safety (rolling stats do not use future data)
- Edge cases: zero division in penetration pct, empty rolling windows
- Integration: derived features appear in the backtest frame

---

## Notes

- `price_zscore_20d` uses `min_periods=5` to handle the start of the series.
- `rolling_vol_7d` and `rolling_vol_14d` use `min_periods=3`.
- `wind_forecast_error` and `load_surprise` may be NaN for the earliest rows
  where there is no prior-day realised generation. Strategies using these
  features should handle NaN via `pd.Series.fillna(0.0)`.
- `fuel_cost_index` weights (7.5 heat rate, 0.37 emission factor) are derived
  from typical combined-cycle gas turbine characteristics.
