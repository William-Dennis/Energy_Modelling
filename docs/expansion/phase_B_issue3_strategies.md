# Phase B: Issue 3 Strategies

## Status: ⏳ In Progress

## Objective

Implement the 7 strategies specified in `issues/issue_3_new_strategies.md`.
These exploit features that are unused by the 11 baseline strategies.

---

## Strategy Inventory

| # | Class | File | Feature(s) | Signal | Status |
|---|-------|------|-----------|--------|--------|
| 1 | `SolarForecastStrategy` | `solar_forecast.py` | `forecast_solar_mw_mean` | Merit-order: high solar → short | ⏳ |
| 2 | `CommodityCostStrategy` | `commodity_cost.py` | `gas_price_usd_mean`, `carbon_price_usd_mean` | Fuel index above median → long | ⏳ |
| 3 | `TemperatureExtremeStrategy` | `temperature_extreme.py` | `weather_temperature_2m_degc_mean` | P10 cold or P90 hot → long; moderate → short | ⏳ |
| 4 | `CrossBorderSpreadStrategy` | `cross_border_spread.py` | `price_fr_eur_mwh_mean`, `price_nl_eur_mwh_mean` | DE cheaper than neighbours → long | ⏳ |
| 5 | `VolatilityRegimeStrategy` | `volatility_regime.py` | `price_std`, `price_change_eur_mwh` (history) | High vol + up → short; low vol → momentum | ⏳ |
| 6 | `NuclearAvailabilityStrategy` | `nuclear_availability.py` | `gen_nuclear_mw_mean` (history) | 1σ below rolling mean → long | ⏳ |
| 7 | `RenewablesSurplusStrategy` | `renewables_surplus.py` | `forecast_wind_offshore_mw_mean`, `forecast_wind_onshore_mw_mean`, `forecast_solar_mw_mean` | P80 surplus → short; P20 drought → long | ⏳ |

---

## Notes on Signal Quality (from EDA)

The EDA (`docs/eda/04_signal_extraction.md:361-369`) flagged several of
these features as low individual signal:

| Feature | EDA Finding | Implication |
|---------|-------------|-------------|
| `forecast_solar_mw_mean` | Near-zero correlation alone | SolarForecast likely weak; useful in ensembles |
| `weather_temperature_2m_degc_mean` | Collinear with load; ~0 direct corr | TemperatureExtreme only works in non-linear extremes |
| `gen_nuclear_mw_mean` | Structurally zero from 2023 | NuclearAvailability unreliable post-2023 |
| Gas/carbon price level | Near-zero daily correlation | CommodityCost needs trend form, not level |

The **CommodityCostStrategy** in Issue 3 uses fuel index level (not trend).
This may underperform. A `CommodityTrendStrategy` in Phase C addresses this.

---

## Testing Approach

Each strategy: 10+ tests.
- `fit()` computes correct threshold/parameters from synthetic data
- `act()` returns correct direction for known high/low/boundary inputs
- `reset()` clears ephemeral state but preserves fitted parameters
- Unfitted strategy raises `RuntimeError` (or returns safe default)
- Integration test with a real-data slice (if CSV is available)

---

## Completion Criteria

- [ ] 7 strategy files created under `strategies/`
- [ ] 7 test files created under `tests/backtest/`
- [ ] All 7 classes registered in `strategies/__init__.py`
- [ ] All existing 295+ tests still pass
- [ ] Each strategy has docstring with economic rationale
