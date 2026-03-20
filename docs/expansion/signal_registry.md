# Signal Registry

## Overview

All signals identified across EDA, phases, and strategy implementations.
Updated whenever a new signal is validated or a strategy is implemented.

---

## Raw Feature Signals

| Feature | Timing | Corr with Direction | Used By (baseline) | Phase Adding Strategy |
|---------|--------|---------------------|-------------------|----------------------|
| `load_forecast_mw_mean` | same-day | +0.235 | CompositeSignal, LoadForecast | — |
| `forecast_wind_offshore_mw_mean` | same-day | −0.212 | WindForecast, CompositeSignal | — |
| `gen_wind_onshore_mw_mean` | lagged | +0.210 | none (confounded by DOW) | C (NLFlowSignal uses correlated flow) |
| `weather_wind_speed_10m_kmh_mean` | lagged | +0.207 | none | — |
| `gen_fossil_gas_mw_mean` | lagged | −0.195 | FossilDispatch, CompositeSignal | — |
| `flow_nl_net_import_mw_mean` | lagged | −0.192 | none | C (NLFlowSignal) |
| `gen_fossil_brown_coal_lignite_mw_mean` | lagged | −0.187 | FossilDispatch, CompositeSignal | — |
| `forecast_wind_onshore_mw_mean` | same-day | −0.186 | WindForecast, CompositeSignal | — |
| `gen_wind_offshore_mw_mean` | lagged | +0.153 | none | — |
| `gen_fossil_hard_coal_mw_mean` | lagged | −0.148 | FossilDispatch, CompositeSignal | — |
| `price_min` | lagged | −0.143 | none | C (PriceMinReversion) |
| `load_actual_mw_mean` | lagged | −0.133 | none | — |
| `flow_fr_net_import_mw_mean` | lagged | −0.099 | none | C (FRFlowSignal) |
| `price_mean` | lagged | −0.092 | none | C (via spread) |
| `price_fr_eur_mwh_mean` | lagged | −0.044 | none | B (CrossBorderSpread), C (DEFRSpread) |
| `price_nl_eur_mwh_mean` | lagged | −0.062 | none | C (DENLSpread) |
| `price_at_eur_mwh_mean` | lagged | ~−0.06 | none | C (MultiSpread) |
| `price_cz_eur_mwh_mean` | lagged | ~−0.07 | none | C (MultiSpread) |
| `price_pl_eur_mwh_mean` | lagged | ~−0.10 | none | C (MultiSpread) |
| `price_dk_1_eur_mwh_mean` | lagged | ~−0.09 | none | C (MultiSpread) |
| `forecast_solar_mw_mean` | same-day | ~−0.004 | none | B (SolarForecast) |
| `gas_price_usd_mean` | lagged | +0.013 | none | B (CommodityCost) |
| `carbon_price_usd_mean` | lagged | +0.006 | none | B (CommodityCost) |
| `weather_temperature_2m_degc_mean` | lagged | ~0.004 | none | B (TemperatureExtreme) |
| `gen_nuclear_mw_mean` | lagged | ~−0.030 | none | B (NuclearAvailability) |
| `price_std` | lagged | +0.058 | none | B (VolatilityRegime) |
| `price_max` | lagged | −0.053 | none | — |
| `gen_solar_mw_mean` | lagged | ~0.0 | none | — |
| `delivery_date` (weekday) | reference | DOW effect | DayOfWeek, DowComposite | — |

---

## Derived Feature Signals (Phase A)

| Feature | Formula | Corr (approx) | Phase |
|---------|---------|--------------|-------|
| `net_demand_mw` | load_fcast − wind_on − wind_off − solar | ~+0.30 | A→C |
| `renewable_penetration_pct` | (wind_on + wind_off + solar) / load_fcast | ~−0.25 | A→C |
| `de_fr_spread` | price_mean − price_fr | −0.132 | A→C |
| `de_nl_spread` | price_mean − price_nl | ~−0.10 | A→C |
| `de_avg_neighbour_spread` | price_mean − avg(neighbours) | ~−0.12 | A→C |
| `price_zscore_20d` | (price_mean − MA20) / std20 | −0.173 | A→C |
| `price_range` | price_max − price_min | ~+0.06 | A→C |
| `gas_trend_3d` | gas_price.diff(3) | ~+0.07 | A→C |
| `carbon_trend_3d` | carbon_price.diff(3) | ~+0.05 | A→C |
| `fuel_cost_index` | gas*7.5 + carbon*0.37 | weak | A→B |
| `wind_forecast_error` | wind_fcast − wind_gen_t−1 | TBD | A→C |
| `load_surprise` | load_fcast − load_actual_t−1 | TBD | A→C |
| `rolling_vol_7d` | 7d std(price_change) | regime | A→B,E |
| `rolling_vol_14d` | 14d std(price_change) | regime | A→B,E |
| `total_fossil_mw` | gas + coal + lignite | ~−0.20 | A→C |
| `net_flow_mw` | flow_fr + flow_nl | ~−0.15 | A→C |
| `dow_int` | weekday as int (1-7) | DOW | A→D |
| `is_weekend` | dow ≥ 6 | DOW | A→D |

---

## Calendar Signals

| Signal | Corr / Up-Rate | Source |
|--------|---------------|--------|
| Monday delivery | 90.4% up | DayOfWeekStrategy |
| Saturday delivery | 13.0% up | DayOfWeekStrategy |
| Wind offshore on Wed/Thu | −0.30 (above avg) | Phase 6 EDA |
| Quarterly bias | None stable | Phase 6 EDA (negative finding) |
| Monthly seasonality | Weak, inconsistent | Phase E (MonthOfYear) |

---

## Anti-Signals (known dead ends)

| Feature | Reason |
|---------|--------|
| Raw gas price level | Near-zero daily corr; dominated by 2022 regime |
| Raw carbon price level | Near-zero daily corr |
| Solar forecast alone | Demand offsets supply; ~0 net signal |
| Lag-1 price change | Corr ~+0.02; essentially zero |
| Nuclear generation | Structurally zero from April 2023 onwards |
| Raw temperature | Collinear with load; no unique signal |
| price_max | Weaker than price_min; covered by price_std |
