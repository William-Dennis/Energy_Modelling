# Phase C: Derived-Feature Threshold Strategies

## Status: ⏳ Pending

## Objective

Implement ~15 simple threshold strategies, each exploiting one or more of the
new derived features added in Phase A. These are intentionally simple — their
value is in diversity and providing strong signal to ensemble strategies.

---

## Strategy Inventory

| # | Class | File | Feature(s) | Logic | EDA Corr | Status |
|---|-------|------|-----------|-------|----------|--------|
| 1 | `NetDemandStrategy` | `net_demand.py` | `net_demand_mw` | Above median → long | ~+0.30 | ⏳ |
| 2 | `NetDemandWithSolarStrategy` | `net_demand_solar.py` | `net_demand_mw` (incl solar) | Same as above, full formula | ~+0.30 | ⏳ |
| 3 | `PriceZScoreReversionStrategy` | `price_zscore_reversion.py` | `price_zscore_20d` | z>1 → short; z<-1 → long; else skip | −0.173 | ⏳ |
| 4 | `GasTrendStrategy` | `gas_trend.py` | `gas_trend_3d` | Rising gas → long | ~+0.07 | ⏳ |
| 5 | `CarbonTrendStrategy` | `carbon_trend.py` | `carbon_trend_3d` | Rising carbon → long | ~+0.05 | ⏳ |
| 6 | `FuelIndexTrendStrategy` | `fuel_index_trend.py` | `gas_trend_3d`, `carbon_trend_3d` | Combined fuel cost momentum | ~+0.08 | ⏳ |
| 7 | `DEFRSpreadStrategy` | `de_fr_spread.py` | `de_fr_spread` | DE cheap vs FR → long | −0.132 | ⏳ |
| 8 | `DENLSpreadStrategy` | `de_nl_spread.py` | `de_nl_spread` | DE cheap vs NL → long | ~−0.10 | ⏳ |
| 9 | `MultiSpreadStrategy` | `multi_spread.py` | `de_avg_neighbour_spread` | DE cheap vs avg neighbours → long | ~−0.12 | ⏳ |
| 10 | `NLFlowSignalStrategy` | `nl_flow_signal.py` | `flow_nl_net_import_mw_mean` | Heavy export to NL → long | −0.192 | ⏳ |
| 11 | `FRFlowSignalStrategy` | `fr_flow_signal.py` | `flow_fr_net_import_mw_mean` | Heavy export to FR → long | −0.099 | ⏳ |
| 12 | `PriceMinReversionStrategy` | `price_min_reversion.py` | `price_min` | Low yesterday minimum → long | −0.143 | ⏳ |
| 13 | `WindForecastErrorStrategy` | `wind_forecast_error.py` | `wind_forecast_error` | Positive error (forecast > actual) → short | TBD | ⏳ |
| 14 | `LoadSurpriseStrategy` | `load_surprise.py` | `load_surprise` | Positive surprise (forecast > actual) → long | TBD | ⏳ |
| 15 | `RenewablesPenetrationStrategy` | `renewables_penetration.py` | `renewable_penetration_pct` | High penetration → short | ~−0.25 | ⏳ |

---

## Signal Strength Notes

- **NetDemand** (~+0.30) is the strongest individual signal in the dataset.
  This strategy should rank near the top.
- **NLFlow** (−0.192) is the strongest flow signal — better than FR flow.
  Negative correlation means: heavy NL export → prices fell yesterday → 
  expect recovery (long).
- **PriceZScoreReversion** uses `price_zscore_20d` from Phase A. The skip
  zone (|z| < 1) gives high selectivity — trades only ~32% of days.
- **Spread strategies** (DE-FR, DE-NL, MultiSpread) have moderate correlation
  but are largely orthogonal to wind/load signals. Valuable for diversification.

---

## Testing Approach

5+ tests per strategy (graduated testing for Tier 2):
- `fit()` computes threshold correctly
- `act()` returns expected direction for above/below threshold
- Boundary condition at threshold
- `reset()` preserves fitted state

---

## Completion Criteria

- [ ] 15 strategy files created
- [ ] 15 test files created
- [ ] All registered in `strategies/__init__.py`
- [ ] All tests pass
