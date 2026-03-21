# Phase C: Derived-Feature Threshold Strategies

> [ROADMAP](../phases/ROADMAP.md) · [Expansion index](README.md)

## Status: ✅ Complete

## Objective

Implement ~15 simple threshold strategies, each exploiting one or more of the
new derived features added in Phase A. These are intentionally simple — their
value is in diversity and providing strong signal to ensemble strategies.

---

## Strategy Inventory

| # | Class | File | Feature(s) | Logic | EDA Corr | Status |
|---|-------|------|-----------|-------|----------|--------|
| 1 | `NetDemandStrategy` | `net_demand.py` | `net_demand_mw` | Above median → long | ~+0.30 | ✅ |
| 2 | `PriceZScoreReversionStrategy` | `price_zscore_reversion.py` | `price_zscore_20d` | z>1 → short; z<-1 → long; else skip | −0.173 | ✅ |
| 3 | `GasTrendStrategy` | `gas_trend.py` | `gas_trend_3d` | Rising gas → long | ~+0.07 | ✅ |
| 4 | `CarbonTrendStrategy` | `carbon_trend.py` | `carbon_trend_3d` | Rising carbon → long | ~+0.05 | ✅ |
| 5 | `FuelIndexTrendStrategy` | `fuel_index_trend.py` | `gas_trend_3d`, `carbon_trend_3d` | Combined fuel cost momentum | ~+0.08 | ✅ |
| 6 | `DEFRSpreadStrategy` | `de_fr_spread.py` | `de_fr_spread` | DE cheap vs FR → long | −0.132 | ✅ |
| 7 | `DENLSpreadStrategy` | `de_nl_spread.py` | `de_nl_spread` | DE cheap vs NL → long | ~−0.10 | ✅ |
| 8 | `MultiSpreadStrategy` | `multi_spread.py` | `de_avg_neighbour_spread` | DE cheap vs avg neighbours → long | ~−0.12 | ✅ |
| 9 | `NLFlowSignalStrategy` | `nl_flow_signal.py` | `flow_nl_net_import_mw_mean` | Heavy export to NL → long | −0.192 | ✅ |
| 10 | `FRFlowSignalStrategy` | `fr_flow_signal.py` | `flow_fr_net_import_mw_mean` | Heavy export to FR → long | −0.099 | ✅ |
| 11 | `PriceMinReversionStrategy` | `price_min_reversion.py` | `price_min` | Low yesterday minimum → long | −0.143 | ✅ |
| 12 | `WindForecastErrorStrategy` | `wind_forecast_error.py` | `wind_forecast_error` | Positive error (forecast > actual) → short | TBD | ✅ |
| 13 | `LoadSurpriseStrategy` | `load_surprise.py` | `load_surprise` | Positive surprise (forecast > actual) → long | TBD | ✅ |
| 14 | `RenewablesPenetrationStrategy` | `renewables_penetration.py` | `renewable_penetration_pct` | High penetration → short | ~−0.25 | ✅ |

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

- [x] 14 strategy files created
- [x] 14 test files created
- [x] All registered in `strategies/__init__.py`
- [x] All tests pass
