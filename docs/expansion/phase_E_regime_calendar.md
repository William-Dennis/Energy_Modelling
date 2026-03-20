# Phase E: Calendar, Temporal & Regime Strategies

## Status: ✅ Complete

## Objective

Implement ~8 strategies exploiting calendar patterns, rolling momentum,
and volatility regime detection.

---

## Strategy Inventory

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 1 | `MonthSeasonalStrategy` | `month_seasonal.py` | Monthly seasonal mean | ✅ |
| 2 | `MondayEffectStrategy` | `monday_effect.py` | Monday/Friday effect | ✅ |
| 3 | `QuarterSeasonalStrategy` | `quarter_seasonal.py` | Quarterly seasonal mean | ✅ |
| 4 | `ZScoreMomentumStrategy` | `zscore_momentum.py` | Z-score momentum follow | ✅ |
| 5 | `NetDemandMomentumStrategy` | `net_demand_momentum.py` | Net demand momentum | ✅ |
| 6 | `RenewableRegimeStrategy` | `renewable_regime.py` | Renewable penetration regime | ✅ |
| 7 | `VolatilityRegimeMLStrategy` | `volatility_regime_ml.py` | Volatility regime learned | ✅ |
| 8 | `GasCarbonJointTrendStrategy` | `gas_carbon_joint_trend.py` | Gas-carbon joint trend | ✅ |

---

## Strategy Details

### MonthSeasonalStrategy
Compute mean `price_change_eur_mwh` per calendar month from training data.
At forecast time, add the month's historical mean change to last settlement.
Hypothesis: seasonal demand patterns create systematic monthly biases.
Note: weaker signal than DOW, but orthogonal.

### MondayEffectStrategy
Exploits the strong Monday up-rate (90.4%) and Friday down-rate (38.7%).
Long on Mondays, short on Fridays, skip all other days.
High win rate on traded days, low coverage.

### QuarterSeasonalStrategy
Similar to MonthSeasonalStrategy but uses quarterly aggregation.
Fewer parameters, more stable estimates per quarter.

### ZScoreMomentumStrategy
Uses `price_zscore_20d` to detect momentum continuation.
Moderate z-score (0.5-2.0) in either direction -> follow the trend.

### NetDemandMomentumStrategy
Tracks the 3-day change in `net_demand_mw`. Rising net demand -> long.
Combines supply-demand fundamentals with momentum signal.

### RenewableRegimeStrategy
Uses `renewable_penetration_pct` to define high/low renewable regimes.
High renewables regime -> short (price-suppressing); low -> long.

### VolatilityRegimeMLStrategy
Uses `rolling_vol_14d` (from Phase A) as input to a trained classifier.
Learns the best action per volatility regime from historical data.

### GasCarbonJointTrendStrategy
Combines `gas_trend_3d` and `carbon_trend_3d` into a joint signal.
Both rising -> strong long; both falling -> strong short; mixed -> skip.

---

## Completion Criteria

- [x] 8 strategy files created
- [x] 8 test files created (5+ tests each)
- [x] All registered in `strategies/__init__.py`
- [x] All tests pass
