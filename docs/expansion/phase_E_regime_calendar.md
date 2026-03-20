# Phase E: Calendar, Temporal & Regime Strategies

## Status: ⏳ Pending

## Objective

Implement ~8 strategies exploiting calendar patterns, rolling momentum,
and volatility regime detection.

---

## Strategy Inventory

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 1 | `MonthOfYearStrategy` | `month_of_year.py` | Monthly mean change from training | ⏳ |
| 2 | `DayOfWeekFilteredWindStrategy` | `dow_filtered_wind.py` | DOW on Mon/Tue/Sat/Sun, wind signal Wed/Thu/Fri | ⏳ |
| 3 | `WeekendOnlyStrategy` | `weekend_only.py` | Short Sat/Sun only; skip all weekdays | ⏳ |
| 4 | `Lag1ReversionStrategy` | `lag1_reversion.py` | Opposite of lag-1 price change (weak signal, diversifier) | ⏳ |
| 5 | `Lag3CycleStrategy` | `lag3_cycle.py` | Lag-3 price change correlation | ⏳ |
| 6 | `RollingMomentum5dStrategy` | `rolling_momentum_5d.py` | 5-day rolling mean of price_change > 0 → long | ⏳ |
| 7 | `RollingMomentum10dStrategy` | `rolling_momentum_10d.py` | 10-day rolling momentum | ⏳ |
| 8 | `HighVolMeanReversionStrategy` | `high_vol_mean_reversion.py` | High vol + yesterday's change → mean-revert | ⏳ |

---

## Strategy Details

### MonthOfYearStrategy
Compute mean `price_change_eur_mwh` per calendar month from training data.
At forecast time, add the month's historical mean change to last settlement.
Hypothesis: seasonal demand patterns create systematic monthly biases.
Note: weaker signal than DOW, but orthogonal.

### DayOfWeekFilteredWindStrategy
Key insight from Phase 6 EDA (`docs/phases/phase_6_feedback_loop.md:75`):
wind offshore correlation with direction is **−0.303 on Wed/Thu**, stronger
than the all-day average of −0.218. This strategy:
- Mon: DOW (long, 90.4% up rate)
- Tue: DOW (long, 61.3% up rate)
- Wed/Thu: Wind forecast signal (−0.30 correlation)
- Fri: DOW (short, 38.7% up rate)
- Sat: DOW (short, 13.0% up rate)
- Sun: DOW (short, 26.8% up rate)

### WeekendOnlyStrategy
Only trade Saturday and Sunday (the two most reliable DOW days).
Short both. Skip all weekdays. High win rate, low trade count.
Maximises hit rate at the cost of coverage.

### HighVolMeanReversionStrategy
Uses `rolling_vol_14d` (from Phase A) to detect high-vol regime.
High vol = vol > P75 of training vol.
In high-vol: if yesterday's price went up → short; if down → long.
Logic: large price moves tend to partially reverse.
In low-vol: skip (momentum works, but this strategy doesn't capture it —
`LowVolMomentumStrategy` in Phase F handles that).

---

## Completion Criteria

- [ ] 8 strategy files created
- [ ] 8 test files created (5+ tests each)
- [ ] All registered in `strategies/__init__.py`
- [ ] All tests pass
