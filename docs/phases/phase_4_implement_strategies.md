# Phase 4: Implement Strategies

## Status: COMPLETE

## Objective

Implement new `ChallengeStrategy` classes in the `strategies/` directory based on
the hypotheses identified in Phase 3. Follow TDD: write tests first, then implement.

## Prerequisites

- Phase 0 complete (consolidated framework, `strategies/` working)
- Phase 3 complete (hypotheses ranked and specified)

## Checklist

### 4a. Implement hypothesis-driven strategies (H1-H7)
- [x] H1: DayOfWeekStrategy — `strategies/day_of_week.py` + `tests/challenge/test_strategy_day_of_week.py` (12 tests)
- [x] H2: WindForecastStrategy — `strategies/wind_forecast.py` + `tests/challenge/test_strategy_wind_forecast.py` (10 tests)
- [x] H3: LoadForecastStrategy — `strategies/load_forecast.py` + `tests/challenge/test_strategy_load_forecast.py` (10 tests)
- [x] H4: Lag2ReversionStrategy — `strategies/lag2_reversion.py` + `tests/challenge/test_strategy_lag2_reversion.py` (10 tests)
- [x] H5: WeeklyCycleStrategy — `strategies/weekly_cycle.py` + `tests/challenge/test_strategy_weekly_cycle.py` (10 tests)
- [x] H6: FossilDispatchStrategy — `strategies/fossil_dispatch.py` + `tests/challenge/test_strategy_fossil_dispatch.py` (9 tests)
- [x] H7: CompositeSignalStrategy — `strategies/composite_signal.py` + `tests/challenge/test_strategy_composite_signal.py` (9 tests)

### 4b. Update `strategies/__init__.py`
- [x] Add imports for all 7 new strategies (9 total with baselines)
- [x] Update `__all__` list
- [x] Update discovery test (`test_discover_returns_all_strategies` → expects 9)

### 4c. Testing
- [x] Each strategy has a dedicated test file with interface, signal, edge-case tests
- [x] Full test suite: **215 tests pass, 0 failures** (11 pre-existing `data_collection` collection errors)
- [x] Test breakdown: 70 new strategy tests + 145 prior tests

## Strategy Implementation Template

```python
"""[Description] strategy for the challenge dashboard."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class [Name]Strategy(ChallengeStrategy):
    """[Docstring explaining the hypothesis and signal logic]."""

    def fit(self, train_data: pd.DataFrame) -> None:
        # Optional: compute parameters from training data
        pass

    def act(self, state: ChallengeState) -> int | None:
        # Return +1 (long), -1 (short), or None (skip)
        pass

    def reset(self) -> None:
        pass
```

## ChallengeStrategy Interface Reference

```python
class ChallengeStrategy(ABC):
    def fit(self, train_data: pd.DataFrame) -> None: ...   # optional training
    def act(self, state: ChallengeState) -> int | None: ... # abstract
    def reset(self) -> None: ...                             # reset between runs
```

```python
@dataclass(frozen=True)
class ChallengeState:
    delivery_date: date
    last_settlement_price: float
    features: pd.Series           # current-day features (no labels)
    history: pd.DataFrame         # all prior rows (with labels)
```

## PnL Formula

```
PnL = direction * (settlement_price - last_settlement_price) * 24
```

Where `direction` is `+1` (long) or `-1` (short). `None` means skip (PnL = 0).

## Strategies Implemented

| Strategy | File | Hypothesis | EDA Sections | Tests | Status |
|----------|------|-----------|-------------|-------|--------|
| AlwaysLong | `strategies/always_long.py` | Baseline (always +1) | — | 3 in submission tests | EXISTING |
| AlwaysShort | `strategies/always_short.py` | Baseline (always -1) | — | 3 in submission tests | EXISTING |
| DayOfWeek | `strategies/day_of_week.py` | H1: Mon long, Sat/Sun short, Wed/Thu skip | 13, 19 | 12 | COMPLETE |
| WindForecast | `strategies/wind_forecast.py` | H2: High wind → short, low wind → long | 15, 16, 23 | 10 | COMPLETE |
| LoadForecast | `strategies/load_forecast.py` | H3: High load → long, low load → short | 15, 16 | 10 | COMPLETE |
| Lag2Reversion | `strategies/lag2_reversion.py` | H4: Fade large moves from 2 days ago | 13, 14 | 10 | COMPLETE |
| WeeklyCycle | `strategies/weekly_cycle.py` | H5: Follow same-day-of-week direction | 13, 14 | 10 | COMPLETE |
| FossilDispatch | `strategies/fossil_dispatch.py` | H6: High fossil gen → short | 16, 18 | 9 | COMPLETE |
| CompositeSignal | `strategies/composite_signal.py` | H7: Weighted z-score ensemble | 16, 17, 24 | 9 | COMPLETE |

## Implementation Notes

### Design Decisions

1. **Threshold-based strategies (H2, H3, H6)**: Use median from training data as threshold.
   Median is robust to outliers and provides a natural 50/50 split. The `fit()` method
   computes the threshold; `reset()` clears it; `act()` raises RuntimeError if called before `fit()`.

2. **History-based strategies (H4, H5)**: Use `state.history` DataFrame which includes
   label columns (since those rows are already realised). Return `None` when insufficient
   history is available.

3. **Stateless strategy (H1)**: DayOfWeek requires no fitting or state — purely
   derived from `state.delivery_date.isoweekday()`.

4. **Composite strategy (H7)**: Uses hardcoded EDA-derived correlation weights (not re-estimated
   from training data) for simplicity and transparency. Only the feature means/stds are
   learned during `fit()`.

### PerfectForesight and SkipAll

The original Phase 4 doc mentioned these as items 4a/4b. They were not implemented because:
- PerfectForesight is relevant for Phase 7 (convergence analysis), not general strategy evaluation
- SkipAll is trivially `act() -> None` and doesn't provide useful signal for Phase 5 evaluation
- These can be added later if needed
