# Phase F: Ensemble / Meta Strategies

## Status: ⏳ Pending

## Objective

Implement ~12 strategies that combine outputs from 2+ existing strategies
or use voting/blending mechanisms. A shared `strategies/ensemble_base.py`
provides the blending infrastructure.

---

## Strategy Inventory

| # | Class | File | Components | Blend Method | Status |
|---|-------|------|-----------|--------------|--------|
| 1 | `DOWWindCompositeStrategy` | `dow_wind_composite.py` | DOW + Wind | Equal delta blend | ⏳ |
| 2 | `DOWNetDemandStrategy` | `dow_net_demand.py` | DOW + NetDemand | Equal delta blend | ⏳ |
| 3 | `TripleSignalStrategy` | `triple_signal.py` | DOW + Wind + Load | Equal 3-way blend | ⏳ |
| 4 | `TopThreeEnsembleStrategy` | `top_three_ensemble.py` | Top-3 by training Sharpe | Majority vote | ⏳ |
| 5 | `ThresholdMajorityVoteStrategy` | `threshold_majority_vote.py` | All simple threshold strategies | Majority vote | ⏳ |
| 6 | `WeightedVoteStrategy` | `weighted_vote.py` | All strategies | Vote weighted by training Sharpe | ⏳ |
| 7 | `StackedEnsembleStrategy` | `stacked_ensemble.py` | All strategies | Logistic regression on predictions | ⏳ |
| 8 | `DOWLassoStrategy` | `dow_lasso.py` | DOW on Mon/Sat/Sun, Lasso otherwise | Conditional | ⏳ |
| 9 | `SignalCountStrategy` | `signal_count.py` | Selected high-corr strategies | Count ≥ threshold → trade | ⏳ |
| 10 | `LowVolMomentumStrategy` | `low_vol_momentum.py` | RollingMomentum5d in low-vol regime | Regime-conditional | ⏳ |
| 11 | `ContraDOWStrategy` | `contra_dow.py` | DOW + NetDemand | Fade DOW when NetDemand disagrees | ⏳ |
| 12 | `AdaptiveWeightEnsembleStrategy` | `adaptive_weight_ensemble.py` | DOW + Composite + NetDemand | 30-day rolling performance weights | ⏳ |

---

## Ensemble Base Class

`strategies/ensemble_base.py`:

```python
class _EnsembleBase(BacktestStrategy):
    """Base class for strategies that blend sub-strategy forecasts."""

    def __init__(self, sub_strategies: list[BacktestStrategy]) -> None:
        self._sub = sub_strategies

    def fit(self, train_data: pd.DataFrame) -> None:
        for s in self._sub:
            s.fit(train_data)

    def reset(self) -> None:
        for s in self._sub:
            s.reset()

    def _deltas(self, state: BacktestState) -> list[float]:
        return [s.forecast(state) - state.last_settlement_price for s in self._sub]
```

---

## Stacked Ensemble Detail

`StackedEnsembleStrategy` collects in-sample predictions from all sub-strategies
during `fit()` using walk-forward cross-validation, then trains a logistic
regression on those predictions to find optimal combination weights.

This is the most sophisticated ensemble and requires careful implementation to
avoid look-ahead bias during the stacking training phase.

---

## AdaptiveWeightEnsemble Detail

In `forecast()`, computes a 30-day trailing PnL for each sub-strategy using
`state.history`. Assigns weights proportional to recent PnL (non-negative).
If all sub-strategies lost money in the trailing window, falls back to equal
weights.

This creates a naturally adaptive strategy that increases weight on
currently-working signals.

---

## Completion Criteria

- [ ] `strategies/ensemble_base.py` created
- [ ] 12 strategy files created
- [ ] 5+ tests per strategy (12 test files)
- [ ] All registered in `strategies/__init__.py`
- [ ] All tests pass
- [ ] `StackedEnsembleStrategy` has look-ahead bias test
