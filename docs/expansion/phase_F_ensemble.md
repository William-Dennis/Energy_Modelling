# Phase F: Ensemble / Meta Strategies

## Status: ✅ Complete

## Objective

Implement ~12 strategies that combine outputs from 2+ existing strategies
or use voting/blending mechanisms. A shared `strategies/ensemble_base.py`
provides the blending infrastructure.

---

## Strategy Inventory

| # | Class | File | Components | Blend Method | Status |
|---|-------|------|-----------|--------------|--------|
| 1 | `ConsensusSignalStrategy` | `consensus_signal.py` | Unanimous 3-member consensus | All-agree filter | ✅ |
| 2 | `MajorityVoteRuleBasedStrategy` | `majority_vote_rule.py` | Rule-based majority vote | Majority direction | ✅ |
| 3 | `MajorityVoteMLStrategy` | `majority_vote_ml.py` | ML classifier majority vote | Majority direction | ✅ |
| 4 | `MeanForecastRegressionStrategy` | `mean_forecast_regression.py` | Mean regression forecast | Mean blend | ✅ |
| 5 | `MedianForecastEnsembleStrategy` | `median_forecast_ensemble.py` | Median regression forecast | Median blend | ✅ |
| 6 | `TopKEnsembleStrategy` | `top_k_ensemble.py` | Top-K by validation Sharpe | Majority vote | ✅ |
| 7 | `WeightedVoteMixedStrategy` | `weighted_vote_mixed.py` | Weighted rule+ML vote | Performance-weighted | ✅ |
| 8 | `DiversityEnsembleStrategy` | `diversity_ensemble.py` | Diverse 3-source ensemble | Majority direction | ✅ |
| 9 | `RegimeConditionalEnsembleStrategy` | `regime_conditional_ensemble.py` | Vol-regime conditional ensemble | Regime-switch | ✅ |
| 10 | `StackedRidgeMetaStrategy` | `stacked_ridge_meta.py` | Stacked Ridge meta-learner | Ridge on predictions | ✅ |
| 11 | `WeekdayWeekendEnsembleStrategy` | `weekday_weekend_ensemble.py` | Weekday/weekend dual ensemble | Calendar-conditional | ✅ |
| 12 | `BoostedSpreadMLStrategy` | `boosted_spread_ml.py` | Spread+GBM agreement filter | Agreement gate | ✅ |

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

- [x] `strategies/ensemble_base.py` created
- [x] 12 strategy files created
- [x] 5+ tests per strategy (12 test files)
- [x] All registered in `strategies/__init__.py`
- [x] All tests pass
- [x] `StackedRidgeMetaStrategy` has look-ahead bias test
