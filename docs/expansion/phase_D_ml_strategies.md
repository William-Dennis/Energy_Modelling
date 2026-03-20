# Phase D: ML Model Strategies

## Status: ⏳ Pending

## Objective

Implement ~15 strategies based on different ML models, feature sets, and
training configurations. A shared `_MLStrategyBase` class in
`strategies/common.py` (or a new `strategies/ml_base.py`) handles the common
pipeline logic (feature selection, scaling, CV, predict).

---

## Strategy Inventory

| # | Class | Model | Feature Set | CV | Status |
|---|-------|-------|------------|-----|--------|
| 1 | `RidgeRegressionStrategy` | Ridge (L2) | All 47 features | 5-fold TSCV alpha | ⏳ |
| 2 | `ElasticNetStrategy` | ElasticNet | All features | 5-fold TSCV (alpha, l1_ratio) | ⏳ |
| 3 | `LogisticDirectionStrategy` | Logistic Regression | All features | 5-fold TSCV C | ⏳ |
| 4 | `RandomForestStrategy` | RandomForestClassifier | All features | 5-fold TSCV max_depth | ⏳ |
| 5 | `GradientBoostingStrategy` | GradientBoostingClassifier | All features | 5-fold TSCV n_estimators | ⏳ |
| 6 | `LassoTopFeaturesStrategy` | Lasso | Top 10 by corr | 5-fold TSCV alpha | ⏳ |
| 7 | `RidgeNetDemandStrategy` | Ridge | Derived features only | 5-fold TSCV alpha | ⏳ |
| 8 | `KNNDirectionStrategy` | KNeighborsClassifier | Z-scored features | 5-fold TSCV n_neighbors | ⏳ |
| 9 | `SVMDirectionStrategy` | LinearSVC | Z-scored features | 5-fold TSCV C | ⏳ |
| 10 | `DecisionTreeStrategy` | DecisionTreeClassifier | All features | 5-fold TSCV max_depth | ⏳ |
| 11 | `LassoCalendarAugmentedStrategy` | Lasso | All + DOW/month dummies | 5-fold TSCV alpha | ⏳ |
| 12 | `GBMNetDemandStrategy` | GradientBoosting | Derived features | Fixed params | ⏳ |
| 13 | `BayesianRidgeStrategy` | BayesianRidge | All features | None (auto-regularised) | ⏳ |
| 14 | `PLSRegressionStrategy` | PLSRegression | All features | 5-fold TSCV n_components | ⏳ |
| 15 | `NeuralNetStrategy` | MLPClassifier | All features | 5-fold TSCV hidden_layer_sizes | ⏳ |

---

## Shared ML Base Class

`strategies/ml_base.py`:

```python
class _MLStrategyBase(BacktestStrategy):
    """Shared infrastructure for all ML-based strategies."""

    _EXCLUDE_COLUMNS: frozenset = ...  # same as LassoRegressionStrategy

    def _get_feature_cols(self, train_data) -> list[str]: ...
    def _build_X(self, feature_cols, data_or_series) -> np.ndarray: ...
```

The base handles:
- Column exclusion
- NaN fill
- Feature vector construction at forecast time

Each subclass provides its own `_build_pipeline()` and CV logic.

---

## Feature Sets

### "All features" (47 total)
The 29 raw features + 18 derived features from Phase A.
Includes `dow_int`, `is_weekend` as integer/bool encodings.

### "Derived features only" (18)
Only the features computed in Phase A. Useful for testing whether derived
features add signal beyond what's already in the raw set.

### "Top 10 by corr"
The 10 highest-correlation raw features:
`load_forecast_mw_mean`, `forecast_wind_offshore_mw_mean`, `gen_wind_onshore_mw_mean`,
`weather_wind_speed_10m_kmh_mean`, `gen_fossil_gas_mw_mean`, `flow_nl_net_import_mw_mean`,
`gen_fossil_brown_coal_lignite_mw_mean`, `forecast_wind_onshore_mw_mean`,
`gen_wind_offshore_mw_mean`, `gen_fossil_hard_coal_mw_mean`.

---

## Computational Budget

| Model | Train time | CV overhead |
|-------|-----------|-------------|
| Ridge/Lasso/ElasticNet | <100ms | ×9 alphas × 5 folds = <5s total |
| Logistic/SVM | <200ms | <10s total |
| RandomForest | <1s | <20s total |
| GradientBoosting | <2s | <30s total |
| KNN | <50ms | <5s total |
| BayesianRidge | <100ms | None |
| PLS | <100ms | <5s total |
| MLP | <500ms | <25s total |

**Total for all 15 ML strategies: ~2 minutes**. Feasible for batch compute.
The dashboard loads from saved pickle results, so live training only happens
once per `recompute-all` CLI run.

---

## Completion Criteria

- [ ] `strategies/ml_base.py` created with shared base class
- [ ] 15 ML strategy files created
- [ ] 5+ tests per strategy
- [ ] All registered in `strategies/__init__.py`
- [ ] All tests pass
- [ ] `recompute-all` completes without timeout
