# Strategy Registry (Live)

## Current Count: 11 → target 68+

Last updated: Phase A in progress.

---

## Baseline Strategies (11)

| # | Class | File | Signal | Tests |
|---|-------|------|--------|-------|
| 1 | `AlwaysLongStrategy` | `always_long.py` | Naive long baseline | — |
| 2 | `AlwaysShortStrategy` | `always_short.py` | Naive short baseline | — |
| 3 | `DayOfWeekStrategy` | `day_of_week.py` | Calendar DOW | ✅ |
| 4 | `Lag2ReversionStrategy` | `lag2_reversion.py` | Lag-2 autocorrelation | ✅ |
| 5 | `WeeklyCycleStrategy` | `weekly_cycle.py` | Lag-7 autocorrelation | ✅ |
| 6 | `WindForecastStrategy` | `wind_forecast.py` | Wind merit-order | ✅ |
| 7 | `LoadForecastStrategy` | `load_forecast.py` | Load demand | ✅ |
| 8 | `FossilDispatchStrategy` | `fossil_dispatch.py` | Fossil dispatch | ✅ |
| 9 | `CompositeSignalStrategy` | `composite_signal.py` | Weighted z-score composite | ✅ |
| 10 | `LassoRegressionStrategy` | `lasso_regression.py` | Lasso on all features | — |
| 11 | `DowCompositeStrategy` | `dow_composite.py` | DOW + CompositeSignal blend | — |

---

## Phase B: Issue 3 Strategies (target +7, total 18)

| # | Class | File | Status |
|---|-------|------|--------|
| 12 | `SolarForecastStrategy` | `solar_forecast.py` | ⏳ |
| 13 | `CommodityCostStrategy` | `commodity_cost.py` | ⏳ |
| 14 | `TemperatureExtremeStrategy` | `temperature_extreme.py` | ⏳ |
| 15 | `CrossBorderSpreadStrategy` | `cross_border_spread.py` | ⏳ |
| 16 | `VolatilityRegimeStrategy` | `volatility_regime.py` | ⏳ |
| 17 | `NuclearAvailabilityStrategy` | `nuclear_availability.py` | ⏳ |
| 18 | `RenewablesSurplusStrategy` | `renewables_surplus.py` | ⏳ |

---

## Phase C: Derived-Feature Threshold Strategies (target +15, total 33)

| # | Class | File | Status |
|---|-------|------|--------|
| 19 | `NetDemandStrategy` | `net_demand.py` | ⏳ |
| 20 | `NetDemandWithSolarStrategy` | `net_demand_solar.py` | ⏳ |
| 21 | `PriceZScoreReversionStrategy` | `price_zscore_reversion.py` | ⏳ |
| 22 | `GasTrendStrategy` | `gas_trend.py` | ⏳ |
| 23 | `CarbonTrendStrategy` | `carbon_trend.py` | ⏳ |
| 24 | `FuelIndexTrendStrategy` | `fuel_index_trend.py` | ⏳ |
| 25 | `DEFRSpreadStrategy` | `de_fr_spread.py` | ⏳ |
| 26 | `DENLSpreadStrategy` | `de_nl_spread.py` | ⏳ |
| 27 | `MultiSpreadStrategy` | `multi_spread.py` | ⏳ |
| 28 | `NLFlowSignalStrategy` | `nl_flow_signal.py` | ⏳ |
| 29 | `FRFlowSignalStrategy` | `fr_flow_signal.py` | ⏳ |
| 30 | `PriceMinReversionStrategy` | `price_min_reversion.py` | ⏳ |
| 31 | `WindForecastErrorStrategy` | `wind_forecast_error.py` | ⏳ |
| 32 | `LoadSurpriseStrategy` | `load_surprise.py` | ⏳ |
| 33 | `RenewablesPenetrationStrategy` | `renewables_penetration.py` | ⏳ |

---

## Phase D: ML Strategies (target +15, total 48)

| # | Class | File | Model | Status |
|---|-------|------|-------|--------|
| 34 | `RidgeRegressionStrategy` | `ridge_regression.py` | Ridge | ⏳ |
| 35 | `ElasticNetStrategy` | `elastic_net.py` | ElasticNet | ⏳ |
| 36 | `LogisticDirectionStrategy` | `logistic_direction.py` | Logistic | ⏳ |
| 37 | `RandomForestStrategy` | `random_forest.py` | RandomForest | ⏳ |
| 38 | `GradientBoostingStrategy` | `gradient_boosting.py` | GBM | ⏳ |
| 39 | `LassoTopFeaturesStrategy` | `lasso_top_features.py` | Lasso (top 10) | ⏳ |
| 40 | `RidgeNetDemandStrategy` | `ridge_net_demand.py` | Ridge (derived) | ⏳ |
| 41 | `KNNDirectionStrategy` | `knn_direction.py` | KNN | ⏳ |
| 42 | `SVMDirectionStrategy` | `svm_direction.py` | LinearSVC | ⏳ |
| 43 | `DecisionTreeStrategy` | `decision_tree.py` | Decision Tree | ⏳ |
| 44 | `LassoCalendarAugmentedStrategy` | `lasso_calendar.py` | Lasso + calendar | ⏳ |
| 45 | `GBMNetDemandStrategy` | `gbm_net_demand.py` | GBM (derived) | ⏳ |
| 46 | `BayesianRidgeStrategy` | `bayesian_ridge.py` | BayesianRidge | ⏳ |
| 47 | `PLSRegressionStrategy` | `pls_regression.py` | PLS | ⏳ |
| 48 | `NeuralNetStrategy` | `neural_net.py` | MLP | ⏳ |

---

## Phase E: Calendar/Temporal/Regime Strategies (target +8, total 56)

| # | Class | File | Status |
|---|-------|------|--------|
| 49 | `MonthOfYearStrategy` | `month_of_year.py` | ⏳ |
| 50 | `DayOfWeekFilteredWindStrategy` | `dow_filtered_wind.py` | ⏳ |
| 51 | `WeekendOnlyStrategy` | `weekend_only.py` | ⏳ |
| 52 | `Lag1ReversionStrategy` | `lag1_reversion.py` | ⏳ |
| 53 | `Lag3CycleStrategy` | `lag3_cycle.py` | ⏳ |
| 54 | `RollingMomentum5dStrategy` | `rolling_momentum_5d.py` | ⏳ |
| 55 | `RollingMomentum10dStrategy` | `rolling_momentum_10d.py` | ⏳ |
| 56 | `HighVolMeanReversionStrategy` | `high_vol_mean_reversion.py` | ⏳ |

---

## Phase F: Ensemble/Meta Strategies (target +12, total 68)

| # | Class | File | Components | Status |
|---|-------|------|-----------|--------|
| 57 | `DOWWindCompositeStrategy` | `dow_wind_composite.py` | DOW + Wind | ⏳ |
| 58 | `DOWNetDemandStrategy` | `dow_net_demand.py` | DOW + NetDemand | ⏳ |
| 59 | `TripleSignalStrategy` | `triple_signal.py` | DOW + Wind + Load | ⏳ |
| 60 | `TopThreeEnsembleStrategy` | `top_three_ensemble.py` | Top-3 Sharpe | ⏳ |
| 61 | `ThresholdMajorityVoteStrategy` | `threshold_majority_vote.py` | All threshold vote | ⏳ |
| 62 | `WeightedVoteStrategy` | `weighted_vote.py` | All weighted | ⏳ |
| 63 | `StackedEnsembleStrategy` | `stacked_ensemble.py` | All (stacked LR) | ⏳ |
| 64 | `DOWLassoStrategy` | `dow_lasso.py` | DOW + Lasso | ⏳ |
| 65 | `SignalCountStrategy` | `signal_count.py` | Threshold count | ⏳ |
| 66 | `LowVolMomentumStrategy` | `low_vol_momentum.py` | Momentum in low-vol | ⏳ |
| 67 | `ContraDOWStrategy` | `contra_dow.py` | DOW × NetDemand | ⏳ |
| 68 | `AdaptiveWeightEnsembleStrategy` | `adaptive_weight_ensemble.py` | Rolling weights | ⏳ |
