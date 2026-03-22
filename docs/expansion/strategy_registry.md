# Strategy Registry (Live)

## Current Count: 100

Last updated: Phase 12B complete. 26 new strategies added (Phases B-F: 67, Phase 11: +7, Phase 12B: +26).

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
| 10 | `LassoRegressionStrategy` | `lasso_regression.py` | Lasso on all features | ✅ |
| 11 | `DowCompositeStrategy` | `dow_composite.py` | DOW + CompositeSignal blend | ✅ |

---

## Phase B: Issue 3 Strategies (+7, total 18)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 12 | `SolarForecastStrategy` | `solar_forecast.py` | Solar merit-order | ✅ |
| 13 | `CommodityCostStrategy` | `commodity_cost.py` | Gas/carbon fuel index | ✅ |
| 14 | `TemperatureExtremeStrategy` | `temperature_extreme.py` | Temperature non-linear demand | ✅ |
| 15 | `CrossBorderSpreadStrategy` | `cross_border_spread.py` | FR/NL cross-border spread | ✅ |
| 16 | `VolatilityRegimeStrategy` | `volatility_regime.py` | Vol regime switching | ✅ |
| 17 | `NuclearAvailabilityStrategy` | `nuclear_availability.py` | Nuclear outage event | ✅ |
| 18 | `RenewablesSurplusStrategy` | `renewables_surplus.py` | Extreme renewables regime | ✅ |

---

## Phase C: Derived-Feature Threshold Strategies (+14, total 32)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 19 | `NetDemandStrategy` | `net_demand.py` | Net demand threshold | ✅ |
| 20 | `PriceZScoreReversionStrategy` | `price_zscore_reversion.py` | Price z-score reversion | ✅ |
| 21 | `GasTrendStrategy` | `gas_trend.py` | Gas 3-day momentum | ✅ |
| 22 | `CarbonTrendStrategy` | `carbon_trend.py` | Carbon 3-day momentum | ✅ |
| 23 | `FuelIndexTrendStrategy` | `fuel_index_trend.py` | Combined fuel momentum | ✅ |
| 24 | `DEFRSpreadStrategy` | `de_fr_spread.py` | DE-FR spread convergence | ✅ |
| 25 | `DENLSpreadStrategy` | `de_nl_spread.py` | DE-NL spread convergence | ✅ |
| 26 | `MultiSpreadStrategy` | `multi_spread.py` | Multi-market avg spread | ✅ |
| 27 | `NLFlowSignalStrategy` | `nl_flow_signal.py` | NL net import flow | ✅ |
| 28 | `FRFlowSignalStrategy` | `fr_flow_signal.py` | FR net import flow | ✅ |
| 29 | `PriceMinReversionStrategy` | `price_min_reversion.py` | Price-min mean reversion | ✅ |
| 30 | `WindForecastErrorStrategy` | `wind_forecast_error.py` | Wind forecast error | ✅ |
| 31 | `LoadSurpriseStrategy` | `load_surprise.py` | Load demand surprise | ✅ |
| 32 | `RenewablesPenetrationStrategy` | `renewables_penetration.py` | Renewable share threshold | ✅ |

---

## Phase D: ML Strategies (+15, total 47)

| # | Class | File | Model | Status |
|---|-------|------|-------|--------|
| 33 | `RidgeRegressionStrategy` | `ridge_regression.py` | Ridge (L2) | ✅ |
| 34 | `ElasticNetStrategy` | `elastic_net.py` | ElasticNet | ✅ |
| 35 | `LogisticDirectionStrategy` | `logistic_direction.py` | Logistic Regression | ✅ |
| 36 | `RandomForestStrategy` | `random_forest_direction.py` | RandomForest | ✅ |
| 37 | `GradientBoostingStrategy` | `gradient_boosting_direction.py` | GBM | ✅ |
| 38 | `LassoTopFeaturesStrategy` | `lasso_top_features.py` | Lasso (top 10) | ✅ |
| 39 | `RidgeNetDemandStrategy` | `ridge_net_demand.py` | Ridge (derived) | ✅ |
| 40 | `KNNDirectionStrategy` | `knn_direction.py` | KNN | ✅ |
| 41 | `SVMDirectionStrategy` | `svm_direction.py` | LinearSVC | ✅ |
| 42 | `DecisionTreeStrategy` | `decision_tree_direction.py` | Decision Tree | ✅ |
| 43 | `LassoCalendarAugmentedStrategy` | `lasso_calendar_augmented.py` | Lasso + calendar | ✅ |
| 44 | `GBMNetDemandStrategy` | `gbm_net_demand.py` | GBM (derived) | ✅ |
| 45 | `BayesianRidgeStrategy` | `bayesian_ridge.py` | BayesianRidge | ✅ |
| 46 | `PLSRegressionStrategy` | `pls_regression.py` | PLS | ✅ |
| 47 | `NeuralNetStrategy` | `neural_net.py` | MLP | ✅ |

---

## Phase E: Calendar/Temporal/Regime Strategies (+8, total 55)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 48 | `MonthSeasonalStrategy` | `month_seasonal.py` | Monthly seasonal mean | ✅ |
| 49 | `MondayEffectStrategy` | `monday_effect.py` | Monday/Friday effect | ✅ |
| 50 | `QuarterSeasonalStrategy` | `quarter_seasonal.py` | Quarterly seasonal mean | ✅ |
| 51 | `ZScoreMomentumStrategy` | `zscore_momentum.py` | Z-score momentum follow | ✅ |
| 52 | `NetDemandMomentumStrategy` | `net_demand_momentum.py` | Net demand momentum | ✅ |
| 53 | `RenewableRegimeStrategy` | `renewable_regime.py` | Renewable penetration regime | ✅ |
| 54 | `VolatilityRegimeMLStrategy` | `volatility_regime_ml.py` | Volatility regime learned | ✅ |
| 55 | `GasCarbonJointTrendStrategy` | `gas_carbon_joint_trend.py` | Gas-carbon joint trend | ✅ |

---

## Phase F: Ensemble/Meta Strategies (+12, total 67)

| # | Class | File | Components | Status |
|---|-------|------|-----------|--------|
| 56 | `ConsensusSignalStrategy` | `consensus_signal.py` | Unanimous 3-member consensus | ✅ |
| 57 | `MajorityVoteRuleBasedStrategy` | `majority_vote_rule.py` | Rule-based majority vote | ✅ |
| 58 | `MajorityVoteMLStrategy` | `majority_vote_ml.py` | ML classifier majority vote | ✅ |
| 59 | `MeanForecastRegressionStrategy` | `mean_forecast_regression.py` | Mean regression forecast | ✅ |
| 60 | `MedianForecastEnsembleStrategy` | `median_forecast_ensemble.py` | Median regression forecast | ✅ |
| 61 | `TopKEnsembleStrategy` | `top_k_ensemble.py` | Top-K validation ensemble | ✅ |
| 62 | `WeightedVoteMixedStrategy` | `weighted_vote_mixed.py` | Weighted rule+ML vote | ✅ |
| 63 | `DiversityEnsembleStrategy` | `diversity_ensemble.py` | Diverse 3-source ensemble | ✅ |
| 64 | `RegimeConditionalEnsembleStrategy` | `regime_conditional_ensemble.py` | Vol-regime conditional ensemble | ✅ |
| 65 | `StackedRidgeMetaStrategy` | `stacked_ridge_meta.py` | Stacked Ridge meta-learner | ✅ |
| 66 | `WeekdayWeekendEnsembleStrategy` | `weekday_weekend_ensemble.py` | Weekday/weekend dual ensemble | ✅ |
| 67 | `BoostedSpreadMLStrategy` | `boosted_spread_ml.py` | Spread+GBM agreement filter | ✅ |

---

## Phase 11: New Strategies (+7, total 74)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 68 | `SpreadMomentumStrategy` | `spread_momentum.py` | Cross-border spread EMA momentum | ✅ |
| 69 | `SelectiveHighConvictionStrategy` | `selective_high_conviction.py` | Z-score filtered CompositeSignal | ✅ |
| 70 | `TemperatureCurveStrategy` | `temperature_curve.py` | Quadratic temperature-demand model | ✅ |
| 71 | `NuclearEventStrategy` | `nuclear_event.py` | Nuclear generation drop detector | ✅ |
| 72 | `FlowImbalanceStrategy` | `flow_imbalance.py` | Combined FR+NL flow imbalance | ✅ |
| 73 | `RegimeRidgeStrategy` | `regime_ridge.py` | Volatility-regime dual Ridge ML | ✅ |
| 74 | `PrunedMLEnsembleStrategy` | `pruned_ml_ensemble.py` | Ridge+Lasso+RF equal-weight ensemble | ✅ |

---

## Phase 12B Batch 1: Unused Features (+5, total 79)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 75 | `RadiationSolarStrategy` | `radiation_solar.py` | Shortwave radiation vs training median | ✅ |
| 76 | `IntradayRangeStrategy` | `intraday_range.py` | Price range percentile: mean-revert/trend | ✅ |
| 77 | `OffshoreWindAnomalyStrategy` | `offshore_wind_anomaly.py` | Actual vs forecast offshore wind | ✅ |
| 78 | `ForecastPriceErrorStrategy` | `forecast_price_error.py` | Self-correcting forecast feedback loop | ✅ |
| 79 | `PolandSpreadStrategy` | `poland_spread.py` | DE-PL spread convergence | ✅ |

---

## Phase 12B Batch 2: Spreads & Ratios (+5, total 84)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 80 | `DenmarkSpreadStrategy` | `denmark_spread.py` | DE-DK1 spread convergence | ✅ |
| 81 | `CzechAustrianMeanStrategy` | `czech_austrian_mean.py` | Average CZ+AT price vs DE spread | ✅ |
| 82 | `SparkSpreadStrategy` | `spark_spread.py` | Gas-to-electricity spark spread signal | ✅ |
| 83 | `CarbonGasRatioStrategy` | `carbon_gas_ratio.py` | Carbon/gas price ratio regime signal | ✅ |
| 84 | `WeeklyAutocorrelationStrategy` | `weekly_autocorrelation.py` | 7-day autocorrelation pattern | ✅ |

---

## Phase 12B Batch 3: Supply-Demand (+5, total 89)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 85 | `MonthlyMeanReversionStrategy` | `monthly_mean_reversion.py` | Calendar month average reversion | ✅ |
| 86 | `LoadGenerationGapStrategy` | `load_generation_gap.py` | Load vs total generation gap | ✅ |
| 87 | `RenewableRampStrategy` | `renewable_ramp.py` | Day-over-day renewable generation change | ✅ |
| 88 | `NuclearGasSubstitutionStrategy` | `nuclear_gas_substitution.py` | Nuclear-gas substitution dynamics | ✅ |
| 89 | `VolatilityBreakoutStrategy` | `volatility_breakout.py` | Momentum breakouts in high volatility | ✅ |

---

## Phase 12B Batch 4: Regime & Calendar (+5, total 94)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 90 | `SeasonalRegimeSwitchStrategy` | `seasonal_regime_switch.py` | Winter/summer regime switch | ✅ |
| 91 | `WeekendMeanReversionStrategy` | `weekend_mean_reversion.py` | Weekday/weekend price reversion | ✅ |
| 92 | `HighVolSkipStrategy` | `high_vol_skip.py` | Skip trading in high-volatility regimes | ✅ |
| 93 | `RadiationRegimeStrategy` | `radiation_regime.py` | Three-regime radiation classification | ✅ |
| 94 | `IndependentVoteStrategy` | `independent_vote.py` | Majority vote from 5 independent features | ✅ |

---

## Phase 12B Batch 5: Meta & Ensemble (+5, total 99)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 95 | `MedianIndependentStrategy` | `median_independent.py` | Median of 3 independent signals | ✅ |
| 96 | `SpreadConsensusStrategy` | `spread_consensus.py` | Multi-border spread consensus | ✅ |
| 97 | `SupplyDemandBalanceStrategy` | `supply_demand_balance.py` | Forecast load vs generation balance | ✅ |
| 98 | `ContrarianMomentumStrategy` | `contrarian_momentum.py` | Follow strong trends, fade weak | ✅ |
| 99 | `ConvictionWeightedStrategy` | `conviction_weighted.py` | Z-score mean-reversion with conviction | ✅ |

---

## Phase 12B Batch 6: Milestone (+1, total 100)

| # | Class | File | Signal | Status |
|---|-------|------|--------|--------|
| 100 | `BalancedLongShortStrategy` | `balanced_long_short.py` | Position-tracking balanced long/short | ✅ |
