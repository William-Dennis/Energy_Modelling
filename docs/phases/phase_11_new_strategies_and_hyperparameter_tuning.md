# Phase 11: New Strategies and Hyperparameter Tuning

## Status: COMPLETE

## Objective

Add 7 new strategies to the platform that address the redundancy, concentration,
and signal-diversity problems identified in Phase 10. The engine code is frozen;
only hyperparameter recommendations and new strategy implementations are in scope.

## Constraints

- **Engine is frozen**: No changes to any file under `src/energy_modelling/backtest/`.
- **Hyperparameter changes are OK**: Recommended values for `ema_alpha`,
  `max_iterations`, etc. can be documented and passed at call sites.
- **Efficiency**: All new strategies must be fast. `fit()` under 1 second on
  typical training data (~300-400 rows). `forecast()` under 1 millisecond.
  No heavy ML models (no neural nets, no gradient boosting with >100 trees).

## Hyperparameter Recommendations

Based on Phase 10c findings, future market simulation runs should use:

| Parameter | Current Default | Recommended | Rationale |
|-----------|----------------|-------------|-----------|
| `ema_alpha` | 0.1 | 0.01 | Only value achieving healthy convergence for both 2024 and 2025 |
| `max_iterations` | 500 | 200 | MAE plateaus by iter 90-100; 200 captures all improvement with 60% compute savings |
| `convergence_threshold` | 0.01 | 0.01 | No change needed |

These values should be passed explicitly when calling `run_futures_market()` or
`run_futures_market_evaluation()`. The engine defaults are not changed.

> **NOTE (Phase 13, 2026-03-22):** The `ema_alpha=0.01` recommendation above has
> now been implemented. `recompute.py` passes `ema_alpha=0.01` explicitly to both
> `run_futures_market_evaluation()` calls. The engine function defaults remain 0.1
> for backwards compatibility. The regenerated artifacts (`market_2024.pkl`,
> `market_2025.pkl`) reflect this change.

## New Strategies (7)

### 1. SpreadMomentumStrategy (rule-based, cross-border)

- **Signal**: 3-day EMA of the average DE-FR/DE-NL spread
- **Logic**: Long when spread EMA is rising and positive; short when falling and
  negative; skip otherwise
- **Rationale**: Cross-border spread was the most positively contributing signal
  family in Phase 10f LOO analysis

### 2. SelectiveHighConvictionStrategy (meta-wrapper)

- **Signal**: Wraps CompositeSignalStrategy; only trades on high-conviction days
- **Logic**: Compute rolling z-score of forecast deviation from entry price.
  Trade only when |z| > 1.5; forecast = entry price on low-conviction days (no trade)
- **Rationale**: Phase 10e found the market degrades per-date accuracy by
  over-weighting strategies profitable overall but poor on individual days

### 3. TemperatureCurveStrategy (rule-based, weather)

- **Signal**: Quadratic temperature-demand model capturing heating/cooling asymmetry
- **Logic**: Fit quadratic on temperature vs price change during training.
  Forecast = entry + predicted change from quadratic model
- **Rationale**: Temperature Extreme exists but is a simple threshold; no strategy
  models the U-shaped temperature-demand relationship

### 4. NuclearEventStrategy (rule-based, binary)

- **Signal**: Sharp drop in nuclear generation availability
- **Logic**: If nuclear generation drops >15% from its rolling 7-day mean,
  forecast a price increase of mean_abs_change magnitude; otherwise neutral
- **Rationale**: Rare but high-impact supply-side signal currently unused

### 5. FlowImbalanceStrategy (rule-based, cross-border flow)

- **Signal**: Combined FR+NL flow imbalance
- **Logic**: When net imports exceed the historical 75th percentile, go short
  (DE is importing = foreign prices pulling DE up, expect mean-reversion).
  When below 25th percentile, go long
- **Rationale**: Cross-border flow dynamics are underrepresented; only 2 flow
  strategies exist vs 6 price-spread strategies

### 6. RegimeRidgeStrategy (lightweight ML, volatility-adaptive)

- **Signal**: Two separate Ridge regression models for low-vol and high-vol regimes
- **Logic**: During `fit()`, split training data at median `rolling_vol_7d`.
  Train a Ridge on each half. During `forecast()`, check current volatility
  regime and use the appropriate model
- **Rationale**: Phase 10d found ML cluster is more robust to volatility but
  rule-based strategies show larger regime-dependent accuracy gaps

### 7. PrunedMLEnsembleStrategy (ensemble, diversity-focused)

- **Signal**: Equal-weight ensemble of Ridge, Lasso, and RandomForest (the 3 most
  structurally different ML approaches)
- **Logic**: Average the 3 member forecasts. Uses existing _EnsembleBase.
- **Rationale**: Current 11 ML strategies have >0.99 pairwise correlation.
  Pruning to 3 structurally different models reduces redundancy

## Acceptance Criteria

From Phase 10g, each new strategy must satisfy:

1. Tests pass (5+ per strategy)
2. All 1089+ existing tests still pass
3. Ruff clean, theorems verified
4. Implementation is fast (no heavy training)

Market-level acceptance criteria (standalone PnL, LOO delta-MAE, orthogonality,
balanced exposure) will be evaluated in a future re-run phase once all strategies
are registered.

## Checklist

- [x] Implement SpreadMomentumStrategy + tests
- [x] Implement SelectiveHighConvictionStrategy + tests
- [x] Implement TemperatureCurveStrategy + tests
- [x] Implement NuclearEventStrategy + tests
- [x] Implement FlowImbalanceStrategy + tests
- [x] Implement RegimeRidgeStrategy + tests
- [x] Implement PrunedMLEnsembleStrategy + tests
- [x] Register all 7 in `strategies/__init__.py`
- [x] Full verification: ruff + pytest + verify_theorems
- [x] Update ROADMAP.md and AGENTS.md
- [x] Update strategy_registry.md (67 -> 74 strategies)
