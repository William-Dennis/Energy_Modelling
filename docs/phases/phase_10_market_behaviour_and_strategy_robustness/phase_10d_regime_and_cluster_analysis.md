# Phase 10d: Regime and Cluster Analysis

## Status: COMPLETE

## Objective

Explain market behaviour through the structure of the strategy pool: forecast
clusters, profit clusters, regime dependence, and cross-year differences.

## Findings

### Forecast Clustering Structure

Hierarchical clustering by forecast correlation reveals a highly concentrated
structure in both years:

| Year | Cluster 1 (ML regression) | Cluster 2/3 (broad mix) | Singletons |
|------|--------------------------|------------------------|------------|
| 2024 | 11 strategies | 49 strategies | 7 (1 each) |
| 2025 | 12 strategies | 49 strategies | 6 (1 each) |

**Key finding**: The forecast space is dominated by **two large clusters**:
1. **ML regression cluster** (11-12 strategies): Lasso, Ridge, ElasticNet,
   BayesianRidge, PLS, Ridge Net Demand, Stacked Ridge Meta, Mean/Median
   Forecast ensembles. These produce highly correlated forecasts.
2. **Broad mix cluster** (49 strategies): Everything else — baselines, calendar,
   momentum, supply, demand, cross-border, ML classification, most ensembles.
   Despite diverse approaches, their forecasts are more correlated with each
   other than with the ML regression cluster.

The remaining 6-7 strategies form singletons: Load Surprise, Multi Spread,
Price Min Reversion, Weekly Cycle, Lag2 Reversion, and (in 2024) Fossil Dispatch.

### Forecast vs Profit Cluster Disagreement

Forecast clusters and profit clusters do **not** align well:

- **2024**: Forecast cluster 1 (ML regression) maps entirely to profit cluster 2.
  Forecast cluster 3 (49 strategies) fragments into **8 different profit clusters**.
  Strategies that forecast similarly can have very different profit profiles
  depending on when they are right vs wrong.

- **2025**: Similar pattern. Forecast cluster 1 maps to profit cluster 1.
  Forecast cluster 2 (49 strategies) fragments into **8 profit clusters**.

**Implication**: Forecast similarity is a poor predictor of profit similarity.
The market engine's sign-based trading rule and cumulative scoring create very
different profit landscapes even for strategies with similar forecasts.

### Cluster Dominance

The ML regression cluster (forecast cluster 1) dominates the market in both
years:

| Year | ML Cluster Early Weight | ML Cluster Late Weight | Broad Cluster Early | Broad Cluster Late |
|------|------------------------|----------------------|--------------------|--------------------|
| 2024 | 0.8948 | 0.9125 | 0.0520 | 0.0334 |
| 2025 | 0.9197 | 0.9147 | 0.0469 | 0.0298 |

**The ML regression cluster captures >90% of the market weight throughout both
years.** The 49-strategy broad cluster contributes only 3-5% of weight. The
remaining singletons contribute <1% each (and most collapse to 0 weight).

This means the market price is almost entirely determined by the ML regression
cluster's forecasts. The other 55+ strategies contribute very little to
price formation despite participating in the market.

### Regime-Dependent Forecast Accuracy

Both years show consistent regime effects:

| Year | Regime | ML Cluster MAE | Broad Cluster MAE | Gap |
|------|--------|---------------|-------------------|-----|
| 2024 | Low vol | 8.71 | 15.17 | 6.46 |
| 2024 | High vol | 13.36 | 25.70 | 12.34 |
| 2025 | Low vol | 8.08 | 16.50 | 8.42 |
| 2025 | High vol | 10.57 | 22.17 | 11.60 |

**Findings**:
1. The ML regression cluster is consistently more accurate than the broad cluster
   in both regimes and both years.
2. The accuracy gap **widens in high-volatility periods**: the ML cluster
   degrades less than the broad cluster when volatility rises.
3. Both clusters show negative bias (under-forecasting) in most conditions.
4. The ML cluster achieves MAE ~8-13 EUR/MWh vs the broad cluster's 15-26.

### Singleton Strategies

The singleton clusters (Load Surprise, Multi Spread, Price Min Reversion,
Weekly Cycle, Lag2 Reversion, Fossil Dispatch) all have:
- Zero or near-zero late-iteration weight
- High MAE (19-34 EUR/MWh)
- Unique forecast patterns that don't correlate with either main cluster

These strategies are **informationally redundant** in the context of the
market engine — they provide neither accuracy nor diversity that survives the
profit-weighting mechanism.

## Cross-Year Comparison

The cluster structure is remarkably stable across years:
- Same dominant cluster (ML regression) with 90%+ weight
- Same large "everything else" cluster with <5% weight
- Same singleton strategies isolated
- 2025 adds Wind Forecast Error to the ML regression cluster (it was in the
  broad cluster in 2024)
- Cluster membership is 97% stable across years

## Implications for Later Phases

1. **For Phase 10f (Strategy Robustness)**: The broad cluster (49 strategies)
   is essentially dead weight — they consume computation but contribute <5%
   weight. Most are candidates for removal or redesign.

2. **For Phase 10g (Stronger Strategy Design)**: New strategies should either
   join the ML regression cluster (by being accurate) or provide genuinely
   orthogonal forecasts that survive profit-weighting. Adding more strategies
   to the broad cluster is futile.

3. **For Phase 10e (Sentinel Case Studies)**: High-volatility periods are
   the most interesting — that's where the ML cluster's advantage is largest
   and where cluster-switching (if any) would be most informative.

## Outputs

- `scripts/phase10d_cluster_analysis.py` — full clustering pipeline
- `data/results/phase10/strategy_clusters.csv` — per-strategy cluster assignments (134 rows)
- `data/results/phase10/cluster_dominance.csv` — per-cluster dominance by iteration phase
- `data/results/phase10/regime_forecast_accuracy.csv` — per-cluster, per-regime accuracy
- `tests/backtest/test_cluster_analysis.py` — 11 unit tests

## Checklist

- [x] Build forecast-similarity clustering:
  - pairwise correlation, hierarchical clustering into 8 families
- [x] Build profit-similarity clustering:
  - pairwise correlation, compare to forecast clusters (they disagree)
- [x] Compare cluster patterns by year:
  - ML regression cluster dominates in both years (90%+ weight)
  - Broad cluster (49 strategies) contributes <5% weight
- [x] Identify regime-dependent behaviour:
  - Low-vol vs high-vol split; ML cluster more robust to volatility
- [x] Write `scripts/phase10d_cluster_analysis.py`
- [x] Save cluster assignments to `data/results/phase10/strategy_clusters.csv`
- [ ] ~~Produce cluster dendrogram and heatmap visualisations~~ (deferred —
  text-based analysis provides sufficient insight; visualisation can be
  added to the dashboard in a future phase)
- [ ] ~~Identify sentinel cluster-switching episodes~~ (deferred — no meaningful
  cluster switching detected; the ML cluster dominates throughout)
