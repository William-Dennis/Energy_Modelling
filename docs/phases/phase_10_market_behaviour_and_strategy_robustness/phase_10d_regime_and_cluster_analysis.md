# Phase 10d: Regime and Cluster Analysis

## Status: PLANNED

## Objective

Explain market behaviour through the structure of the strategy pool: forecast
clusters, profit clusters, regime dependence, and cross-year differences.

## Motivation

The market does not respond to individual strategies in isolation. It responds
to structured groups of strategies that become profitable under different price
regimes. Phase 10d aims to identify those groups and explain when each group
dominates.

## Planned Analyses

- cluster strategies by forecast similarity over time
- cluster strategies by profit-response similarity over iterations
- compare low-volatility and high-volatility periods
- compare 2024 and 2025 cluster dominance patterns
- identify sentinel windows where cluster leadership changes abruptly

## Checklist

- [ ] Build forecast-similarity clustering:
  - compute pairwise correlation of strategy forecast time series
  - use hierarchical clustering to identify 5-10 forecast families
- [ ] Build profit-similarity clustering:
  - compute pairwise correlation of strategy profit vectors
  - compare to forecast clusters — are they the same groupings?
- [ ] Compare cluster patterns by year:
  - which clusters dominate early iterations vs late iterations?
  - which clusters dominate 2024 vs 2025?
- [ ] Identify regime-dependent behaviour:
  - split data into low-vol and high-vol windows (e.g., rolling 20-day std)
  - compare cluster dominance across regimes
- [ ] Identify the most important cluster-switching episodes:
  - find iterations where the dominant cluster changes abruptly
  - record as sentinel candidates for Phase 10e
- [ ] Produce cluster dendrogram and heatmap visualisations
- [ ] Write `scripts/phase10d_cluster_analysis.py`
- [ ] Save cluster assignments to `data/results/phase10/strategy_clusters.csv`
