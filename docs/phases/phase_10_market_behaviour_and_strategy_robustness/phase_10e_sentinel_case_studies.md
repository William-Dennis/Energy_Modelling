# Phase 10e: Sentinel Case Studies

## Status: PLANNED

## Objective

Produce a small set of iteration-by-iteration case studies that make the market
behaviour understandable in concrete causal terms.

## Purpose

Aggregate metrics alone will not explain the system clearly. This sub-phase is
for narrative-quality evidence: a few cases that show exactly how prices,
profits, weights, and strategy clusters interact.

## Candidate Case Types

- a high-volatility 2024 window with persistent non-convergence
- a 2025 window approaching zero-active convergence
- a day where iter-0 is better than the final iteration
- a day dominated by a single cluster shift

## Checklist

- [ ] Select 3-5 sentinel windows based on Phase 10b/10d findings:
  - at least one high-volatility 2024 window with persistent non-convergence
  - at least one 2025 window approaching zero-active convergence
  - at least one window where iter-0 MAE < final MAE (early accuracy lost)
  - at least one cluster-switching episode from Phase 10d
- [ ] For each sentinel window, build iteration-level trace showing:
  - market price trajectory vs real price
  - active strategy count and top-3 strategy weights
  - convergence delta per iteration
  - which strategy cluster is dominant at each step
- [ ] Write plain-language causal explanation for each case:
  - what triggered the behaviour
  - which mechanism (from Phase 10c) is responsible
  - whether the outcome is desirable or pathological
- [ ] Produce 1-page narrative summary per sentinel case
- [ ] Write `scripts/phase10e_sentinel_traces.py`
- [ ] Save trace data to `data/results/phase10/sentinel_traces/`
