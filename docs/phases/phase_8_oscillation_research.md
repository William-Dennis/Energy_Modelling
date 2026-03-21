# Phase 8: Market Oscillation Research

## Status: COMPLETE ✅

## WARNING — Historical Document (2026-03-21)

> **The Phase 8 winner (`running_avg_k=5`) was never implemented in
> production.** The current production engine uses **EMA dampening**
> (`ema_alpha=0.1` by default) instead. There is no `running_avg_k`
> parameter anywhere in the current codebase.
>
> Specifically:
>
> - The "Implementation" section below originally
>   claimed `running_avg_k=5` was the production default. **This is
>   false.** The engine was later reworked to use `ema_alpha` EMA
>   dampening as the convergence mechanism.
> - Phase 9 (EMA Price-Update Experiments) documents the actual dampening
>   approach that replaced the running-average idea.
> - The saved artifacts (`market_2024.pkl`, `market_2025.pkl`) were
>   generated with the **EMA-damped** engine, not the `running_avg_k`
>   configuration documented here.
> - The experiment results and analysis in this document remain valid as
>   historical research — they correctly show that iteration-level
>   smoothing solves oscillation. The production team chose a different
>   smoothing mechanism (EMA) that achieves the same goal.
>
> **This document is preserved for historical clarity.** It should not be
> read as a description of the current production engine.

### Reading guide

- Phase 7 documents the undampened convergence theorems.
- Phase 9 documents the EMA price-update experiments that replaced this
  phase's `running_avg_k` approach.
- Phase 10 is the forward-looking reconciliation and market robustness
  phase.

## Objective

Systematically investigate and resolve the non-convergence (oscillation) of the
spec-compliant synthetic futures market engine.  Both 2024 and 2025 market runs
fail to converge within 20 iterations, entering stable limit cycles with deltas
of 81.4 EUR/MWh (2024) and 42.5 EUR/MWh (2025) — four orders of magnitude above
the 0.01 EUR/MWh convergence threshold.

This phase is structured as a scientific research programme with 6 sub-phases:
problem characterisation, four independent investigation tracks, and a unified
evaluation framework.

## Result Summary

## Historical-Record Note

The tables and winner selection below should be interpreted as the recorded
outcome of the Phase 8 experiment set, not as a guarantee that the same method,
defaults, or artifact contents still represent the live system state today.

Phase 9 documents the EMA experiments that replaced this approach. Phase 10 is
responsible for the broader reconciliation and forward-looking market robustness
work.

**Winner: E1 Running Average K=5** (Track 8e, Iteration-Level Smoothing)

| Metric | 2024 | 2025 |
|--------|------|------|
| Converged | **Yes** | **Yes** |
| Final delta (EUR/MWh) | **0.0034** | **0.0037** |
| Iterations | **32** | **40** |
| Final MAE (EUR/MWh) | **10.85** | **9.29** |
| Iter-0 MAE (EUR/MWh) | 15.10 | N/A |
| Prev-day baseline MAE | 22.02 | N/A |

All 4 success criteria met:
1. ✅ Converges on both 2024 and 2025 (delta < 0.01)
2. ✅ MAE 10.85 ≤ 15.10 (beats iter-0 baseline by 4.25 EUR/MWh)
3. ✅ Iterations ≤ 50 (32 and 40)
4. ✅ Principled: running average smooths the 3-step limit cycle

## Implementation

> **WARNING: The implementation described below was never deployed.**
>
> The `running_avg_k` parameter does not exist in the current codebase.
> The production engine uses `ema_alpha=0.1` (EMA dampening) instead.
> See Phase 9 for the actual production dampening mechanism.

~~The winning configuration is now the production default in:~~
- ~~`futures_market_engine.py` — `run_futures_market(..., running_avg_k=None)`~~
- ~~`futures_market_runner.py` — `run_futures_market_evaluation(..., running_avg_k=5)`~~
- ~~`data/results/market_2024.pkl` and `market_2025.pkl` regenerated with this config~~

**Actual production state (2026-03-21):**
- `futures_market_engine.py` — `run_futures_market(..., ema_alpha=0.1)` is the default
- `futures_market_runner.py` — `run_futures_market_evaluation(..., ema_alpha=0.1)` is the default
- There is no `running_avg_k` parameter in either file

## Prerequisites

- Phase 7 complete (spec-compliant engine, convergence theorems) ✅
- Market pkl files generated with undampened engine ✅
- Oscillation root-cause analysis complete ✅

## Sub-Phase Structure

```
Phase 8: Market Oscillation Research
│
├── 8a. Problem Characterisation        (what is happening and why)
│
├── 8b. Dampening Mechanisms            (modify the iteration update rule)
│   ├── B1: Fixed dampening alpha sweep
│   ├── B2: Adaptive dampening
│   └── B3: Two-phase convergence
│
├── 8c. Weighting Reforms               (modify profit-to-weight mapping)
│   ├── C1: Per-strategy weight cap
│   ├── C2: Weighted median price
│   ├── C3: Log-profit weighting
│   └── C4: Cluster-aware averaging
│
├── 8d. Initialisation Strategies       (modify the starting point)
│   ├── D1: Rolling mean initial price
│   ├── D2: Forecast mean initial price
│   ├── D3: Forecast clipping / anchoring
│   └── D4: Percentile initial price
│
├── 8e. Iteration-Level Smoothing       (post-process iteration history)
│   ├── E1: Running average of K iterations  ← WINNER
│   ├── E2: Exponential moving average
│   ├── E3: Best-iteration selection
│   └── E4: Delta-weighted average
│
└── 8f. Evaluation Framework            (metrics, baselines, comparison)
    ├── Experiment registry (15 individual + 5 combination)
    ├── Baseline definitions
    └── Success criteria
```

## Checklist

### 8a. Problem Characterisation
- [x] Document oscillation structure (3-step limit cycle in 2024)
- [x] Quantify profit magnitudes at each cycle phase
- [x] Identify worst oscillation days and root causes
- [x] Formal characterisation as limit cycle of discrete dynamical system
- [x] Identify 5 interacting root causes

### 8b. Dampening Mechanisms
- [x] B1: Fixed dampening sweep (alpha = 0.1 to 0.9) — ran, none converged
- [ ] B2: Adaptive dampening (not needed — winner found in 8e)
- [ ] B3: Two-phase convergence (not needed — winner found in 8e)
- [x] Document spec-compatibility justification

### 8c. Weighting Reforms
- [x] C2b: Weighted median market price — ran, did not converge
- [x] C2: Log-profit weighting — ran, did not converge
- [x] C3: Weight cap sweep (0.10–0.50) — ran, none converged
- [ ] C4: Cluster-aware averaging (not needed)

### 8d. Initialisation Strategies
- [x] D2: Forecast mean initial price — ran, did not converge

### 8e. Iteration-Level Smoothing
- [x] E1: Running average K=2 — converged, MAE 12.78 (2024)
- [x] E1: Running average K=3 — converged, MAE 11.43 (2024)
- [x] E1: Running average K=5 — converged, MAE 10.85 (2024) ← **WINNER**
- [x] E2: EMA beta sweep — ran, none converged

### 8f. Evaluation and Synthesis
- [x] Build unified evaluation script (`scripts/phase8_experiments.py`)
- [x] Run all individual experiments (32 total across B/C/D/E tracks)
- [x] Run 5 combination experiments (X1–X5)
- [x] Produce comparison table (`data/results/phase8/results.csv`)
- [x] Select recommended approach: **E1 running average K=5**
- [x] Apply winner to engine and regenerate pkl files

---

## Full Experiment Results (2024)

### Converged experiments

| ID | Label | MAE | Iters | Delta |
|----|-------|-----|-------|-------|
| E1_ravg5 | Running-avg K=5 | **10.85** | 32 | 0.0034 |
| E1_ravg3 | Running-avg K=3 | 11.43 | 15 | 0.0026 |
| E1_ravg2 | Running-avg K=2 | 12.78 | 14 | 0.0017 |

### Best non-converged (50 iters, 2024)

| ID | MAE | Delta |
|----|-----|-------|
| B1_a03 (alpha=0.3) | 10.26 | 5.97 |
| B1_a02 (alpha=0.2) | 10.29 | 4.41 |
| E2_ema07 (EMA beta=0.7) | 10.26 | 5.97 |
| Baseline (spec) | 19.39 | 81.42 |

---

## Key Numbers

| Metric | Before (spec) | After (E1_ravg5) |
|--------|---------------|------------------|
| 2024 converged | No | **Yes** |
| 2024 final delta | 81.42 EUR/MWh | **0.0034 EUR/MWh** |
| 2024 iterations | 20 (limit) | **32** |
| 2024 MAE | 19.39 EUR/MWh | **10.85 EUR/MWh** |
| 2025 converged | No | **Yes** |
| 2025 final delta | 42.49 EUR/MWh | **0.0037 EUR/MWh** |
| 2025 iterations | 20 (limit) | **40** |
| 2025 MAE | N/A | **9.29 EUR/MWh** |

---

## Sub-Phase Documents

- [8a: Problem Characterisation](phase_8_oscillation_research/phase_8a_problem_characterisation.md)
- [8b: Dampening Mechanisms](phase_8_oscillation_research/phase_8b_dampening_mechanisms.md)
- [8c: Weighting Reforms](phase_8_oscillation_research/phase_8c_weighting_reforms.md)
- [8d: Initialisation Strategies](phase_8_oscillation_research/phase_8d_initialisation_strategies.md)
- [8e: Iteration-Level Smoothing](phase_8_oscillation_research/phase_8e_iteration_smoothing.md)
- [8f: Evaluation Framework](phase_8_oscillation_research/phase_8f_evaluation_framework.md)
