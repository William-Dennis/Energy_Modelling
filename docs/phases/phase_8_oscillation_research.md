# Phase 8: Market Oscillation Research

## Status: IN PROGRESS

## Objective

Systematically investigate and resolve the non-convergence (oscillation) of the
spec-compliant synthetic futures market engine.  Both 2024 and 2025 market runs
fail to converge within 20 iterations, entering stable limit cycles with deltas
of 81.4 EUR/MWh (2024) and 42.5 EUR/MWh (2025) — four orders of magnitude above
the 0.01 EUR/MWh convergence threshold.

This phase is structured as a scientific research programme with 6 sub-phases:
problem characterisation, four independent investigation tracks, and a unified
evaluation framework.

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
│   ├── E1: Running average of K iterations
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
- [x] B1: Fixed dampening sweep (alpha = 0.1 to 1.0)
- [x] B2: Adaptive dampening (proportional control)
- [x] B3: Two-phase convergence (dampened warm-start, then undampened refinement)
- [x] Document spec-compatibility justification

### 8c. Weighting Reforms
- [x] C1: Per-strategy weight cap sweep (w_max = 0.05 to 1.0)
- [x] C2: Weighted median market price
- [x] C3: Log-profit weighting
- [x] C4: Cluster-aware cross-pole averaging

### 8d. Initialisation Strategies
- [ ] D1: Rolling mean initial price (window = 1 to 20)
- [ ] D2: Forecast mean initial price (equal-weighted strategy mean)
- [ ] D3: Forecast clipping / anchoring (max_deviation sweep)
- [ ] D4: Percentile initial price (window + percentile)

### 8e. Iteration-Level Smoothing
- [x] E1: Running average of last K iterations
- [x] E2: Exponential moving average across iterations
- [x] E3: Best-iteration selection (lowest delta)
- [x] E4: Delta-weighted iteration average

### 8f. Evaluation and Synthesis
- [ ] Build unified evaluation script
- [ ] Run all 15 individual experiments
- [ ] Run 5 combination experiments
- [ ] Produce comparison table
- [ ] Select recommended approach
- [ ] Write final results document

---

## Key Numbers (from 8a Analysis)

| Metric | 2024 | 2025 |
|--------|------|------|
| Converged | No | No |
| Final delta (EUR/MWh) | 81.42 | 42.49 |
| Iterations | 20 (max) | 20 (max) |
| Strategies | 67 | 67 |
| Cycle period | 3 iterations | 3 iterations |
| Final MAE vs real | 19.39 | N/A |
| Iter-0 MAE vs real | 15.10 | N/A |
| Prev-day baseline MAE | 22.02 | N/A |

The critical target: any solution must achieve MAE <= 15.10 on 2024 (no worse
than simply running one iteration and stopping).

---

## Research Tracks at a Glance

| Track | Mechanism | Spec Impact | Confidence | Addresses Root Cause |
|-------|-----------|-------------|------------|---------------------|
| 8b Dampening | Reduce loop gain | Low (solver technique) | High | #1 (no dampening) |
| 8c Weighting | Reduce weight sensitivity | Medium (changes Step 3) | Medium | #3 (winner-take-all) |
| 8d Init | Better starting point | None (init only) | Medium | #5 (last_settlement anchor) |
| 8e Smoothing | Post-process iteration trace | None (post-processing) | Medium | Symptom treatment |

**Recommended investigation order**: 8b (most likely to work alone) → 8c (complementary) → 8d + 8e (additive improvements) → combinations.

---

## Sub-Phase Documents

- [8a: Problem Characterisation](phase_8_oscillation_research/phase_8a_problem_characterisation.md)
- [8b: Dampening Mechanisms](phase_8_oscillation_research/phase_8b_dampening_mechanisms.md)
- [8c: Weighting Reforms](phase_8_oscillation_research/phase_8c_weighting_reforms.md)
- [8d: Initialisation Strategies](phase_8_oscillation_research/phase_8d_initialisation_strategies.md)
- [8e: Iteration-Level Smoothing](phase_8_oscillation_research/phase_8e_iteration_smoothing.md)
- [8f: Evaluation Framework](phase_8_oscillation_research/phase_8f_evaluation_framework.md)
