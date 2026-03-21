# Phase 9a: Baseline Reconciliation

## Status: STARTED

## Objective

Establish the canonical current baseline for the synthetic futures market by
reconciling live code, current defaults, saved artifacts, and historical Phase
7-8 documentation.

This sub-phase exists to answer one foundational question before all others:

**What system are we actually trying to explain?**

## Motivation

Phase 7 and Phase 8 remain valuable, but they describe historical analytical and
experimental states. The current codebase and the saved market artifacts now
need a dedicated reconciliation layer so later research does not accidentally
mix historical claims with current behaviour.

## Canonical Sources of Truth

For Phase 9, the current baseline should be defined from the following sources
in this priority order:

1. Live source code in:
   - `src/energy_modelling/backtest/futures_market_engine.py`
   - `src/energy_modelling/backtest/futures_market_runner.py`
2. Current saved artifacts:
   - `data/results/market_2024.pkl`
   - `data/results/market_2025.pkl`
3. Dashboard assumptions and displays:
   - `src/energy_modelling/dashboard/_futures_market.py`
   - `src/energy_modelling/dashboard/_accuracy.py`
4. Historical context:
   - `docs/phases/phase_7_convergence_analysis.md`
   - `docs/phases/phase_8_oscillation_research.md`

## Initial Verified Baseline (2026-03-21)

### Live engine / runner defaults

- `run_futures_market(..., ema_alpha=0.1)` is present in
  `src/energy_modelling/backtest/futures_market_engine.py`.
- `run_futures_market_evaluation(..., ema_alpha=0.1)` is present in
  `src/energy_modelling/backtest/futures_market_runner.py`.
- The current live production path therefore includes EMA dampening by default.

### Saved artifact summary

| Year | Converged | Iterations | Final Delta | Active First 5 | Active Last 5 |
|------|-----------|------------|-------------|----------------|---------------|
| 2024 | No | 500 | 1.7729 | 60, 40, 40, 38, 38 | 5, 7, 7, 5, 8 |
| 2025 | Yes | 327 | ~0.0 | 60, 39, 39, 37, 35 | 7, 2, 4, 2, 0 |

### Immediate implications

- 2024 is not currently converged in the saved artifact despite substantial
  dampening and a long iteration budget.
- 2025 does converge, but only after the active strategy set collapses to zero.
- This suggests that at least one important current behaviour class is
  **absorbing carry-forward convergence**, not just market consensus convergence.
- Active-strategy collapse must be treated as a primary explanatory target, not
  a side effect.

## Reconciliation Targets

### Phase 7

- Preserve the theorem/proof record for the undampened analytical model.
- Add a scope note clarifying that current production defaults differ.
- Separate "theoretical truth under the undampened model" from
  "current empirical artifact behaviour".

### Phase 8

- Preserve the historical experiment record and winner narrative.
- Mark implementation-specific claims as historical where they no longer match
  the current code signatures.
- Treat Phase 8 as evidence and context, not as an automatically current system
  description.

## Working Questions

1. Which behaviour claims in Phases 7 and 8 remain directly true of the live
   engine and artifact set?
2. Which claims are historically true but no longer operationally current?
3. Which current behaviours are absent from the historical narrative?
4. Which settings should be treated as the canonical baseline for all Phase 9
   experiments?

## Deliverables

- A short baseline memo for the current engine and artifact set
- A table mapping historical claims to current status:
  - still current
  - historical-only
  - needs re-validation
- Additive note blocks in the relevant historical docs

## Checklist

- [ ] Verify current live engine defaults from source
- [ ] Verify current live runner defaults from source
- [ ] Summarise saved 2024 artifact behaviour
- [ ] Summarise saved 2025 artifact behaviour
- [ ] Catalogue doc/code/result mismatches
- [ ] Freeze the canonical baseline configuration for Phase 9
