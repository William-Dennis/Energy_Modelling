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

## Initial Mismatch Table

The following mismatches are already confirmed and should be treated as the
starting reconciliation set for Phase 9a:

| Topic | Historical claim | Current live state | Status |
|------|-------------------|--------------------|--------|
| Production default mechanism | Phase 8 implementation section references `running_avg_k`-based production defaults | Live engine / runner expose `ema_alpha=0.1` defaults | confirmed mismatch |
| 2024 saved result interpretation | Phase 8 summary presents a converged 2024 outcome | Current `market_2024.pkl` is not converged after 500 iterations | confirmed mismatch |
| 2025 convergence interpretation | Phase 8 summary presents converged 2025 via winning smoothing config | Current `market_2025.pkl` converges, but with zero active strategies at the end | partial mismatch / needs interpretation |
| Phase 7 empirical scope | Phase 7 empirical language can read as generally current | Phase 7 theory remains valid, but current production defaults are damped | scope clarification needed |

## Canonical Baseline Configuration

Unless explicitly stated otherwise, Phase 9 should treat the following as the
canonical baseline configuration:

- engine: `run_futures_market()` from the live codebase
- runner: `run_futures_market_evaluation()` from the live codebase
- default dampening: `ema_alpha=0.1`
- convergence threshold: `0.01`
- iteration cap: `500`
- artifact set:
  - `data/results/market_2024.pkl`
  - `data/results/market_2025.pkl`

Historical Phase 7-8 settings remain important context, but they should not be
treated as default current-state facts unless re-verified.

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

## Verification Notes (2026-03-21)

### Linting

- `python -m ruff check .` was run against the repository.
- Result: fails due to pre-existing repo-wide issues outside the Phase 9 docs,
  including legacy/import-order violations in scripts, simplification warnings,
  long lines, and several existing strategy/dashboard lint findings.
- Phase 9 documentation work did not introduce Python lint errors because it is
  markdown-only, but the repository is not currently lint-clean as a whole.

### Tests

- `pytest -q` was run against the full repository.
- Result: test collection fails in `tests/data_collection/` because the current
  environment does not have `pytest_mock` installed.
- This is an environment/dependency issue rather than a regression caused by the
  Phase 9 documentation work.

### Interpretation

- The Phase 9a documentation changes are complete and the baseline findings are
  recorded.
- Full repo verification remains partially blocked by existing lint debt and the
  missing `pytest_mock` dependency in the current environment.

## Checklist

- [x] Verify current live engine defaults from source
- [x] Verify current live runner defaults from source
- [x] Summarise saved 2024 artifact behaviour
- [x] Summarise saved 2025 artifact behaviour
- [x] Catalogue doc/code/result mismatches
- [x] Freeze the canonical baseline configuration for Phase 9
- [x] Record current lint / test verification status
