# Phase 10: Futures Market Behaviour and Strategy Robustness

## Status: IN PROGRESS (10a-10d complete, 10e-10h planned)

## Objective

Explain the observed behaviour of the iterative synthetic futures market under the
current implementation, reconcile that behaviour with the historical Phase 7-8
research record, and use the resulting insights to design stronger, more
market-robust strategies.

This phase has two linked goals:

1. **Scientific explanation** -- understand why the market converges, fails to
   converge, oscillates, stalls, or collapses into low-activity states under
   different data regimes.
2. **Strategy improvement** -- use those explanations to strengthen the strategy
   pool so that more strategies remain informative after equilibrium repricing,
   improve market price quality, and avoid redundant or destabilising behaviour.

## Why Phase 10 Is Needed

The project now has three distinct layers of truth that need to be reconciled:

- **Historical theory** from Phase 7, which focuses on the undampened,
  spec-compliant engine and perfect-foresight analysis.
- **Historical remedy research** from Phase 8, which investigated oscillation
  under earlier saved results and selected a practical mitigation path.
- **Current live implementation and artifacts**, where the engine and saved
  market result files exhibit behaviour that is no longer fully described by the
  existing Phase 7-8 narrative.

Without an explicit reconciliation phase, the project risks drawing conclusions
from stale assumptions, over-explaining historical behaviour, or optimising
strategies for the wrong market dynamics.

## Current-State Motivation

Initial inspection shows that:

- `src/energy_modelling/backtest/futures_market_engine.py` currently exposes
  `ema_alpha` dampening in the production engine.
- `src/energy_modelling/backtest/futures_market_runner.py` also defaults to
  `ema_alpha=0.1`.
- The saved market artifacts in `data/results/market_2024.pkl` and
  `data/results/market_2025.pkl` do not align cleanly with all claims in the
  existing Phase 8 summary.
- A new explanatory layer is needed before doing more engine changes or
  evaluating new strategies solely on standalone backtest performance.

## Initial Baseline Snapshot (2026-03-21)

The first live-artifact inspection for Phase 10 shows:

| Year | Converged | Iterations | Final Delta | Late Active Strategies |
|------|-----------|------------|-------------|------------------------|
| 2024 | No | 500 | 1.7729 | 5-8 in the final iterations |
| 2025 | Yes | 327 | ~0.0 | 0 by the final iteration |

Additional baseline observations:

- 2024 appears to be **slowly damped but not converged** by the current 500
  iteration cap.
- 2025 **does converge**, but does so while the active strategy set collapses
  toward zero, suggesting a possible absorbing carry-forward state.
- Late-iteration active sets are much smaller than early-iteration active sets
  in both years, making active-strategy collapse a first-class research target.
- The current saved artifacts therefore present a richer behaviour set than the
  simpler historical split of "oscillates" vs "converges".

## First Confirmed Reconciliation Findings

Phase 10a has already confirmed several concrete mismatches between the
historical Phase 7-8 narrative and the live current-state baseline:

- Phase 8 still contains implementation text describing `running_avg_k`-based
  production defaults, while the live engine and runner currently expose
  `ema_alpha=0.1` defaults.
- The historical Phase 8 summary presents a converged 2024 result, but the
  current saved `data/results/market_2024.pkl` artifact is not converged after
  500 iterations.
- The current saved `data/results/market_2025.pkl` artifact does converge, but
  does so with the active strategy set collapsing to zero at the final step,
  introducing an absorbing-state interpretation that is not foregrounded in the
  historical narrative.

These findings confirm that Phase 10 should treat live code plus current saved
artifacts as canonical and should treat Phases 7-8 as historical context unless
specific claims are re-validated.

## Current Verification Status

- All 965 tests pass via `uv run pytest` (948 original + 17 new Phase 10b tests).
- `scripts/verify_theorems.py` exits with ALL THEOREMS VERIFIED.
- `uv run ruff check .` reports zero warnings.

## Prerequisites

- Phase 5 complete (baseline market simulation and strategy assessment) ✅
- Phase 7 complete (formal convergence analysis for the undampened model) ✅
- Phase 8 complete as historical oscillation research record ✅
- Phase 9 complete (EMA price update experiments; alpha=0.1 adopted) ✅
- Current market artifacts available in `data/results/market_2024.pkl` and
  `data/results/market_2025.pkl` ✅
- Strategy registry and 67-strategy pool available for analysis ✅

## Sub-Phase Structure

```
Phase 10: Futures Market Behaviour and Strategy Robustness
│
├── 10a. Baseline Reconciliation
│   ├── A1: Canonical current-engine audit
│   ├── A2: Saved-artifact verification
│   └── A3: Additive documentation notes for Phases 7 and 8
│
├── 10b. Behaviour Inventory
│   ├── B1: Convergence / non-convergence catalogue
│   ├── B2: Active-strategy collapse and zero-weight states
│   └── B3: Accuracy-over-iteration trajectories
│
├── 10c. Mechanism Attribution
│   ├── C1: Sign-rule contribution
│   ├── C2: Profit truncation and weight concentration
│   ├── C3: EMA dampening sensitivity
│   └── C4: Initialisation sensitivity
│
├── 10d. Regime and Cluster Analysis
│   ├── D1: Forecast-cluster identification
│   ├── D2: Profit-cluster identification
│   ├── D3: 2024 vs 2025 regime comparison
│   └── D4: Sentinel periods and dominant days
│
├── 10e. Sentinel Case Studies
│   ├── E1: Iteration-by-iteration causal traces
│   ├── E2: Extreme-day reconstructions
│   └── E3: Human-readable market narratives
│
├── 10f. Strategy Robustness Analysis
│   ├── F1: Standalone vs market-adjusted ranking shifts
│   ├── F2: Strategy contribution-to-market metrics
│   └── F3: Destabilising vs stabilising strategy families
│
├── 10g. Stronger Strategy Design
│   ├── G1: Market-aware strategy design principles
│   ├── G2: Candidate new strategies / revisions
│   ├── G3: Acceptance criteria for stronger strategies
│   └── G4: Priority shortlist for implementation
│
└── 10h. Synthesis and Forward Plan
    ├── H1: Unified explanation of observed behaviour
    ├── H2: Implications for engine and dashboard interpretation
    └── H3: Recommended next implementation phases
```

## Research Questions

### Market Behaviour

1. Why does the current market configuration converge on some years or regimes
   but not others?
2. Why can early iterations be more accurate than later ones?
3. Under what conditions does the active strategy set shrink sharply?
4. When and why can the market enter an absorbing all-zero-weight state?
5. Are the dominant dynamics best described as oscillation, damped oscillation,
   regime switching, or piecewise convergence?

### Mechanism Attribution

6. How much of the observed behaviour is driven by the sign-based trading rule?
7. How much is driven by positive-profit truncation and linear weight
   normalisation?
8. How sensitive are outcomes to `ema_alpha`, `max_iterations`, and initial
   prices?
9. Which dates or clusters dominate the aggregate convergence trajectory?

### Strategy Quality

10. Which strategies are strong only in the standalone backtest, and which stay
    useful after market repricing?
11. Which strategy families improve the equilibrium price, and which mostly add
    noise, redundancy, or instability?
12. What properties make a strategy **market-robust** rather than merely
    **backtest-good**?
13. Can new or revised strategies improve both individual performance and the
    quality of the aggregate market price?

## Planned Methods

### 10a. Baseline Reconciliation

- Audit live code paths, defaults, saved artifacts, and dashboard assumptions.
- Verify current canonical market settings from source rather than relying on
  prior documentation.
- Add additive notes to Phase 7 and Phase 8 documenting historical scope,
  current-state divergence, and Phase 10 as the reconciliation follow-up.

### 10b. Behaviour Inventory

- Extract per-iteration metrics: convergence delta, MAE, RMSE, bias, active
  strategy count, weight entropy, top-strategy share.
- Build a compact behaviour inventory for 2024 and 2025.
- Record whether each run exhibits contraction, oscillation, damped oscillation,
  regime switching, or absorbing-state behaviour.

### 10c. Mechanism Attribution

- Run controlled sweeps over `ema_alpha`, initial prices, and strategy subsets.
- Perform ablations on strategy families and extreme dates.
- Quantify how each mechanism changes convergence speed, price quality, and
  weight concentration.

### 10d. Regime and Cluster Analysis

- Cluster strategies by forecast similarity and by profit similarity.
- Compare cluster dominance across low-volatility and high-volatility periods.
- Directly compare 2024 and 2025 to identify which differences are structural
  and which are path-dependent.

### 10e. Sentinel Case Studies

- Select high-information windows and days that dominate the overall dynamics.
- Build iteration-by-iteration traces showing price, winning cluster, weight
  shift, and next-step response.
- Produce a narrative explanation for each sentinel case.

### 10f. Strategy Robustness Analysis

- Rank strategies under both original and market-adjusted PnL.
- Measure contribution to market quality, not just individual returns.
- Flag strategies that are redundant, destabilising, or only win because of the
  pre-market entry-price convention.

### 10g. Stronger Strategy Design

- Derive market-aware design rules from the findings above.
- Prioritise orthogonal signals, balanced long/short exposure, regime-aware
  forecasts, and strategies that remain informative after repricing.
- Connect this work to the strategy backlog in `issues/issue_3_new_strategies.md`
  and the live registry in `docs/expansion/strategy_registry.md`.

## Deliverables

- `docs/phases/phase_10_market_behaviour_and_strategy_robustness.md` -- master
  plan and live record for the phase
- Additive current-state notes in:
  - `docs/phases/phase_7_convergence_analysis.md`
  - `docs/phases/phase_8_oscillation_research.md`
- A reproducible results table covering current-engine market behaviour by year
- A strategy robustness comparison table covering standalone, market-adjusted,
  and market-contribution metrics
- A shortlist of high-priority stronger strategy candidates

## Sub-Phase Documents

- [10a: Baseline Reconciliation](phase_10_market_behaviour_and_strategy_robustness/phase_10a_baseline_reconciliation.md)
- [10b: Behaviour Inventory](phase_10_market_behaviour_and_strategy_robustness/phase_10b_behaviour_inventory.md)
- [10c: Mechanism Attribution](phase_10_market_behaviour_and_strategy_robustness/phase_10c_mechanism_attribution.md)
- [10d: Regime and Cluster Analysis](phase_10_market_behaviour_and_strategy_robustness/phase_10d_regime_and_cluster_analysis.md)
- [10e: Sentinel Case Studies](phase_10_market_behaviour_and_strategy_robustness/phase_10e_sentinel_case_studies.md)
- [10f: Strategy Robustness Analysis](phase_10_market_behaviour_and_strategy_robustness/phase_10f_strategy_robustness_analysis.md)
- [10g: Stronger Strategy Design](phase_10_market_behaviour_and_strategy_robustness/phase_10g_stronger_strategy_design.md)
- [10h: Synthesis and Forward Plan](phase_10_market_behaviour_and_strategy_robustness/phase_10h_synthesis_and_forward_plan.md)

## Checklist

### 10a. Baseline Reconciliation
- [x] Audit current engine defaults and convergence logic
- [x] Verify saved `market_2024.pkl` and `market_2025.pkl` behaviour summaries
- [x] Add additive current-state notes to Phase 7 and Phase 8 docs
- [x] Define the canonical baseline configuration for all Phase 10 analysis

### 10b. Behaviour Inventory
- [x] Compute per-iteration metric panel for 2024
- [x] Compute per-iteration metric panel for 2025
- [x] Classify run-level behaviour modes
- [x] Identify top-priority behaviours that require explanation

### 10c. Mechanism Attribution
- [x] Sweep `ema_alpha` under the current strategy pool
- [x] Compare alternative initialisation choices
- [x] Run cluster / strategy-family ablations
- [x] Test sensitivity to extreme-day removal and sentinel-period masking

### 10d. Regime and Cluster Analysis
- [x] Cluster strategies by forecast similarity
- [x] Cluster strategies by profit similarity
- [x] Quantify cluster dominance by regime and by iteration
- [x] Compare 2024 and 2025 drivers directly

### 10e. Sentinel Case Studies
- [ ] Select sentinel days and multi-day windows
- [ ] Build iteration-level traces for each case
- [ ] Write plain-language explanations for each case

### 10f. Strategy Robustness Analysis
- [ ] Produce standalone vs market-adjusted leaderboard comparison
- [ ] Define and compute market-contribution metrics
- [ ] Identify robust, redundant, and destabilising strategies

### 10g. Stronger Strategy Design
- [ ] Define acceptance criteria for a stronger strategy
- [ ] Produce a shortlist of strategy revisions / additions
- [ ] Prioritise candidates for implementation in the next phase

### 10h. Synthesis
- [ ] Write final explanation of current market behaviour
- [ ] Summarise implications for engine interpretation and strategy development
- [ ] Recommend the next implementation phase

## Working Definitions

### Stronger Strategy

For Phase 10, a stronger strategy is not merely one with strong standalone PnL.
It should satisfy most of the following:

- remains profitable or useful after market repricing,
- contributes non-redundant information to the market,
- improves or at least does not materially degrade aggregate market price
  accuracy,
- avoids pathological concentration or avoidable instability,
- has a defensible economic or statistical rationale.

### Market Contribution

The phase should treat a strategy as contributing to the market if its presence
improves one or more of:

- final market MAE / RMSE,
- convergence behaviour,
- robustness across regimes,
- diversity of informative forecasts.

## Success Criteria

Phase 10 is successful if:

1. The current live engine, saved artifacts, and Phase 7-8 history are
   reconciled into one coherent narrative.
2. Each major observed market behaviour has a concrete mechanism-backed
   explanation.
3. The project gains a reliable definition of strategy robustness inside the
   synthetic market, not just in standalone backtests.
4. The phase produces a justified shortlist of strategy improvements or new
   strategy candidates.
5. The resulting documentation reduces ambiguity about what is historical,
   what is current, and what should be treated as canonical going forward.

## Files Likely To Be Used

- `src/energy_modelling/backtest/futures_market_engine.py`
- `src/energy_modelling/backtest/futures_market_runner.py`
- `src/energy_modelling/backtest/convergence.py`
- `docs/phases/phase_7_convergence_analysis.md`
- `docs/phases/phase_8_oscillation_research.md`
- `docs/expansion/strategy_registry.md`
- `issues/issue_3_new_strategies.md`
- `data/results/market_2024.pkl`
- `data/results/market_2025.pkl`
