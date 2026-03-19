# Energy Modelling Platform -- Phase Roadmap

## Overall Status: PHASE 6 COMPLETE — PHASE 7 NEXT

## Phase Overview

| Phase | Title | Status | Depends On | Key Deliverable |
|-------|-------|--------|------------|-----------------|
| 0 | [Codebase Consolidation](phase_0_consolidation.md) | COMPLETE | - | Single framework, green tests |
| 1 | [EDA Audit](phase_1_eda_audit.md) | COMPLETE | Phase 0 | Audit report, gap priority list |
| 2 | [Deepen EDA](phase_2_deepen_eda.md) | COMPLETE | Phase 1 | New EDA sections with trading-relevant analysis |
| 3 | [Hypothesis Checkpoints](phase_3_hypothesis_checkpoints.md) | COMPLETE | Phase 2 | Ranked list of testable trading hypotheses |
| 4 | [Implement Strategies](phase_4_implement_strategies.md) | COMPLETE | Phase 0, 3 | 7 new ChallengeStrategy implementations + 70 tests |
| 5 | [Run & Assess](phase_5_run_and_assess.md) | COMPLETE | Phase 4 | Leaderboard, market sim, hypothesis assessment |
| 6 | [EDA Feedback Loop](phase_6_feedback_loop.md) | COMPLETE | Phase 5 | 6 new dashboard sections, no strategy refinements needed |
| 7 | [Convergence Analysis](phase_7_convergence_analysis.md) | NOT STARTED | Phase 4, 5 | Theoretical + empirical convergence proof |

## Dependency Graph

```
Phase 0 (Consolidation)
  |
  +---> Phase 1 (EDA Audit)
  |       |
  |       +---> Phase 2 (Deepen EDA)
  |               |
  |               +---> Phase 3 (Hypotheses)
  |                       |
  +---> Phase 4 (Strategies) <--- Phase 3
          |
          +---> Phase 5 (Run & Assess)
          |       |
          |       +---> Phase 6 (Feedback Loop)
          |       |
          +---> Phase 7 (Convergence Analysis)
```

## Principles

- **TDD**: Write tests first, then implement
- **DRY**: No duplicated logic; shared functions extracted
- **YAGNI**: Only build what's needed for the current phase
- **Tight loops**: Code -> test -> verify -> update docs at each step
- **Live documents**: Each phase `.md` is updated in real-time as work progresses
- **GIT commits**: Frequent, descriptive commits tied to specific phase goals to maintain a clear history and revertibility

## Architecture (Post Phase 0)

```
strategies/                          # Student strategies (ChallengeStrategy subclasses)
  __init__.py
  always_long.py
  always_short.py
  day_of_week.py           # H1: Calendar effect
  wind_forecast.py         # H2: Wind forecast contrarian
  load_forecast.py         # H3: Load forecast level
  lag2_reversion.py        # H4: Lag-2 mean reversion
  weekly_cycle.py          # H5: Weekly cycle exploitation
  fossil_dispatch.py       # H6: Fossil dispatch contrarian
  composite_signal.py      # H7: Weighted z-score composite

src/energy_modelling/
  challenge/                         # THE strategy framework
    types.py                         #   ChallengeStrategy ABC, ChallengeState
    runner.py                        #   run_challenge_backtest()
    scoring.py                       #   metrics, leaderboard, monthly_pnl, rolling_sharpe
    market.py                        #   Synthetic futures market engine
    market_runner.py                 #   Market evaluation orchestrator
    data.py                          #   Daily challenge data builder
    __init__.py

  market_simulation/                 # Shared data utilities (kept)
    data.py                          #   load_dataset, build_daily_features, compute_daily_settlement
    contract.py                      #   compute_pnl, compute_settlement_price
    types.py                         #   DayState, Signal, Trade, Settlement (retained for data.py)
    __init__.py

  dashboard/                         # Streamlit dashboard (4 tabs post-consolidation)
    app.py                           #   Thin orchestrator
    __init__.py                      #   Shared helpers
    _eda.py                          #   Tab 1: EDA
    _challenge.py                    #   Tab 2: Challenge leaderboard
    _market.py                       #   Tab 3: Futures market
    _accuracy.py                     #   Tab 4: Market price accuracy

  data_collection/                   # ENTSO-E + weather + commodity data collection
```

## Change Log

| Date | Phase | Change |
|------|-------|--------|
| 2026-03-19 | 0 | COMPLETE — 150 tests pass, single framework, all stale refs cleaned |
| 2026-03-19 | 1 | COMPLETE — 12 sections audited, 10 gaps ranked by trading relevance |
| 2026-03-19 | 2 | COMPLETE — 6 new trading-focused EDA sections, 24 tests, 10 pure functions |
| 2026-03-19 | 3 | COMPLETE — 7 hypotheses (H1-H7) with formal specs, real data findings |
| 2026-03-19 | 4 | COMPLETE — 7 strategies implemented (H1-H7), 70 new tests, 215 total passing |
| 2026-03-19 | 5 | COMPLETE — All 7 hypotheses HELD; market sim non-convergence discovered; reset() bug fixed |
| 2026-03-19 | 6 | COMPLETE — 5 new analysis functions, 14 new tests, 6 new dashboard sections (19-24); no strategy refinements warranted |
