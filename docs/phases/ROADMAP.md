# Energy Modelling Platform -- Phase Roadmap

## Overall Status: PHASE 0 IN PROGRESS

## Phase Overview

| Phase | Title | Status | Depends On | Key Deliverable |
|-------|-------|--------|------------|-----------------|
| 0 | [Codebase Consolidation](phase_0_consolidation.md) | NOT STARTED | - | Single framework, green tests |
| 1 | [EDA Audit](phase_1_eda_audit.md) | NOT STARTED | Phase 0 | Audit report, gap priority list |
| 2 | [Deepen EDA](phase_2_deepen_eda.md) | NOT STARTED | Phase 1 | New EDA sections with trading-relevant analysis |
| 3 | [Hypothesis Checkpoints](phase_3_hypothesis_checkpoints.md) | NOT STARTED | Phase 2 | Ranked list of testable trading hypotheses |
| 4 | [Implement Strategies](phase_4_implement_strategies.md) | NOT STARTED | Phase 0, 3 | New ChallengeStrategy implementations |
| 5 | [Run & Assess](phase_5_run_and_assess.md) | NOT STARTED | Phase 4 | Performance results, strategy assessment |
| 6 | [EDA Feedback Loop](phase_6_feedback_loop.md) | NOT STARTED | Phase 5 | Deeper conditional analysis |
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

## Architecture (Post Phase 0)

```
strategies/                          # Student strategies (ChallengeStrategy subclasses)
  __init__.py
  always_long.py
  always_short.py
  [new strategies from Phase 4]

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
| | | |
