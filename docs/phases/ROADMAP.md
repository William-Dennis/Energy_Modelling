# Energy Modelling Platform -- Phase Roadmap

## Overall Status: PHASES 0-13 COMPLETE

## Phase Overview

| Phase | Title | Status | Depends On | Key Deliverable |
|-------|-------|--------|------------|-----------------|
| 0 | [Codebase Consolidation](phase_0_consolidation.md) | COMPLETE | - | Single framework, green tests |
| 1 | [EDA Audit](phase_1_eda_audit.md) | COMPLETE | Phase 0 | Audit report, gap priority list |
| 2 | [Deepen EDA](phase_2_deepen_eda.md) | COMPLETE | Phase 1 | New EDA sections with trading-relevant analysis |
| 3 | [Hypothesis Checkpoints](phase_3_hypothesis_checkpoints.md) | COMPLETE | Phase 2 | Ranked list of testable trading hypotheses |
| 4 | [Implement Strategies](phase_4_implement_strategies.md) | COMPLETE | Phase 0, 3 | 7 new BacktestStrategy implementations + 70 tests |
| 5 | [Run & Assess](phase_5_run_and_assess.md) | COMPLETE | Phase 4 | Leaderboard, market sim, hypothesis assessment |
| 6 | [EDA Feedback Loop](phase_6_feedback_loop.md) | COMPLETE | Phase 5 | 6 new dashboard sections, no strategy refinements needed |
| 7 | [Convergence Analysis](phase_7_convergence_analysis.md) | COMPLETE | Phase 4, 5 | Forecast-based convergence proven (contraction mapping); 4 theorems validated empirically. **WARNING: Applies to undampened model only (ema_alpha=1.0); not taken forward to production.** |
| 8 | [Oscillation Research](phase_8_oscillation_research.md) | COMPLETE | Phase 7 | Historical oscillation research record. **WARNING: Winner (running_avg_k=5) was never implemented; see Phase 9 for the approach actually adopted.** |
| 9 | [EMA Price Update Experiments](phase_9_ema_price_update.md) | COMPLETE | Phase 7, 8 | EMA dampening sweep; alpha=0.1 adopted as initial production default; 2025 converges, 2024 does not. **NOTE: alpha=0.01 later applied in Phase 13 fix.** |
| 10 | [Futures Market Behaviour and Strategy Robustness](phase_10_market_behaviour_and_strategy_robustness.md) | COMPLETE | Phase 5, 7, 8, 9 | Reconcile current market behaviour, explain dynamics, and design stronger market-robust strategies |
| 11 | [New Strategies and Hyperparameter Tuning](phase_11_new_strategies_and_hyperparameter_tuning.md) | COMPLETE | Phase 10 | 7 new strategies (74 total), hyperparameter recommendations, 44 new tests |
| 12 | [Forecast Cache & Strategy Expansion](phase_12_forecast_cache_and_strategy_expansion.md) | COMPLETE | Phase 11 | SQLite forecast cache (recompute-all in 2.6s), 26 new strategies reaching 100 total, 1279 tests |
| 13 | ema_alpha Production Fix | COMPLETE | Phase 10c, 12 | Apply ema_alpha=0.01 to recompute.py call sites; regenerate market_2024.pkl and market_2025.pkl |

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
          |
          +---> Phase 8 (Oscillation Research) <--- Phase 7
          |
          +---> Phase 9 (EMA Price Update) <--- Phase 7, 8
          |
          +---> Phase 10 (Behaviour + Robust Strategies) <--- Phase 5, 7, 8, 9
          |
          +---> Phase 11 (Engine Hardening + Refinement) <--- Phase 10
          |
          +---> Phase 12 (Forecast Cache + Strategy Expansion) <--- Phase 11
```

## Expansion Phases (Parallel Track)

Strategy expansion from 9 to 67 strategies was tracked under a parallel work
stream in `docs/expansion/`.  This work ran alongside the main Phase 0-9
roadmap and is fully complete.

| Sub-Phase | Title | Status | Key Deliverable |
|-----------|-------|--------|-----------------|
| A | [Feature Engineering](../expansion/phase_A_feature_engineering.md) | COMPLETE | 18 derived features |
| B | [Issue 3 Strategies](../expansion/phase_B_issue3_strategies.md) | COMPLETE | +7 strategies (solar, commodity, temperature, cross-border, volatility, nuclear, renewables) |
| C | [Derived Threshold](../expansion/phase_C_derived_threshold.md) | COMPLETE | +14 derived-feature threshold strategies |
| D | [ML Strategies](../expansion/phase_D_ml_strategies.md) | COMPLETE | +15 ML model strategies |
| E | [Regime Calendar](../expansion/phase_E_regime_calendar.md) | COMPLETE | +8 calendar/temporal/regime strategies |
| F | [Ensemble](../expansion/phase_F_ensemble.md) | COMPLETE | +12 ensemble/meta strategies |
| G | [Feedback Loop](../expansion/phase_G_feedback_loop.md) | COMPLETE | Automated feedback loop infrastructure |

See `docs/expansion/strategy_registry.md` for the full 100-strategy inventory and
`docs/expansion/signal_registry.md` for the signal catalog.

## Principles

- **TDD**: Write tests first, then implement
- **DRY**: No duplicated logic; shared functions extracted
- **YAGNI**: Only build what's needed for the current phase
- **Tight loops**: Code -> test -> verify -> update docs at each step
- **Live documents**: Each phase `.md` is updated in real-time as work progresses
- **GIT commits**: Frequent, descriptive commits tied to specific phase goals to maintain a clear history and revertibility
- **Historical clarity**: When later work supersedes operational defaults or
  saved artifacts, preserve the older phase record and add explicit current-state
  notes rather than silently rewriting history

## Architecture (Current — Post Expansion)

```
strategies/                          # 100 registered strategies + 1 analysis-only
  __init__.py                        #   Registry: imports all 100 strategies
  ml_base.py                         #   _MLStrategyBase (scikit-learn adapter)
  ensemble_base.py                   #   _EnsembleBase (ensemble adapter)
  always_long.py                     #   Baseline: always long
  always_short.py                    #   Baseline: always short
  day_of_week.py                     #   H1: Calendar effect
  wind_forecast.py                   #   H2: Wind forecast contrarian
  load_forecast.py                   #   H3: Load forecast level
  lag2_reversion.py                  #   H4: Lag-2 mean reversion
  weekly_cycle.py                    #   H5: Weekly cycle exploitation
  fossil_dispatch.py                 #   H6: Fossil dispatch contrarian
  composite_signal.py                #   H7: Weighted z-score composite
  perfect_foresight.py               #   Analysis-only (Phase 7, NOT in __init__.py)
  # ... +36 rule-based/fundamental strategies (Phases B-E)
  # ... +18 ML strategies (Phase D)
  # ... +11 ensemble strategies (Phase F)
  # See docs/expansion/strategy_registry.md for full list

src/energy_modelling/
  backtest/                          # THE strategy framework (16 modules)
    types.py                         #   BacktestStrategy ABC, BacktestState
    runner.py                        #   run_backtest()
    scoring.py                       #   metrics, leaderboard, monthly_pnl, rolling_sharpe
    futures_market_engine.py         #   Synthetic futures market engine (ema_alpha dampening)
    futures_market_runner.py         #   Market evaluation orchestrator
    convergence.py                   #   Convergence analysis (Phase 7)
    data.py                          #   Daily backtest data builder
    benchmarks.py                    #   Entry-price benchmark factories (Issue 1)
    walk_forward.py                  #   Walk-forward validation
    io.py                            #   Save/load results (pickle) (Issue 2)
    feedback.py                      #   Strategy feedback & correlation (Phase 6)
    feature_engineering.py           #   Feature engineering pipeline (Expansion Phase A)
    recompute.py                     #   recompute-all CLI (Issue 6)
    forecast_cache.py                #   SQLite per-strategy forecast cache (Phase 12)
    cli.py                           #   build-backtest-data CLI
    __main__.py                      #   CLI entry point
    __init__.py                      #   Re-exports all public API

  futures_market/                    # Shared data utilities (4 modules)
    data.py                          #   load_dataset, build_daily_features, compute_daily_settlement
    contract.py                      #   compute_pnl, compute_settlement_price
    types.py                         #   DayState, Signal, Trade, Settlement
    __init__.py

  dashboard/                         # Streamlit dashboard (~20 modules, 4 tabs)
    app.py                           #   Thin orchestrator
    __init__.py                      #   Shared helpers
    _eda.py                          #   Tab 1: EDA (thin shim)
    _eda_core.py                     #   EDA core rendering logic
    _eda_constants.py                #   EDA constants/config
    _eda_sections_basic.py           #   EDA: basic data sections
    _eda_sections_distributions.py   #   EDA: distribution analysis
    _eda_sections_forecasts.py       #   EDA: forecast analysis
    _eda_sections_market.py          #   EDA: market analysis
    _eda_sections_signals.py         #   EDA: signal extraction
    _eda_sections_trading.py         #   EDA: trading analysis
    _eda_sections_volatility.py      #   EDA: volatility analysis
    _eda_sections_feedback.py        #   EDA: feedback loop analysis (Phase 6)
    eda_analysis.py                  #   Standalone EDA analysis module
    eda_analysis_advanced.py         #   Advanced EDA analysis
    _backtest.py                     #   Tab 2: Backtest leaderboard
    _backtest_render.py              #   Backtest rendering helpers
    _benchmark_charts.py             #   Benchmark comparison charts
    _futures_market.py               #   Tab 3: Futures market
    _accuracy.py                     #   Tab 4: Futures Market Simulation accuracy

  data_collection/                   # ENTSO-E + weather + commodity data collection (14 modules)
    cli.py, config.py, utils.py, join.py,
    entsoe_prices.py, entsoe_load.py, entsoe_generation.py,
    entsoe_forecasts.py, entsoe_flows.py, entsoe_ntc.py,
    entsoe_neighbours.py, gas_price.py, carbon_price.py, weather.py
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
| 2026-03-19 | 7 | COMPLETE — Convergence analysis: 3 theorems, 6 experiments, 29+7 new tests; non-convergence explained by constant step size |
| 2026-03-20 | 7 | REDO — Forecast-first refactor: forecast() is sole abstract method; act() derived with skip_buffer. 4 theorems proven (contraction mapping), 6 experiments validate theory to 4 decimal places. 225 tests pass |
| 2026-03-20 | 8 | STARTED — Oscillation research: 6 sub-phase documents (8a-8f), 15 experiments + 5 combinations planned across 4 research tracks |
| 2026-03-21 | 8 | DOCUMENTED AS HISTORICAL — Phase 8 retained as research record; winner (running_avg_k=5) was never implemented |
| 2026-03-21 | 9 | COMPLETE — EMA price-update experiments: alpha sweep {0.1..1.0} across 2024/2025; alpha=0.1 adopted as production default |
| 2026-03-21 | 10 | PLANNED — New phase to reconcile live market behaviour, explain dynamics, and drive stronger market-robust strategy design |
| 2026-03-21 | — | AUDIT — Phase renumbering: old Phase 9 → Phase 10; EMA experiments documented as new Phase 9. Warnings added to Phases 7 and 8. Architecture updated to reflect 67-strategy, 16-module backtest, 20-module dashboard state. 948 tests passing. |
| 2026-03-21 | 10a | COMPLETE — Baseline reconciliation: canonical config frozen (ema_alpha=0.1, 500 iters, 0.01 threshold), mismatches catalogued, WARNING blocks added to Phases 7 and 8 |
| 2026-03-21 | 10b | COMPLETE — Behaviour inventory: per-iteration metrics for 2024 (500 iters) and 2025 (327 iters), 827-row CSV, behaviour classifications (2024=oscillating_non_convergence, 2025=absorbing_collapse), 17 new tests, 5 high-priority behaviours identified |
| 2026-03-21 | 10c | COMPLETE — Mechanism attribution: 56-run ablation suite (EMA sweep, init sensitivity, 12 family ablations, 4 structural ablations). Key findings: ML strategies drive 2024 oscillation; 2025 convergence is fragile/path-dependent; alpha=0.01 is the only value achieving healthy convergence for both years. 28 new tests |
| 2026-03-21 | 10d | COMPLETE — Regime and cluster analysis: ML regression cluster (11-12 strategies) captures >90% market weight in both years; 49-strategy broad cluster contributes <5%; forecast and profit clusters disagree; ML cluster more robust to volatility. 11 new tests |
| 2026-03-21 | 10e | COMPLETE — Sentinel case studies: 5 sentinel cases (hvnc_2024, clsw_2024, ealost_2024, zact_2025, ealost_2025) with iteration-level causal traces. Key findings: 57 leadership changes in high-vol window, absorbing collapse traced step-by-step, early-accuracy-lost effect quantified. 23 new tests |
| 2026-03-21 | 10f | COMPLETE — Strategy robustness analysis: LOO analysis for all 67 strategies across both years. 49-67% strategies redundant (corr >0.95). PLSRegression worst destabiliser. Cross-border signals most valuable. Standalone PnL poor proxy for market contribution. 18 new tests |
| 2026-03-21 | 10g | COMPLETE — Stronger strategy design: 5 design rules derived from 10a-10f findings. 10 candidate strategies in 3 priority tiers. Top-3 implementation briefs. Acceptance criteria for Phase 11 |
| 2026-03-21 | 10h | COMPLETE — Synthesis and forward plan: unified causal explanation of market dynamics, historical vs current truth table, engine change recommendations (alpha=0.01, active-strategy floor, early stopping), Phase 11 scope defined (11a-11e) |
| 2026-03-21 | 10 | COMPLETE — All 8 sub-phases (10a-10h) finished. 1004+ tests pass. Phase 11 placeholder added to roadmap |
| 2026-03-22 | 11 | COMPLETE — 7 new strategies (SpreadMomentum, SelectiveHighConviction, TemperatureCurve, NuclearEvent, FlowImbalance, RegimeRidge, PrunedMLEnsemble) implemented with 44 tests. Strategy count 67->74. 1089 tests pass. Hyperparameter recommendations documented (ema_alpha=0.01, max_iterations=200) |
| 2026-03-22 | 12a | COMPLETE — SQLite forecast cache: per-strategy fingerprinting, warm-cache recompute-all in 2.4s (down from ~8-10 min). 19 new tests. 1108 tests pass. All theorems verified |
| 2026-03-22 | 12b | COMPLETE — Strategy expansion from 74 to 100: 26 new strategies in 6 batches (RadiationSolar, IntradayRange, OffshoreWindAnomaly, ForecastPriceError, PolandSpread, DenmarkSpread, CzechAustrianMean, SparkSpread, CarbonGasRatio, WeeklyAutocorrelation, MonthlyMeanReversion, LoadGenerationGap, RenewableRamp, NuclearGasSubstitution, VolatilityBreakout, SeasonalRegimeSwitch, WeekendMeanReversion, HighVolSkip, RadiationRegime, IndependentVote, MedianIndependent, SpreadConsensus, SupplyDemandBalance, ContrarianMomentum, ConvictionWeighted, BalancedLongShort). 171 new tests. 1279 tests pass. Warm-cache recompute-all: 2.6s with 100 strategies |
| 2026-03-22 | 13 | COMPLETE — ema_alpha production fix: applied ema_alpha=0.01 explicitly to both run_futures_market_evaluation() calls in recompute.py (engine defaults unchanged for backwards compat). Regenerated market_2024.pkl (converged=False, delta=0.0718, 500 iters) and market_2025.pkl (converged=True, delta=0.0093, 499 iters). 1279 tests pass. |
