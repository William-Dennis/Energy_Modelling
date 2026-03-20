# Strategy Expansion: 11 → 67 Strategies

## Overview

This folder tracks the live progress of expanding the strategy pool from 11 to 67
strategies. Documents are updated alongside each commit and serve as ground truth
for design decisions, signal analysis, and implementation status.

## Documents

| File | Contents |
|------|----------|
| [phase_A_feature_engineering.md](phase_A_feature_engineering.md) | 18 derived features added to the pipeline |
| [phase_B_issue3_strategies.md](phase_B_issue3_strategies.md) | 7 new strategies from Issue 3 |
| [phase_C_derived_threshold.md](phase_C_derived_threshold.md) | 14 single derived-feature threshold strategies |
| [phase_D_ml_strategies.md](phase_D_ml_strategies.md) | 15 ML model strategies |
| [phase_E_regime_calendar.md](phase_E_regime_calendar.md) | 8 calendar/temporal/regime strategies |
| [phase_F_ensemble.md](phase_F_ensemble.md) | 12 ensemble/meta strategies |
| [phase_G_feedback_loop.md](phase_G_feedback_loop.md) | Automated feedback loop infrastructure |
| [signal_registry.md](signal_registry.md) | All signals with correlations and usage |
| [strategy_registry.md](strategy_registry.md) | Full inventory of all strategies (live) |

## Constraints

- No strict line limits per file (relaxed from original 80-line rule)
- Graduated testing: 10+ tests for Tier 1-2, 5+ tests for ML/ensemble
- Derived features computed at **pipeline level** (`build_daily_backtest_frame`)
- All strategies must use only features available at decision time (no look-ahead bias)
- Test command: `pytest --ignore=tests/data_collection -q`

## Status Summary

| Phase | Strategies | Status |
|-------|-----------|--------|
| Baseline (pre-expansion) | 11 | ✅ Complete |
| A — Feature Engineering | infrastructure (18 features) | ✅ Complete |
| B — Issue 3 | +7 → 18 total | ✅ Complete |
| C — Derived Threshold | +14 → 32 total | ✅ Complete |
| D — ML Models | +15 → 47 total | ✅ Complete |
| E — Regime/Calendar | +8 → 55 total | ✅ Complete |
| F — Ensembles | +12 → 67 total | ✅ Complete |
| G — Feedback Loop | infrastructure | ⏳ Pending |
| **Total** | **67** | |
