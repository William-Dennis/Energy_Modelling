# Phase 0: Codebase Consolidation

## Status: COMPLETE

**Final test run (2026-03-19): 150 passed, 0 failed, 11 collection errors (all pre-existing `pytest-mock` in `data_collection/`)**

## Objective

Consolidate the two parallel strategy frameworks into one. The `challenge.*` framework
becomes THE strategy framework. Delete `strategy.*` (Framework 2) and `MarketEnvironment`.
Fix the broken `strategies/` package. Remove the Backtest dashboard tab. Port useful
metrics into challenge scoring.

## Prerequisites

- None (this is the first phase)

## Checklist

### 0a. Fix `strategies/` package (immediate)
- [x] Fix `always_long.py` — rename class from `AlwaysShortStrategy` to `AlwaysLongStrategy`, fix docstring, keep `return 1`
- [x] Rewrite `strategies/__init__.py` — import only `AlwaysLongStrategy` and `AlwaysShortStrategy`
- [x] Clear stale `strategies/__pycache__/` files
- [x] Write test: `tests/challenge/test_submission_strategies.py` — test only existing strategies

### 0b. Fix `_challenge.py` discovery
- [x] Change `module_name = f"submission.{stem}"` to `f"strategies.{stem}"` in `_challenge.py:68`
- [x] Verify `import strategies as _pkg` works (line 58)
- [x] Verify `obj.__module__ == module_name` filter still works after rename

### 0c. Port useful metrics into challenge scoring
- [x] Add `profit_factor` to `compute_challenge_metrics()` in `challenge/scoring.py`
- [x] Add `annualized_pnl_eur` to `compute_challenge_metrics()`
- [x] Add `monthly_pnl(daily_pnl)` standalone function to `challenge/scoring.py`
- [x] Add `rolling_sharpe(daily_pnl, window)` standalone function to `challenge/scoring.py`
- [x] Write tests for all new metrics in `tests/challenge/test_scoring.py`
- [x] Run scoring tests green

### 0d. Delete Framework 2 (`strategy.*` package)
- [x] Delete `src/energy_modelling/strategy/` directory (base.py, runner.py, analysis.py, perfect_foresight.py, naive_copy.py, template.py, `__init__.py`)
- [x] Delete `tests/strategy/` directory (test_runner.py, test_analysis.py, test_perfect_foresight.py, test_naive_copy.py)

### 0e. Delete `MarketEnvironment`
- [x] Delete `src/energy_modelling/market_simulation/market.py`
- [x] Delete `tests/market_simulation/test_market.py`
- [x] Update `src/energy_modelling/market_simulation/__init__.py` — remove `MarketEnvironment`, `DayState`, `Settlement`, `Signal`, `Trade` re-exports if only used by deleted code
- [x] Verify `challenge/data.py` still imports from `market_simulation.data` correctly

### 0f. Remove Backtest dashboard tab
- [x] Delete `src/energy_modelling/dashboard/_backtest.py`
- [x] Update `src/energy_modelling/dashboard/app.py` — remove Tab 2, reduce to 4 tabs
- [x] Clean up docstring references in `dashboard/__init__.py`
- [x] Update `dashboard/app.py` module docstring

### 0g. Clean up remaining references
- [x] Update `docs/market_implementation_status.md` to reflect consolidation
- [x] Remove docstring reference to `BacktestRunner` in `market_simulation/types.py`
- [x] Remove docstring reference to `PerfectForesightStrategy` in `market_simulation/market.py` (deleted)
- [x] Update `challenge/__init__.py` if any exports changed
- [x] Clean stale `strategy.analysis` comment in `challenge/scoring.py`
- [x] Clean stale `BacktestRunner` comment in `tests/market_simulation/test_types.py`

### 0h. Final verification
- [x] Run full test suite: 0 collection errors from our code (11 pre-existing in data_collection)
- [x] All 150 remaining tests pass
- [x] `import strategies` works
- [x] `import energy_modelling.challenge` works
- [x] `import energy_modelling.market_simulation` works (data.py, contract.py, types.py)
- [x] `import energy_modelling.dashboard` works

## Files to DELETE

| File | Reason |
|------|--------|
| `src/energy_modelling/strategy/__init__.py` | Framework 2 removal |
| `src/energy_modelling/strategy/base.py` | Framework 2 removal |
| `src/energy_modelling/strategy/runner.py` | Framework 2 removal |
| `src/energy_modelling/strategy/analysis.py` | Metrics ported to scoring.py |
| `src/energy_modelling/strategy/perfect_foresight.py` | Will be rewritten as ChallengeStrategy in Phase 4 |
| `src/energy_modelling/strategy/naive_copy.py` | Replaced by always_long.py |
| `src/energy_modelling/strategy/template.py` | Framework 2 removal |
| `src/energy_modelling/market_simulation/market.py` | MarketEnvironment removal |
| `src/energy_modelling/dashboard/_backtest.py` | Uses Framework 2 |
| `tests/strategy/test_runner.py` | Tests Framework 2 |
| `tests/strategy/test_analysis.py` | Tests Framework 2 |
| `tests/strategy/test_perfect_foresight.py` | Tests Framework 2 |
| `tests/strategy/test_naive_copy.py` | Tests Framework 2 |
| `tests/market_simulation/test_market.py` | Tests MarketEnvironment |

## Files to MODIFY

| File | Change |
|------|--------|
| `strategies/__init__.py` | Strip broken imports, keep only AlwaysLong + AlwaysShort |
| `strategies/always_long.py` | Rename class to `AlwaysLongStrategy`, fix docstring |
| `src/energy_modelling/dashboard/_challenge.py` | `submission` -> `strategies` in import path |
| `src/energy_modelling/dashboard/app.py` | Remove Backtest tab (5 tabs -> 4) |
| `src/energy_modelling/dashboard/__init__.py` | Clean docstring references |
| `src/energy_modelling/market_simulation/__init__.py` | Remove MarketEnvironment re-export |
| `src/energy_modelling/market_simulation/types.py` | Remove BacktestRunner docstring ref |
| `src/energy_modelling/challenge/scoring.py` | Add ported metrics |
| `tests/challenge/test_submission_strategies.py` | Rewrite for existing strategies only |
| `tests/challenge/test_scoring.py` | Add tests for new metrics |
| `docs/market_implementation_status.md` | Reflect consolidation |

## Test Commands

```bash
# Run only scoring tests (fast feedback)
python -m pytest tests/challenge/test_scoring.py -v

# Run only submission strategy tests
python -m pytest tests/challenge/test_submission_strategies.py -v

# Run all challenge tests
python -m pytest tests/challenge/ -v

# Full regression
python -m pytest -v
```

## Verification Criteria

- 0 test collection errors
- All remaining tests pass (target: ~180+ tests, after removing ~60 Framework 2 tests)
- No import errors for `strategies`, `energy_modelling.challenge`, `energy_modelling.market_simulation`, `energy_modelling.dashboard`
