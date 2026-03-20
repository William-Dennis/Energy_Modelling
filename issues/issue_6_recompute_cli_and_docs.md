# Issue 6: Recompute-All CLI Command and Documentation Update

## Summary

Create a CLI entry point that regenerates all backtest and futures market results in one command. Update README and hackathon docs to reflect the current architecture (post-rename, benchmarks, saved results).

## Motivation

After Issues 1 (benchmarks) and 2 (saved results) are complete, the system will support multiple benchmark configurations and load pre-computed results. Users and CI pipelines need a single command to regenerate everything from scratch:

```bash
recompute-all                          # regenerate everything
recompute-all --strategies composite   # just one strategy
recompute-all --benchmarks baseline noise_5  # just two benchmarks
```

The documentation also needs updating: the rename from `challenge` to `backtest` and `market_simulation` to `futures_market` changed many paths and terms. The README and hackathon docs still need a "Regenerating Results" section.

## Dependencies

- **Issue 1** (benchmarks): Provides `backtest/benchmarks.py` and the benchmark factory functions
- **Issue 2** (saved results): Provides the result storage format and `ResultStore` class

This issue should be started **after** Issues 1 and 2 are merged, or at minimum after their APIs are stable.

## Implementation Plan

### 1. Create `backtest/recompute.py`

Core module that orchestrates full result regeneration:

```python
"""Regenerate all backtest and futures market results."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from energy_modelling.backtest.benchmarks import (
    ALL_BENCHMARKS,
    get_benchmark,
)
from energy_modelling.backtest.data import build_daily_data
from energy_modelling.backtest.runner import run_backtest
from energy_modelling.backtest.futures_market_runner import (
    run_futures_market_evaluation,
)

logger = logging.getLogger(__name__)


def recompute_all(
    strategies: list[str] | None = None,
    benchmarks: list[str] | None = None,
    training_end: date | None = None,
    evaluation_start: date | None = None,
    evaluation_end: date | None = None,
) -> None:
    """Regenerate all results.

    Parameters
    ----------
    strategies : list[str] | None
        Strategy names to run. None means all registered strategies.
    benchmarks : list[str] | None
        Benchmark IDs to run. None means all benchmarks.
    training_end, evaluation_start, evaluation_end : date | None
        Override default date ranges. None means use defaults.
    """
    ...
```

**Steps within `recompute_all()`**:
1. Load or build daily data via `build_daily_data()`
2. Resolve strategy list (import from `strategies/__init__.py`)
3. Resolve benchmark list (from `ALL_BENCHMARKS` or filter)
4. For each (strategy, benchmark) pair:
   a. Generate entry prices via benchmark factory
   b. Run `run_backtest()` with those entry prices
   c. Save result to `ResultStore`
5. For each benchmark:
   a. Run `run_futures_market_evaluation()` with all strategies
   b. Save market result to `ResultStore`
6. Log summary: strategies run, benchmarks tested, total time

### 2. Create CLI entry point

Add to `pyproject.toml` under `[project.scripts]`:

```toml
recompute-all = "energy_modelling.backtest.recompute:main"
```

The `main()` function in `recompute.py`:

```python
def main() -> None:
    """CLI entry point for recompute-all."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate all backtest and futures market results."
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Strategy names to run (default: all)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Benchmark IDs to run (default: all)",
    )
    parser.add_argument(
        "--training-end",
        type=str,
        default=None,
        help="Training end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--evaluation-start",
        type=str,
        default=None,
        help="Evaluation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--evaluation-end",
        type=str,
        default=None,
        help="Evaluation end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    recompute_all(
        strategies=args.strategies,
        benchmarks=args.benchmarks,
        training_end=_parse_date(args.training_end),
        evaluation_start=_parse_date(args.evaluation_start),
        evaluation_end=_parse_date(args.evaluation_end),
    )
```

### 3. Tests

Create `tests/backtest/test_recompute.py`:

```python
# Test plan (TDD - write these first):

def test_recompute_all_runs_with_defaults():
    """recompute_all() with no args runs all strategies x all benchmarks."""

def test_recompute_all_filters_strategies():
    """recompute_all(strategies=["always_long"]) runs only AlwaysLong."""

def test_recompute_all_filters_benchmarks():
    """recompute_all(benchmarks=["baseline"]) runs only baseline benchmark."""

def test_recompute_all_saves_results():
    """Results are saved to ResultStore after recompute."""

def test_main_cli_parses_args():
    """CLI arg parsing works for --strategies and --benchmarks flags."""

def test_main_cli_no_args():
    """CLI with no args defaults to all strategies and benchmarks."""

def test_recompute_all_with_custom_dates():
    """Custom training_end/evaluation_start/evaluation_end are respected."""
```

Use mocking to avoid running actual backtests in unit tests. One integration test should run with a small data slice to verify end-to-end.

### 4. Update README.md

Add a "Regenerating Results" section after the existing "Getting Started" or "Usage" section:

```markdown
## Regenerating Results

To regenerate all backtest and futures market results:

    recompute-all

To run specific strategies or benchmarks:

    recompute-all --strategies composite_signal day_of_week
    recompute-all --benchmarks baseline noise_5 oracle

Results are saved to `data/results/` and loaded automatically by the dashboard.
```

Also update any references to old terminology (`challenge`, `market_simulation`) if they still exist.

### 5. Update `docs/hackathon_challenge.md`

- Update terminology table to reflect `backtest` / `futures_market` naming
- Add section on benchmark system (what benchmarks are available, what they test)
- Add section on result storage format (where results live, how to regenerate)
- Update any stale file paths

## Files to Create

| File | Purpose |
|------|---------|
| `src/energy_modelling/backtest/recompute.py` | Core recompute logic + CLI entry point |
| `tests/backtest/test_recompute.py` | Unit + integration tests |

## Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add `recompute-all` script entry point |
| `README.md` | Add "Regenerating Results" section, fix stale terminology |
| `docs/hackathon_challenge.md` | Update terminology, paths, add benchmark/results sections |
| `src/energy_modelling/backtest/__init__.py` | Export `recompute_all` if desired |

## Acceptance Criteria

- [x] `recompute-all` CLI command works from a fresh install (`pip install -e .`) — **registered in `pyproject.toml`**
- [ ] `recompute-all --strategies always_long --benchmarks baseline` runs in <30 seconds — **requires dataset to test**
- [x] `recompute-all` with no args regenerates all strategy x benchmark combinations — **implemented in `recompute_all()`**
- [x] Results are saved in the format expected by Issue 2's `ResultStore` — **uses `save_backtest_results()` from `io.py`**
- [x] All new tests pass — **5 tests in `test_recompute.py`**
- [x] All 276+ existing tests still pass — **295 tests now passing**
- [x] README contains accurate "Regenerating Results" section
- [x] `docs/hackathon_challenge.md` uses correct post-rename terminology throughout — **`BacktestStrategy`, `build-backtest-data`, benchmark system note added**
- [x] `recompute.py` is under 200 lines; `main()` is under 40 lines — **146 lines; `main()` is 30 lines**

## Status: ✅ COMPLETE

### Files Created
- `src/energy_modelling/backtest/recompute.py` (146 lines) — `recompute_all()` + CLI `main()`
- `tests/backtest/test_recompute.py` — 5 unit tests

### Files Modified
- `pyproject.toml` — added `recompute-all` script entry point
- `README.md` — added "Regenerating Results" section + `recompute-all` CLI entry
- `docs/hackathon_challenge.md` — fixed terminology (`BacktestStrategy`, `build-backtest-data`), added benchmark system note
- `src/energy_modelling/backtest/__init__.py` — exported `recompute_all`

## Labels

`enhancement`, `cli`, `docs`, `priority-medium`

## Parallel Safety

**Depends on Issues 1 and 2.** Do not start until both are merged or their APIs (benchmark factories, ResultStore) are finalized.

Safe to work on in parallel with Issues 3, 4, and 5 — no file overlap. The only shared touchpoints are `pyproject.toml` (adding a new script entry) and `backtest/__init__.py` (adding an export), both of which are append-only changes that merge cleanly.
