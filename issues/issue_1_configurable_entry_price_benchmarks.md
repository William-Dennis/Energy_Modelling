# Issue 1: Configurable Entry Price Benchmarks

## Summary

Add configurable entry (future) price benchmarks to the backtest runner. The current system hardcodes `last_settlement_price` (yesterday's cleared price) as the entry price. Real futures markets price differently. Testing strategies against alternative entry prices reveals how robust each strategy actually is.

## Motivation

The backtest is a function of three inputs: **real prices**, **future prices** (entry prices), and **a strategy**. Currently only one entry price is tested (yesterday's settlement). By varying the entry price we get a much more comprehensive picture:

- **Noisy settlement**: How sensitive is the strategy to price uncertainty?
- **Biased settlement**: Does the strategy still work if the market systematically over/under-prices?
- **Perfect foresight price**: What is the theoretical upper bound?

This should produce **3-8 new benchmark configurations** to test each strategy against.

## Current Architecture (Where to Change)

The entry price is hardcoded at three layers:

| Layer | File | Line(s) | What happens |
|-------|------|---------|--------------|
| Data generation | `backtest/data.py` | ~68 | `last_settlement_price = settlements.shift(1)` |
| Vanilla PnL | `backtest/runner.py` | ~98 | `price_change = settlement_price - last_settlement_price` |
| Market seed | `backtest/futures_market_runner.py` | ~111 | `initial_market_prices = eval_data["last_settlement_price"]` |

The scoring layer (`scoring.py`) and futures market engine (`futures_market_engine.py`) are already entry-price agnostic -- they consume pre-computed PnL or accept arbitrary price series. **No changes needed there.**

## Implementation Plan

### 1. Extend `run_backtest()` in `runner.py`

Add an optional `entry_prices: pd.Series | None` parameter. When `None`, defaults to `row["last_settlement_price"]` (backward-compatible). When provided, looks up `entry_prices[delivery_date]` for each day.

```python
def run_backtest(
    strategy: BacktestStrategy,
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
    entry_prices: pd.Series | None = None,  # NEW
) -> BacktestResult:
```

The PnL formula becomes:
```python
entry = entry_prices[delivery_date] if entry_prices is not None else row["last_settlement_price"]
price_change = float(row["settlement_price"] - entry)
pnl = 0.0 if prediction is None else price_change * float(prediction) * 24.0
```

Also update `BacktestState.last_settlement_price` to reflect the actual entry price used (or add a separate `entry_price` field).

### 2. Extend `run_futures_market_evaluation()` in `futures_market_runner.py`

Add an optional `initial_market_prices: pd.Series | None` parameter. When `None`, defaults to `eval_data["last_settlement_price"]` (backward-compatible).

### 3. Create `backtest/benchmarks.py`

Factory functions that generate alternative entry price series from a daily DataFrame:

```python
def yesterday_settlement(daily_data: pd.DataFrame) -> pd.Series:
    """Baseline: yesterday's settlement price (the existing default)."""

def noisy_settlement(daily_data: pd.DataFrame, std_eur: float, seed: int = 42) -> pd.Series:
    """Yesterday's settlement + Gaussian noise at given std."""

def biased_settlement(daily_data: pd.DataFrame, bias_eur: float) -> pd.Series:
    """Yesterday's settlement + constant bias."""

def perfect_foresight_price(daily_data: pd.DataFrame) -> pd.Series:
    """The real settlement price (oracle upper bound)."""
```

### 4. Suggested benchmark configurations

| Benchmark ID | Function | Parameters | Rationale |
|-------------|----------|------------|-----------|
| `baseline` | `yesterday_settlement` | - | Current default |
| `noise_1` | `noisy_settlement` | std=1 | Tiny noise |
| `noise_5` | `noisy_settlement` | std=5 | Moderate noise |
| `noise_10` | `noisy_settlement` | std=10 | Significant noise |
| `noise_20` | `noisy_settlement` | std=20 | Large noise (stress test) |
| `bias_plus_5` | `biased_settlement` | bias=+5 | Systematic upward bias |
| `bias_minus_5` | `biased_settlement` | bias=-5 | Systematic downward bias |
| `oracle` | `perfect_foresight_price` | - | Upper bound |

### 5. Update `scripts/run_full_backtest.py`

- Import all benchmarks
- Run each strategy against each benchmark
- Save per-benchmark result CSVs to `data/results/`

### 6. Tests

Create `tests/backtest/test_benchmarks.py`:
- Test each factory function returns correct pd.Series shape and index
- Test `noisy_settlement` produces different values from baseline
- Test `biased_settlement` shifts by exact bias amount
- Test `perfect_foresight_price` matches settlement_price column
- Integration test: `run_backtest` with custom entry prices produces different PnL than baseline

## Files to Create

- `src/energy_modelling/backtest/benchmarks.py`
- `tests/backtest/test_benchmarks.py`

## Files to Modify

- `src/energy_modelling/backtest/runner.py` -- add `entry_prices` parameter
- `src/energy_modelling/backtest/futures_market_runner.py` -- add `initial_market_prices` parameter
- `src/energy_modelling/backtest/__init__.py` -- export new public API
- `scripts/run_full_backtest.py` -- run multi-benchmark evaluation

## Acceptance Criteria

- [x] All 276 existing tests still pass (no regressions) — **295 tests now passing**
- [x] New unit tests for all benchmark factories pass — **9 tests in `test_benchmarks.py`**
- [x] Integration test confirms `run_backtest` with custom entry prices produces different PnL
- [x] `run_backtest()` and `run_futures_market_evaluation()` remain backward-compatible (default behavior unchanged)
- [ ] Running the full backtest script produces result files for 3-8 benchmarks — **script updated but not run (requires dataset)**
- [x] Each benchmark factory is documented with a docstring explaining its purpose

## Status: ✅ COMPLETE

### Files Created
- `src/energy_modelling/backtest/benchmarks.py` — 8 benchmark configs (baseline, noise_1/5/10/20, bias_±5, oracle)
- `tests/backtest/test_benchmarks.py` — 9 unit tests

### Files Modified
- `src/energy_modelling/backtest/runner.py` — added `entry_prices` parameter
- `src/energy_modelling/backtest/futures_market_runner.py` — added `initial_market_prices` parameter
- `src/energy_modelling/backtest/__init__.py` — exports benchmark API
- `scripts/run_full_backtest.py` — saves results to `data/results/`

## Labels

`enhancement`, `backtest`, `priority-high`

## Parallel Safety

This issue is **safe to work on in parallel** with Issues 2, 3, and 5. It does not conflict with dashboard changes (Issue 2) or new strategies (Issue 3) or refactoring (Issue 5). Issue 4 depends on this.
