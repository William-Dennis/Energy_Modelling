# Issue 2: Dashboard Loads Saved Results, Recompute on Demand

## Summary

The dashboard currently recomputes all backtests and market simulations live on every button press. Change it to load pre-computed results from disk by default, with a "Recompute" button for live recalculation. Add a CLI command to regenerate all results.

## Motivation

- **Startup speed**: The dashboard should open and show data immediately, not require clicking "Run Comparison" first.
- **Reproducibility**: Saved results are deterministic and versioned.
- **Separation of concerns**: Computation and visualization should be decoupled. The expensive backtest/market runs happen offline via CLI; the dashboard is a read-only viewer by default.
- **Professional UX**: Users expect dashboards to show something on load.

## Current Architecture

### Data flow (all in-memory, no persistence)

```
Tab 2 (Backtest)
  |- Loads raw CSV: load_daily("data/backtest/daily_public.csv")
  |- User clicks "Run Comparison" -> evaluate_all() runs ALL backtests live
  |- Stores into st.session_state (lost on restart)
       |
       v
Tab 3 (Futures Market)
  |- Reads st.session_state["backtest_val_results"]
  |- User clicks "Run Synthetic Market" -> runs market evaluation live
  |- Stores into st.session_state
       |
       v
Tab 4 (Accuracy)
  |- Reads st.session_state["market_2024"] / ["market_2025"]
  |- Pure rendering (no computation)
```

### Session state keys

| Key | Set by | Read by |
|-----|--------|---------|
| `backtest_val_results` | `_backtest.py` | `_futures_market.py` |
| `backtest_hid_results` | `_backtest.py` | (unused) |
| `backtest_selected` | `_backtest.py` | `_futures_market.py` |
| `backtest_public_daily` | `_backtest.py` | `_futures_market.py`, `_accuracy.py` |
| `backtest_hidden_daily` | `_backtest.py` | `_futures_market.py`, `_accuracy.py` |
| `market_2024` | `_futures_market.py` | `_accuracy.py` |
| `market_2025` | `_futures_market.py` | `_accuracy.py` |

## Implementation Plan

### 1. Create `backtest/io.py` -- serialization helpers

```python
RESULTS_DIR = Path("data/results")

def save_backtest_results(results: dict[str, BacktestResult], path: Path) -> None:
    """Serialize backtest results to disk (pickle or JSON)."""

def load_backtest_results(path: Path) -> dict[str, BacktestResult] | None:
    """Load backtest results from disk. Returns None if not found."""

def save_market_results(result: FuturesMarketResult, path: Path) -> None:
    """Serialize market evaluation result to disk."""

def load_market_results(path: Path) -> FuturesMarketResult | None:
    """Load market evaluation result from disk. Returns None if not found."""

def results_exist() -> bool:
    """Check if pre-computed results are available on disk."""
```

**Format decision**: Use pickle for simplicity (the objects contain DataFrames and custom dataclasses). Consider JSON with DataFrame-to-dict conversion for portability if needed.

### 2. Update `scripts/run_full_backtest.py`

After running backtests and market evaluation, call `save_backtest_results()` and `save_market_results()` to persist everything to `data/results/`.

### 3. Update `_backtest.py:render()`

```python
def render() -> None:
    # Try loading saved results on first visit
    if "backtest_val_results" not in st.session_state:
        cached = load_backtest_results(RESULTS_DIR / "backtest_val_2024.pkl")
        if cached is not None:
            st.session_state["backtest_val_results"] = cached
            # ... load other cached state ...

    # Show "Recompute" button (secondary, not primary)
    recompute = st.button("Recompute Results", type="secondary", key="ch_recompute")

    # If we have results (cached or session), render them
    val_results = st.session_state.get("backtest_val_results")
    if val_results is not None and not recompute:
        _render_results(val_results, ...)
        return

    # Otherwise, run live computation
    if recompute or val_results is None:
        with st.spinner("Running backtests..."):
            val_results, hid_results = evaluate_all(...)
        save_backtest_results(val_results, ...)
        # ... store and render ...
```

### 4. Update `_futures_market.py:render()` -- same pattern

Load from `data/results/market_2024.pkl` on initial render. Keep "Run Synthetic Market" button for recomputation.

### 5. Update `_accuracy.py:render()` -- same pattern

Load market results from disk if session state is empty.

### 6. Update README with regeneration docs

Add a "Regenerating Results" section:
```markdown
## Regenerating Results

To recompute all backtest and market results:

    uv run python scripts/run_full_backtest.py

Results are saved to `data/results/` and loaded by the dashboard on startup.
```

## Files to Create

- `src/energy_modelling/backtest/io.py`
- `tests/backtest/test_io.py`

## Files to Modify

- `src/energy_modelling/dashboard/_backtest.py` -- load saved results on startup
- `src/energy_modelling/dashboard/_futures_market.py` -- load saved results on startup
- `src/energy_modelling/dashboard/_accuracy.py` -- load saved results on startup
- `scripts/run_full_backtest.py` -- save results after running
- `src/energy_modelling/backtest/__init__.py` -- export io functions
- `README.md` -- add regeneration docs

## Tests

- `test_io.py`: round-trip test (save -> load -> compare) for both backtest and market results
- Test that `load_backtest_results` returns `None` when file doesn't exist
- Test that `results_exist()` reflects actual disk state

## Acceptance Criteria

- [x] Dashboard opens and displays pre-computed results without any button press — **auto-loads from `data/results/` on startup**
- [x] "Recompute" button triggers live recalculation and saves to disk — **"Run Comparison" saves after computation**
- [x] `scripts/run_full_backtest.py` produces all result files in `data/results/`
- [x] Round-trip serialization tests pass (save -> load == original) — **6 tests in `test_io.py`**
- [x] README documents how to regenerate results — **"Regenerating Results" section added**
- [x] All 276+ existing tests still pass — **295 tests now passing**

## Status: ✅ COMPLETE

### Files Created
- `src/energy_modelling/backtest/io.py` — pickle-based save/load for `BacktestResult` and `FuturesMarketResult`
- `tests/backtest/test_io.py` — 6 round-trip and existence tests

### Files Modified
- `src/energy_modelling/dashboard/_backtest.py` — auto-loads cached results on startup, saves after live computation
- `src/energy_modelling/dashboard/_futures_market.py` — loads/saves market results from disk
- `src/energy_modelling/dashboard/_accuracy.py` — falls back to disk-loaded market results
- `scripts/run_full_backtest.py` — saves results to `data/results/` after running
- `src/energy_modelling/backtest/__init__.py` — exports IO functions

## Labels

`enhancement`, `dashboard`, `priority-high`

## Parallel Safety

Safe to work on in parallel with Issues 1, 3, and 5. Issue 4 depends on this (dashboard benchmark views need saved results). Issue 6 extends this (CLI wrapper).
