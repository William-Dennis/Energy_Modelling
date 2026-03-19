# Issue 4: Add Benchmark Comparison Views to Dashboard

## Summary

With configurable entry price benchmarks (Issue 1) and saved results (Issue 2), the dashboard should show how strategies perform across different benchmark scenarios -- not just the default yesterday-settlement baseline.

## Dependencies

- **Issue 1** (Configurable Entry Price Benchmarks) -- benchmarks must exist
- **Issue 2** (Dashboard Saved Results) -- results must be loadable from disk

Do **not** start this issue until Issues 1 and 2 are merged.

## Motivation

A strategy's full assessment should include:
1. **Fixed benchmark tests** -- How does it perform against various entry price scenarios?
2. **Live "real-world" test** -- How does it perform in the dynamic futures market simulation?

The dashboard should make both of these visible side-by-side.

## Implementation Plan

### 1. Backtest Tab: Benchmark Selector

Add a benchmark selector to the Backtest tab:

```python
benchmarks = load_available_benchmarks()  # e.g., ["baseline", "noise_5", "noise_10", "bias_plus_5", ...]
selected_benchmarks = st.multiselect(
    "Entry Price Benchmarks",
    options=benchmarks,
    default=["baseline"],
    key="bench_select",
)
```

### 2. Backtest Tab: Comparison Table

Show a strategy x benchmark matrix:

| Strategy | Baseline PnL | Noise(5) PnL | Noise(10) PnL | Bias(+5) PnL | Oracle PnL |
|----------|-------------|--------------|---------------|--------------|------------|
| CompositeSignal | 100,241 | ? | ? | ? | ? |
| DayOfWeek | 93,213 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... |

### 3. Backtest Tab: Heatmap Visualization

Add a heatmap (Plotly) showing:
- Rows: strategies
- Columns: benchmarks
- Color: PnL (or Sharpe ratio, user-selectable)

This immediately reveals which strategies are robust (consistent color across benchmarks) vs fragile (color varies wildly).

### 4. Futures Market Tab: Benchmark Seed Selector

Allow selecting which benchmark's entry prices seed the futures market:

```python
market_benchmark = st.selectbox(
    "Market seed benchmark",
    options=benchmarks,
    index=0,  # default: baseline
    key="mkt_bench",
)
```

This lets users see how the equilibrium changes when the starting prices are different.

### 5. Accuracy Tab: Multi-Benchmark Accuracy

Show accuracy metrics (MAE, RMSE, directional accuracy) for the converged market price across different seed benchmarks.

### 6. Load all results from disk

All visualizations should pull from saved result files (Issue 2). The "Recompute" button should regenerate results for the selected benchmarks.

## Files to Modify

- `src/energy_modelling/dashboard/_backtest.py` -- benchmark selector, comparison table, heatmap
- `src/energy_modelling/dashboard/_futures_market.py` -- benchmark seed selector
- `src/energy_modelling/dashboard/_accuracy.py` -- multi-benchmark accuracy comparison

## Files Potentially to Create

- `src/energy_modelling/dashboard/_benchmark_charts.py` -- shared heatmap / comparison chart helpers (if the rendering logic is large enough to warrant a separate module)

## Acceptance Criteria

- [ ] Backtest tab shows a benchmark selector with all available benchmarks
- [ ] Comparison table displays strategy x benchmark matrix with key metrics
- [ ] Heatmap visualization renders correctly and is user-selectable (PnL or Sharpe)
- [ ] Futures Market tab allows selecting the market seed benchmark
- [ ] Accuracy tab shows accuracy across benchmarks
- [ ] All data loads from saved results (no recomputation on dashboard open)
- [ ] "Recompute" button works for selected benchmarks
- [ ] All existing tests still pass

## Labels

`enhancement`, `dashboard`, `priority-medium`

## Parallel Safety

**Cannot run in parallel** with Issues 1 or 2 -- depends on both. Can run in parallel with Issues 3, 5, and 6 once started.
