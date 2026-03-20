# Phase 8d: Initialisation Strategies

## Status: PENDING

## Objective

Investigate how the choice of initial market price `P^m_0` affects oscillation
amplitude and convergence, and whether better initialisation can reduce the
severity of the problem before any iterative mechanism changes are needed.

## Motivation

The current engine initialises at `last_settlement_price` — the previous day's
settlement price.  On gap days or after extreme events, this can be very far
from the true price:

| Date | last_settlement | real_price | Gap (EUR/MWh) |
|------|-----------------|------------|---------------|
| 2024-12-13 | 395.7 | 177.9 | 217.8 |
| 2024-12-15 | 99.4 | 43.8 | 55.6 |
| 2024-10-14 | 11.1 | 110.0 | 98.9 |
| 2024-11-24 | 72.7 | 3.9 | 68.8 |

A 217.8 EUR/MWh gap on Dec 13 creates an enormous initial profit signal that
biases the first iteration heavily toward whichever cluster forecasts the
correction correctly.  This "first-mover advantage" then seeds the oscillation.

---

## Experiment 1: Rolling Mean Initial Price

### Hypothesis

Using a rolling N-day average of settlement prices as the initial market price
will smooth out day-to-day extremes and reduce the initial gap.

### Design

```python
def rolling_initial_prices(
    settlement_prices: pd.Series,
    last_settlement_prices: pd.Series,
    window: int = 5,
) -> pd.Series:
    """Initial market price = rolling mean of last N settlement prices."""
    # Use last_settlement (shifted) as the base, then average over window
    rolling = settlement_prices.shift(1).rolling(window, min_periods=1).mean()
    return rolling.reindex(last_settlement_prices.index).fillna(last_settlement_prices)
```

Sweep `window` over {1, 3, 5, 7, 10, 20} (window=1 is the baseline).

### Expected Outcome

For Dec 13, 2024:
- Window 1 (baseline): P_0 = 395.7
- Window 5: P_0 ~ mean(395.7, 266.8, 177.9, ...) ~ more moderate value
- Window 10: P_0 further smoothed toward monthly mean

Larger windows reduce the gap but also reduce the market's responsiveness to
recent information.  There is a bias-variance tradeoff.

### Measurement

For each window:
1. Initial gap: mean |P_0 - P_real| across all days
2. Post-iteration gap: final market MAE
3. Convergence: does the market converge with this initialisation?

---

## Experiment 2: Equal-Weighted Strategy Mean as Initial Price

### Hypothesis

Instead of using historical prices, initialise the market at the equal-weighted
mean of all strategy forecasts (before any profit weighting).

### Design

```python
def forecast_mean_initial_prices(
    strategy_forecasts: dict[str, dict],
    dates: pd.Index,
) -> pd.Series:
    """Initial price = equal-weighted mean of all strategy forecasts."""
    prices = {}
    for t in dates:
        forecasts = [
            float(f[t]) for f in strategy_forecasts.values()
            if t in f
        ]
        prices[t] = np.mean(forecasts) if forecasts else np.nan
    return pd.Series(prices).dropna()
```

### Expected Outcome

- This is essentially "iteration 0 with uniform weights"
- If the strategy pool is well-calibrated on average, this should be closer
  to the real price than `last_settlement_price`
- Risk: if the strategy pool is biased (e.g. more long-biased strategies than
  short-biased), the mean forecast may be systematically biased

### Advantage Over Iter 0

Iteration 0 already uses all strategies, but profit-weighted.  The equal-weighted
mean avoids the profit-weighting step entirely, giving every strategy equal say
regardless of performance.  This is a different starting point that may lead to
a different convergence basin.

---

## Experiment 3: Forecast Anchoring / Outlier Clipping

### Hypothesis

Clipping strategy forecasts to a plausible range (anchored to the initial price)
will reduce the extreme forecast divergence that drives the oscillation.

### Design

For each day t and strategy i, before using the forecast in the market:
```python
def clip_forecast(
    forecast: float,
    anchor: float,
    max_deviation: float = 50.0,  # EUR/MWh
) -> float:
    """Clip forecast to [anchor - max_dev, anchor + max_dev]."""
    return max(anchor - max_deviation, min(forecast, anchor + max_deviation))
```

The anchor can be:
- `last_settlement_price` (conservative)
- Current market price `P^m_t^(k)` (adapts with iterations)
- A rolling mean of recent prices

### Expected Outcome

- Directly caps the maximum price swing per iteration per day
- For Dec 15, 2024: if anchor = 99.4 and max_dev = 50, forecasts are clipped
  to [49.4, 149.4], preventing the full 57.8 EUR/MWh swing
- Preserves relative ordering of strategies (a strategy forecasting 45 vs 103
  will still have different directions)

### Risks

- Clipping is information-lossy — on days where the real price is far from the
  anchor (e.g. 395.7 vs 266.8), clipping prevents the market from reaching the
  true price
- The `max_deviation` parameter is a hyperparameter that would need tuning
- May create pathological cases where all strategies are clipped to the same
  value, making the market uninformative

---

## Experiment 4: Percentile-Based Initial Price

### Hypothesis

Using the median (or other percentile) of recent settlement prices instead of
the last settlement price reduces sensitivity to single-day spikes.

### Design

```python
def percentile_initial_prices(
    settlement_prices: pd.Series,
    window: int = 5,
    percentile: float = 50.0,
) -> pd.Series:
    """Initial price = rolling percentile of recent settlements."""
    return settlement_prices.shift(1).rolling(window, min_periods=1).quantile(
        percentile / 100.0
    )
```

### Expected Outcome

Similar to rolling mean but more robust to outliers.  The median of the last 5
days is less affected by a single 395.7 EUR/MWh spike than the mean.

---

## Comparison Matrix

| Method | Spec Impact | Hyperparams | Expected Gap Reduction | Convergence Aid |
|--------|-------------|-------------|----------------------|-----------------|
| Rolling mean | None (init only) | window | Moderate | Indirect |
| Forecast mean | None (init only) | None | High if strategies are well-calibrated | Indirect |
| Forecast clipping | Modifies forecasts | max_deviation | High on extreme days | Direct |
| Percentile | None (init only) | window, percentile | Moderate-high | Indirect |

Note: Experiments 1, 2, and 4 only change the starting point.  They do NOT
modify the iteration dynamics — the undampened engine still runs the same update
rule.  Therefore, they can reduce the *initial amplitude* of oscillation but
cannot break a limit cycle if one exists at any starting point.

Experiment 3 (clipping) modifies the forecasts themselves and can potentially
break the cycle by reducing the forecast divergence that causes it.  However,
it is also the most interventionist and information-lossy.

---

## Success Criteria

1. At least one initialisation strategy reduces iter-0 MAE below the current 15.10
2. Combined with a convergence mechanism (8b or 8e), achieves convergence
3. No degradation of final market MAE vs baseline

## Files to Modify

- `src/energy_modelling/backtest/futures_market_runner.py` — alternative `initial_market_prices` computation
- `scripts/sweep_initialisation.py` — new script for initialisation experiments
- `tests/backtest/test_futures_market_runner.py` — tests for new initialisation methods
