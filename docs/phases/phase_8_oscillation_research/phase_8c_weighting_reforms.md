# Phase 8c: Weighting Reforms

## Status: PENDING

## Objective

Investigate modifications to the strategy weighting scheme (spec Step 3) that
reduce winner-take-all dynamics and make the market price more robust to
outlier strategies.

## Motivation

The current weighting scheme is linear-proportional:
```
w_i = max(Pi_i, 0) / sum_j max(Pi_j, 0)
```

This has two problematic properties:

1. **Profit-proportional weight**: A strategy with 6,363 EUR profit gets 6.6x
   the weight of one with 968 EUR profit.  Small profit differences on volatile
   days create large weight swings between iterations.

2. **Cluster dominance**: When 36 strategies from the same regression cluster
   (Lasso, Elastic Net, Ridge, Bayesian Ridge, etc.) are all profitable with
   similar forecasts, they collectively dominate even if each individual weight
   is moderate (e.g. 36 x 0.020 = 0.72 combined weight).

---

## Experiment 1: Per-Strategy Weight Cap

### Hypothesis

Capping the maximum weight any single strategy can receive will prevent
one dominant strategy from steering the entire market, reducing oscillation
amplitude.

### Design

After computing normalised weights, clip any weight exceeding `w_max` and
redistribute the excess proportionally:

```python
def compute_weights_capped(
    strategy_profits: dict[str, float],
    w_max: float = 0.10,
) -> dict[str, float]:
    """Weights with per-strategy cap."""
    raw = {name: max(profit, 0.0) for name, profit in strategy_profits.items()}
    total = sum(raw.values())
    if total == 0.0:
        return {name: 0.0 for name in raw}

    weights = {name: w / total for name, w in raw.items()}

    # Iterative clipping until no weight exceeds cap
    for _ in range(100):
        excess = sum(max(w - w_max, 0) for w in weights.values())
        if excess < 1e-10:
            break
        uncapped = {n: w for n, w in weights.items() if w <= w_max}
        capped = {n: w_max for n, w in weights.items() if w > w_max}
        uncapped_total = sum(uncapped.values())
        if uncapped_total > 0:
            scale = (uncapped_total + excess) / uncapped_total
            uncapped = {n: w * scale for n, w in uncapped.items()}
        weights = {**uncapped, **capped}

    return weights
```

### Sweep

Test `w_max` in {0.05, 0.10, 0.15, 0.20, 0.50, 1.0} (1.0 = no cap, baseline).

### Expected Outcome

- With `w_max = 0.10`, Stacked Ridge Meta (currently 0.043-0.390 depending on
  iteration) would be capped in high-weight iterations, reducing the amplitude
  of the Phase A jump
- The Lasso cluster (0.062 each, 36 strategies) would be less affected since
  individual weights are already below the cap — but their *collective* weight
  would still dominate
- Net effect: moderate reduction in oscillation, not elimination

---

## Experiment 2: Weighted Median Market Price

### Hypothesis

Replacing the weighted mean with the weighted median of strategy forecasts
will make the market price robust to outlier forecasts on extreme days.

### Design

For each day t, instead of:
```
P^m_t = sum_i w_i * forecast_{i,t}
```

Compute:
```
P^m_t = weighted_median({forecast_{i,t}}, {w_i})
```

where the weighted median is the value m such that:
```
sum_{i: forecast_i <= m} w_i >= 0.5  and  sum_{i: forecast_i >= m} w_i >= 0.5
```

### Implementation Sketch

```python
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of values."""
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumulative, 0.5 * cumulative[-1])
    return float(sorted_values[median_idx])

def compute_market_prices_median(
    weights: dict[str, float],
    strategy_forecasts: dict[str, dict],
    current_market_prices: pd.Series,
) -> pd.Series:
    """Market price as weighted median of forecasts."""
    new_prices = current_market_prices.copy().astype(float)
    for t in current_market_prices.index:
        vals, wts = [], []
        for name, w in weights.items():
            if w <= 0:
                continue
            f = strategy_forecasts.get(name, {}).get(t)
            if f is not None:
                vals.append(float(f))
                wts.append(w)
        if vals:
            new_prices.loc[t] = weighted_median(np.array(vals), np.array(wts))
    return new_prices
```

### Expected Outcome

- On calm days: weighted median ~ weighted mean (strategies agree, no effect)
- On volatile days: weighted median ignores the extreme tails, producing a
  more conservative price that is less likely to overshoot
- Dec 15, 2024 example: if half the strategies forecast ~45 and half ~103,
  the median would land at the boundary between clusters rather than the
  profit-weighted extreme
- May break the limit cycle by making the update map *less sensitive* to
  which cluster dominates

### Spec Deviation

This is a more significant deviation from the spec than dampening.  The spec
explicitly defines `P^m = sum w_i * forecast_i` (a weighted mean).  The
weighted median preserves the spirit (profitable strategies determine price)
but changes the aggregation function.

---

## Experiment 3: Log-Profit Weighting

### Hypothesis

Using `log(profit)` instead of raw profit for weight computation will compress
the profit scale and reduce the sensitivity to large profit differentials.

### Design

```python
def compute_weights_log(strategy_profits: dict[str, float]) -> dict[str, float]:
    """Weights proportional to log(1 + profit) for profitable strategies."""
    raw = {}
    for name, profit in strategy_profits.items():
        if profit > 0:
            raw[name] = np.log1p(profit)
        else:
            raw[name] = 0.0
    total = sum(raw.values())
    if total == 0.0:
        return {name: 0.0 for name in raw}
    return {name: w / total for name, w in raw.items()}
```

### Expected Outcome

- At iter 14 (Lasso dominant): profit 6,363 -> log(6364) = 8.76; profit 968 -> log(969) = 6.88
- Weight ratio changes from 6.6:1 to 1.27:1 — dramatically more even
- This should reduce the amplitude of weight swings between iterations
- Combined with a weight cap, could substantially reduce oscillation

---

## Experiment 4: Cluster-Aware Cross-Pole Averaging

### Hypothesis

If strategies cluster into two forecast poles, the market price should be a
balanced blend of both poles rather than letting one pole dominate.

### Design

1. For each day, compute the forecast distribution across profitable strategies
2. Detect bimodality (e.g. using a gap > 20 EUR/MWh between the two densest
   forecast regions)
3. If bimodal: compute the weighted mean within each pole, then blend the two
   pole means 50/50 (or proportional to pole total profit)
4. If unimodal: use standard weighted mean

### Implementation Sketch

```python
def detect_bimodal_clusters(forecasts: list[float], gap_threshold: float = 20.0):
    """Detect two forecast clusters using sorted gap analysis."""
    sorted_f = sorted(forecasts)
    max_gap_idx = 0
    max_gap = 0
    for i in range(len(sorted_f) - 1):
        gap = sorted_f[i + 1] - sorted_f[i]
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i
    if max_gap >= gap_threshold:
        return sorted_f[:max_gap_idx + 1], sorted_f[max_gap_idx + 1:]
    return None  # unimodal
```

### Expected Outcome

- On volatile days with bimodal forecasts: price is pulled toward the midpoint
  of the two clusters, rather than alternating between them
- On calm days with unimodal forecasts: no change
- This is the most "creative" solution — it directly addresses the root cause
  (two forecast poles) rather than treating the symptoms (oscillation)

### Risks

- Bimodality detection is sensitive to the gap threshold
- If three or more clusters exist, the binary split is insufficient
- May underperform weighted mean on days where one cluster is clearly more accurate

---

## Combined Experiment: Log-Profit + Weight Cap

The most promising combination may be:
1. Log-profit weighting (compress profit scale)
2. Per-strategy cap at 0.10 (prevent single-strategy dominance)
3. Standard weighted mean (preserve spec compatibility for aggregation)

This combination attacks both the weight-swing problem (log compression) and
the single-strategy dominance problem (cap), while maintaining the spec's
aggregation function.

---

## Success Criteria

1. At least one weighting reform reduces the final convergence delta by >50%
2. The reform does not degrade market MAE vs the baseline engine
3. The reform generalises to different strategy sets
4. Combined with dampening (8b), achieves full convergence

## Files to Modify

- `src/energy_modelling/backtest/futures_market_engine.py` — add alternative weighting functions
- `tests/backtest/test_futures_market_engine.py` — test new weighting schemes
- `scripts/sweep_weighting.py` — new script for weighting experiments
