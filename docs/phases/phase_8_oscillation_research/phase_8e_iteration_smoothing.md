# Phase 8e: Iteration-Level Smoothing

## Status: IMPLEMENTED

## Objective

Investigate smoothing mechanisms applied at the iteration level — techniques
that blend information across multiple iterations rather than modifying the
single-iteration update rule (dampening) or the weighting scheme.

## Motivation

Dampening (Phase 8b) modifies the update rule within each iteration.
Weighting reforms (Phase 8c) change how profits map to weights.  This phase
explores a third axis: using the *history* of iteration outputs to produce a
more stable final price.

The key insight is that even in the oscillating regime, the market price
*oscillates around a central value* that may be a good consensus estimate.
The 2024 3-step cycle alternates between ~25 and ~81 EUR/MWh deltas, but
the *average* of prices across the cycle may be close to the true equilibrium.

---

## Experiment 1: Running Average of Iteration Prices

### Hypothesis

The arithmetic mean of market prices across the last K iterations will converge
even when individual iterations oscillate, because the oscillation is symmetric
around the equilibrium.

### Design

After running N iterations of the standard (undampened) engine, compute:
```
P_final_t = (1/K) * sum_{k=N-K+1}^{N} P^m_{t}^(k)
```

Sweep K over {2, 3, 5, 10, last_half}.

### Implementation Sketch

```python
def average_last_k_iterations(
    equilibrium: FuturesMarketEquilibrium,
    k: int,
) -> pd.Series:
    """Average market prices over last K iterations."""
    iters = equilibrium.iterations
    if k > len(iters):
        k = len(iters)
    last_k = iters[-k:]
    prices = pd.DataFrame({
        f"iter_{it.iteration}": it.market_prices for it in last_k
    })
    return prices.mean(axis=1)
```

### Expected Outcome

For the 2024 3-step cycle (iters 17-19):
- Iter 17 prices (Lasso dominant): closer to real on volatile days
- Iter 18 prices (Lasso settled): even closer
- Iter 19 prices (Stacked Ridge dominant): overshoots the other way

Averaging iters 17-19 should cancel out the oscillation on each day,
producing a price close to the midpoint of the two forecast poles.

For K = 3 (one full cycle), this should be optimal since the cycle has period 3.

### Risks

- If the oscillation is asymmetric (more time at one pole), the average will be
  biased toward that pole
- Does not help with the iteration *process* — the engine still oscillates internally
- This is a post-processing technique, not a convergence improvement

---

## Experiment 2: Exponential Moving Average of Iterations

### Hypothesis

An EMA across iterations gives more weight to recent iterations (which may be
closer to equilibrium after initial transients) while still smoothing oscillations.

### Design

```
EMA_t^(k) = beta * P^m_t^(k) + (1 - beta) * EMA_t^(k-1)
```

with EMA_t^(0) = P^m_t^(0).  The final price is EMA after the last iteration.

Sweep beta over {0.1, 0.2, 0.3, 0.5}.

### Implementation Sketch

```python
def ema_iteration_prices(
    equilibrium: FuturesMarketEquilibrium,
    beta: float = 0.3,
) -> pd.Series:
    """Exponential moving average across iterations."""
    iters = equilibrium.iterations
    ema = iters[0].market_prices.copy().astype(float)
    for it in iters[1:]:
        ema = beta * it.market_prices + (1 - beta) * ema
    return ema
```

### Expected Outcome

- beta = 0.3: heavy smoothing, strongly dampens oscillation
- beta = 0.5: moderate, each iteration contributes half
- For the 3-step cycle: EMA will converge to a stable value after ~10 iterations
  regardless of whether the underlying engine converges

---

## Experiment 3: Best-Iteration Selection

### Hypothesis

Instead of using the final iteration, select the iteration with the lowest
convergence delta (smallest max price change from previous iteration) as the
output.

### Design

```python
def best_iteration_prices(
    equilibrium: FuturesMarketEquilibrium,
) -> tuple[pd.Series, int]:
    """Select iteration with lowest delta."""
    best_iter = None
    best_delta = float("inf")
    prev_prices = None
    for it in equilibrium.iterations:
        if prev_prices is not None:
            delta = (it.market_prices - prev_prices).abs().max()
            if delta < best_delta:
                best_delta = delta
                best_iter = it
        prev_prices = it.market_prices
    return best_iter.market_prices, best_iter.iteration
```

### Expected Outcome

For 2024, the best iteration is likely iter 6 (delta = 16.76) or iter 9/12/15/18
(delta = 25.5).  These "settle" phases represent the moment when the Lasso
cluster has corrected the Stacked Ridge overshoot but before the next cycle begins.

This is a simple heuristic that doesn't require any engine changes.

---

## Experiment 4: Iteration-Fed Dampening (Hybrid)

### Hypothesis

Use the undampened engine to generate a full iteration trace, then use the
trace to compute a dampened-equivalent price without modifying the engine.

### Design

This is conceptually equivalent to Experiment 1 but weighted by iteration
quality:

```
P_final_t = sum_k alpha_k * P^m_t^(k) / sum_k alpha_k
```

where `alpha_k = 1 / (1 + delta_k)` — iterations with lower delta get higher
weight.

### Implementation Sketch

```python
def delta_weighted_average(
    equilibrium: FuturesMarketEquilibrium,
) -> pd.Series:
    """Average iteration prices, weighted by inverse delta."""
    iters = equilibrium.iterations
    prices_list = []
    weights_list = []
    prev = None
    for it in iters:
        if prev is not None:
            delta = float((it.market_prices - prev).abs().max())
            w = 1.0 / (1.0 + delta)
        else:
            w = 1.0  # first iteration gets weight 1
        prices_list.append(it.market_prices)
        weights_list.append(w)
        prev = it.market_prices

    total_w = sum(weights_list)
    result = sum(w * p for w, p in zip(weights_list, prices_list)) / total_w
    return result
```

### Expected Outcome

- Low-delta iterations (settle phases) get high weight
- High-delta iterations (jump phases) get low weight
- Result should be close to the "settle" price, which is empirically the most
  accurate phase of the cycle

---

## Comparison Matrix

| Method | Engine Modification | Post-Processing | Cycle-Aware | Hyperparams |
|--------|--------------------|--------------------|-------------|-------------|
| Running average | No | Yes | Optimal at K = cycle period | K |
| EMA | No | Yes | Implicitly (via smoothing) | beta |
| Best iteration | No | Yes | Selects settle phase | None |
| Delta-weighted avg | No | Yes | Yes (weights by stability) | None |

All methods in this phase are **post-processing** — they do not modify the
engine's iteration loop.  This means they are fully spec-compatible and can
be applied on top of existing market results.

---

## Interaction with Other Phases

- **With dampening (8b)**: If dampening achieves convergence, iteration smoothing
  is unnecessary (converged = single stable price)
- **Without dampening**: Iteration smoothing is the best available technique to
  extract a stable price from an oscillating engine
- **With weighting reform (8c)**: If weighting reform reduces oscillation amplitude,
  iteration smoothing becomes more effective (averaging over smaller oscillations)

The recommended investigation order is:
1. First try dampening (8b) alone
2. If dampening is insufficient or undesirable, try iteration smoothing (8e)
3. If needed, combine dampening + smoothing

---

## Success Criteria

1. At least one smoothing method produces a final MAE better than iter-0 (15.10 for 2024)
2. The smoothed price is stable (applying one more iteration of smoothing does not change it significantly)
3. The method is deterministic and reproducible

## Files to Modify

- `src/energy_modelling/backtest/convergence.py` — add smoothing functions
- `tests/backtest/test_convergence.py` — test smoothing methods
- `scripts/evaluate_smoothing.py` — new script for smoothing experiments
