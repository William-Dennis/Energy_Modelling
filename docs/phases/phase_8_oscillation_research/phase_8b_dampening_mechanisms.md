# Phase 8b: Dampening Mechanisms

## Status: IMPLEMENTED

## Objective

Investigate dampening-based approaches to break the oscillation limit cycle,
ranging from simple exponential dampening to adaptive and two-phase schemes.

## Motivation

The oscillation is fundamentally a gain-of-1.0 feedback loop.  In control theory,
the standard remedy is to reduce the loop gain below the critical threshold.
For our system, this means blending the new weighted-average forecast with the
previous market price, rather than jumping directly.

The current spec (Step 4) defines:
```
P^m_{t}^(k+1) = sum_i w_i * forecast_{i,t}
```

A dampened variant would be:
```
P^m_{t}^(k+1) = alpha * (sum_i w_i * forecast_{i,t}) + (1 - alpha) * P^m_{t}^(k)
```

where `alpha in (0, 1]` is the dampening factor (alpha = 1.0 recovers the spec).

---

## Experiment 1: Fixed Dampening Factor

### Hypothesis

A fixed alpha in [0.3, 0.7] will convert the limit cycle into a convergent
spiral, achieving convergence within 20-50 iterations.

### Design

Sweep alpha over {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} for both
2024 and 2025 data.  For each alpha, record:

1. Whether the market converges (delta < 0.01 EUR/MWh)
2. Number of iterations to convergence (or delta at iter 50)
3. Final market MAE vs real prices
4. Final market MAE vs iter-0 MAE (does dampening preserve or degrade accuracy?)

### Implementation Sketch

```python
def compute_market_prices_dampened(
    weights: dict[str, float],
    strategy_forecasts: dict[str, dict],
    current_market_prices: pd.Series,
    alpha: float = 1.0,  # 1.0 = no dampening (spec-compliant)
) -> pd.Series:
    """Dampened price update: blend new forecast with previous price."""
    undampened = _compute_undampened(weights, strategy_forecasts, current_market_prices)
    return alpha * undampened + (1 - alpha) * current_market_prices
```

### Expected Outcome

- alpha = 0.3-0.5: convergence likely within 30-50 iterations
- alpha < 0.2: convergence guaranteed but slow (>100 iterations), may "average out" useful signal
- alpha > 0.7: may still oscillate (gain too high for the 81 EUR/MWh cycle amplitude)
- Optimal alpha: likely 0.3-0.5 based on the 3:1 ratio of cycle amplitude to baseline MAE

### Evaluation Criteria

- **Primary**: Does it converge? (binary)
- **Secondary**: MAE(converged) vs MAE(iter 0) — convergence must not degrade accuracy
- **Tertiary**: Iterations to converge — fewer is better for computational cost

---

## Experiment 2: Adaptive Dampening

### Hypothesis

An alpha that starts high (aggressive) and decreases as the price stabilises
can achieve both fast initial correction and stable convergence.

### Design

```
alpha_k = alpha_max * gamma^k
```

where `alpha_max` is the initial step size (e.g. 0.8) and `gamma in (0.9, 0.99)`
is the decay rate.  Alternatively, adapt alpha based on the observed delta:

```
alpha_k = min(alpha_max, target_delta / observed_delta_k)
```

This "proportional control" scheme reduces the step size when the oscillation
is large and increases it when the system is nearly converged.

### Implementation Sketch

```python
def adaptive_alpha(
    delta: float,
    target_delta: float = 1.0,
    alpha_max: float = 0.8,
    alpha_min: float = 0.1,
) -> float:
    """Compute dampening factor proportional to convergence gap."""
    if delta <= 0:
        return alpha_max
    alpha = min(alpha_max, target_delta / delta)
    return max(alpha_min, alpha)
```

### Expected Outcome

- Fast initial correction (high alpha when prices are far from equilibrium)
- Smooth convergence as delta shrinks
- Avoids the "too slow" problem of low fixed alpha

---

## Experiment 3: Two-Phase Convergence

### Hypothesis

Run dampened iterations to get near the fixed point, then switch to undampened
iterations to find the exact spec-compliant equilibrium (if one exists).

### Design

Phase 1 (exploration): alpha = 0.3, max 30 iterations, threshold = 1.0 EUR/MWh
Phase 2 (refinement): alpha = 1.0 (spec), max 20 iterations, threshold = 0.01 EUR/MWh

If Phase 2 diverges (delta increases for 3 consecutive iterations), fall back to
the Phase 1 result.

### Implementation Sketch

```python
def run_two_phase_market(
    initial_market_prices, real_prices, strategy_forecasts,
    phase1_alpha=0.3, phase1_max_iter=30, phase1_threshold=1.0,
    phase2_max_iter=20, phase2_threshold=0.01,
):
    # Phase 1: dampened approach to near-equilibrium
    eq1 = run_futures_market(
        ..., alpha=phase1_alpha, max_iterations=phase1_max_iter,
        convergence_threshold=phase1_threshold,
    )
    # Phase 2: undampened refinement from Phase 1 endpoint
    eq2 = run_futures_market(
        initial_market_prices=eq1.final_market_prices,
        ..., alpha=1.0, max_iterations=phase2_max_iter,
        convergence_threshold=phase2_threshold,
    )
    if eq2.converged:
        return eq2
    # Phase 2 diverged — check if it oscillated
    if eq2.convergence_delta > eq1.convergence_delta:
        return eq1  # dampened result was better
    return eq2
```

### Expected Outcome

- If a true fixed point exists near the dampened equilibrium, Phase 2 will find it
- If no fixed point exists (the limit cycle is intrinsic), Phase 2 will re-diverge,
  and we fall back to the dampened Phase 1 result
- This preserves spec compliance for the final price when possible

### Key Question

Does the system *have* a fixed point?  If F_A and F_B are sufficiently divergent,
the undampened map may have no fixed point at all — only the limit cycle.  In that
case, Two-Phase will always fall back to Phase 1, and dampening becomes the
permanent solution.

---

## Spec Compatibility

The spec defines `P^m = sum w_i * forecast_i` without dampening.  Introducing
dampening requires either:

1. **Spec amendment**: Add an optional `alpha` parameter to Step 4
2. **Wrapper approach**: Apply dampening as a post-processing step *outside* the
   core engine, treating the engine as a black box
3. **Convergence-only dampening**: Use dampening during the iteration phase only;
   the final reported price is still the weighted-average forecast (spec-compliant)

Option 3 is recommended: the dampening is a *solver technique* (like learning rate
in gradient descent), not a change to the economic model.  The equilibrium price
is still the spec-defined weighted average — we're just using a better numerical
method to find it.

---

## Success Criteria

1. At least one dampening scheme achieves convergence (delta < 0.01) on both 2024 and 2025
2. Converged MAE is no worse than iter-0 MAE (15.10 EUR/MWh for 2024)
3. The number of iterations is practical (< 100)
4. The approach generalises — it should work for any reasonable strategy set, not just the current 67

## Files to Modify

- `src/energy_modelling/backtest/futures_market_engine.py` — add `alpha` parameter to `compute_market_prices` and `run_futures_market`
- `tests/backtest/test_futures_market_engine.py` — add tests for dampened convergence
- `scripts/sweep_dampening.py` — new script for the alpha sweep experiment
