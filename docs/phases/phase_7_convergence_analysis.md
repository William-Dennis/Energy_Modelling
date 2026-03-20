# Phase 7: Perfect Foresight Convergence Analysis

## Status: IN PROGRESS

## Objective

Formally analyze whether a perfect-foresight strategy asymptotically dominates all
other strategies under the iterative market weighting scheme, driving the market
price P_t^m to converge to the real settlement price P_t.

This is a redo of the Phase 7 analysis following the forecast-first strategy
refactor. All strategies now produce real-valued price forecasts (not binary
directions), and the market engine uses these forecasts directly in the
weighted-average price update. The analysis is rebuilt from scratch: theory
first, then empirical validation.

## Prerequisites

- Phase 4 complete (strategies implemented) ✅
- Phase 5 complete (market simulation results available) ✅
- Forecast-first strategy refactor complete ✅

## Checklist

### 7a. Theoretical analysis
- [x] Formal statement of the convergence claim
- [x] Define the iterative update rule mathematically (both forecast and legacy modes)
- [x] Prove or disprove: does adding a perfect foresight strategy guarantee convergence?
- [x] Characterize the fixed point(s) of the iteration
- [x] Analyze the dampening parameter's effect on convergence rate

### 7b. Empirical validation of theory
- [ ] Experiment 1: Validate Theorem 1 (geometric convergence rate)
- [ ] Experiment 2: Validate Theorem 2 (legacy oscillation, unchanged)
- [ ] Experiment 3: Validate Theorem 3 (iteration count formula)
- [ ] Experiment 4: Validate Theorem 4 (mixed-strategy equilibrium)
- [ ] Experiment 5: Validate convergence trajectory matches theory exactly

### 7c. Edge cases
- [ ] Legacy mode still oscillates (unchanged behaviour confirmed)
- [ ] PF weight dynamics in mixed markets
- [ ] Dampening α → 0 and α → 1 boundary behaviour

### 7d. Document findings
- [ ] Theoretical result written up
- [ ] Empirical evidence compiled
- [ ] Design implications stated

---

## Strategy Interface (Post-Refactor)

Strategies implement a single abstract method:

```python
@abstractmethod
def forecast(self, state: BacktestState) -> float:
    """Return an explicit price forecast for the delivery date."""
```

The trading direction is derived automatically in the base class:

```python
def act(self, state: BacktestState) -> int | None:
    price_forecast = self.forecast(state)
    entry = state.last_settlement_price
    if price_forecast > entry + self.skip_buffer:
        return 1   # long
    if price_forecast < entry - self.skip_buffer:
        return -1  # short
    return None     # skip (dead zone)
```

Perfect foresight provides `forecast() = P_real` (the actual settlement price).

---

## Theoretical Analysis

### Market Model

The market engine runs iteratively. At each iteration k:

1. **Profit**: `π_i = Σ_t d_{i,t} · (P_{real,t} − P^m_{k,t}) · 24`
   where `d_{i,t} = sign(P̂_{i,t} − P^m_{k,t})` is derived from the forecast.
2. **Weight**: `w_i = max(0, π_i) / Σ_j max(0, π_j)`
3. **Price update**: `P^{raw}_t = Σ_i w_i · P̂_{i,t}` (weighted average of forecasts)
4. **Dampening**: `P^m_{k+1,t} = α · P^{raw}_t + (1−α) · P^m_{k,t}`
5. **Convergence**: stop when `max_t |P^m_{k+1,t} − P^m_{k,t}| < ε`

### Theorem 1: Forecast-Based PF Convergence

**Claim**: When PF is the sole strategy with `P̂_PF = P_real`, the market
price converges to `P_real` with geometric rate `(1 − α)`.

**Proof**:

*Step 1 — PF is always profitable when P^m ≠ P_real.*
PF direction is `d = sign(P_real − P^m_k)`. Its daily profit is:
```
r_t = d_t · (P_{real,t} − P^m_{k,t}) · 24 = |P_{real,t} − P^m_{k,t}| · 24 ≥ 0
```
The total profit `π_PF = Σ_t r_t > 0` whenever at least one day has `P^m ≠ P_real`.
Therefore `w_PF = 1.0`.

*Step 2 — raw price equals real price.*
With only PF active: `P^{raw}_t = w_PF · P̂_{PF,t} = 1.0 · P_{real,t} = P_{real,t}`.

*Step 3 — dampened update is a contraction.*
```
P^m_{k+1,t} = α · P_{real,t} + (1 − α) · P^m_{k,t}
```
Define the error `e_{k,t} = P^m_{k,t} − P_{real,t}`:
```
e_{k+1,t} = (1 − α) · e_{k,t}
```
Therefore `|e_{k+1,t}| = (1 − α) · |e_{k,t}|`.

For `α ∈ (0, 1]`, the mapping is a contraction with rate `(1 − α) < 1`.
By the Banach fixed-point theorem, `e_{k,t} → 0` for all days t.

*Step 4 — RMSE decay.*
```
RMSE_k = √(Σ_t e²_{k,t} / N) = √(Σ_t (1−α)^{2k} · e²_{0,t} / N)
       = (1−α)^k · RMSE_0
```

The RMSE decays geometrically with rate `(1 − α)`. ∎

**Corollary — iteration count**: Convergence to threshold ε requires:
```
k* = ⌈ log(ε / D_max) / log(1 − α) ⌉
```
where `D_max = max_t |P_{real,t} − P_{0,t}|`.

*Testable predictions*:
1. `RMSE_k = RMSE_0 · (1 − α)^k` — exact at every iteration
2. Convergence in `⌈log(ε/D_max) / log(1−α)⌉` iterations
3. All dampening values α ∈ (0, 1] converge; higher α → fewer iterations

### Theorem 2: Legacy Direction ± Spread Non-Convergence

**Claim**: With legacy synthesis `P̂_i = P^m_k ± S` (constant spread), adaptive
PF oscillates when `D/(α·S)` is not an integer.

**Proof**: The step size is `Δ = α · S`, independent of the distance to P_real.
After first overshoot, the market alternates between `P_real + bias` and
`P_real − (Δ − bias)`, where `bias = ⌈D/Δ⌉ · Δ − D`.

The oscillation amplitude is Δ and persists indefinitely because the step
size is constant — this is NOT a contraction mapping.

*Testable predictions*:
1. Legacy mode never converges (delta = α·S at every iteration)
2. Final RMSE ≈ `α·S/√3`
3. Larger spread → larger oscillation amplitude

### Theorem 3: Mixed-Strategy Equilibrium

**Claim**: When PF competes with other strategies that produce forecasts
`{P̂_i}`, the converged market price satisfies:
```
P* = Σ_i w_i(P*) · P̂_i
```
where `w_i(P*)` are the equilibrium weights. If PF has weight `w_PF`, then:
```
P* = w_PF · P_real + (1 − w_PF) · P̄_others
```
where `P̄_others` is the weighted average of other strategies' forecasts.

**Proof**: At a fixed point, the raw price equals the current price
(dampening becomes irrelevant). The weighted average of forecasts must
equal the market price. Since PF's forecast is P_real:
```
P* = w_PF · P_real + Σ_{i≠PF} w_i · P̂_i
```
The RMSE at convergence is:
```
RMSE* = (1 − w_PF) · |P̄_others − P_real|  (per-day)
```

*Testable predictions*:
1. Higher PF weight → lower RMSE
2. PF weight determined by relative profitability against other strategies
3. Market price lies between P_real and the non-PF consensus

### Theorem 4: Resolution of the Constant Step Size Limitation

**Claim**: The forecast mechanism creates distance-proportional steps,
resolving the fundamental limitation of the legacy engine.

**Proof**: In forecast mode, PF's effective contribution to the price
update is `α · (P_real − P^m_k)`. This step is proportional to the error
`e_k`, creating a contraction. In legacy mode, the step is `α · S`
(constant), creating oscillation.

*Key insight*: The `forecast()` abstraction replaces the constant spread S
with strategy-dependent price predictions. For PF, this is equivalent to
an adaptive spread `S_k = |P_real − P^m_k|` that automatically shrinks
as the market approaches truth.

---

## Empirical Validation

*To be populated by running experiments that test each theorem's predictions.*

### Experiment 1: Theorem 1 — RMSE Trajectory (PF Only, Forecast Mode)

*Validates: `RMSE_k = RMSE_0 · (1 − α)^k` at every iteration.*

| Iteration | Market RMSE | Theory RMSE | Match? |
|-----------|------------|-------------|--------|
| | | | |

### Experiment 2: Theorem 1 — Iteration Count vs Dampening

*Validates: `k* = ⌈log(ε/D_max) / log(1−α)⌉` for various α.*

| α | Observed iters | Theory iters | Final RMSE |
|---|----------------|--------------|------------|
| | | | |

### Experiment 3: Theorem 2 — Legacy Mode Oscillation

*Validates: legacy adaptive PF oscillates with delta = α·S.*

| Spread | Converged | Iterations | Delta | Final RMSE | Theory RMSE |
|--------|-----------|------------|-------|------------|-------------|
| | | | | | |

### Experiment 4: Theorem 3 — Mixed Strategy Equilibrium

*Validates: market price is weighted average of surviving forecasts.*

| Configuration | Converged | RMSE | PF Weight |
|---------------|-----------|------|-----------|
| | | | |

### Experiment 5: Theorem 4 — Forecast vs Legacy Side-by-Side

*Validates: forecast mode converges where legacy mode oscillates.*

| Mode | Converged | Iterations | Final RMSE |
|------|-----------|------------|------------|
| | | | |

---

## Summary of Answers to Key Questions

*To be populated after empirical validation.*

---

## Code Deliverables

### Modified files
- `src/energy_modelling/backtest/types.py` — `forecast()` is now the single
  abstract method; `act()` derives direction with `skip_buffer` dead zone
- `strategies/*.py` — All 10 strategies refactored to forecast-first design
- `src/energy_modelling/backtest/convergence.py` — Added `run_forecast_foresight_market()`

### Existing files (unchanged)
- `strategies/perfect_foresight.py` — PerfectForesightStrategy
  (`forecast()` returns the actual settlement price)

### New functions in convergence.py
- `run_forecast_foresight_market()` — Forecast-aware PF convergence analysis
- `_run_forecast_iteration()` — Single iteration with real-valued forecasts
- `_build_pf_forecasts()` — Build PF forecast dict from real prices

### Tests
- `tests/backtest/test_convergence.py` — 36 tests (29 existing + 7 new)
  - `TestForecastForesightMarket` — 7 tests for forecast-based convergence

All 225 tests pass.
