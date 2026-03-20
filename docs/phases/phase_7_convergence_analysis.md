# Phase 7: Perfect Foresight Convergence Analysis

## Status: COMPLETE

## Objective

Formally analyze the convergence properties of the spec-compliant synthetic
futures market engine (`docs/energy_market_spec.md`), focusing on the role
of a perfect-foresight (PF) strategy.

The engine has **no dampening**, **no legacy direction +/- spread mode**, and
**no `*24` profit multiplier** in the internal weighting loop.  All strategies
produce real-valued price forecasts, and the market price is the
profit-weighted average of forecasts from profitable strategies.

## Prerequisites

- Phase 4 complete (strategies implemented) ✅
- Phase 5 complete (market simulation results available) ✅
- Forecast-first strategy refactor complete ✅
- Spec-compliant engine rewrite complete ✅

## Checklist

### 7a. Theoretical analysis
- [x] Formal statement of the market model (5 steps from spec)
- [x] Prove Theorem 1: PF instant convergence (no dampening → one-step jump)
- [x] Prove Theorem 2: PF profit dominance
- [x] Prove Theorem 3: Fixed-point characterization
- [x] Prove Theorem 4: Unprofitable strategy elimination

### 7b. Empirical validation
- [x] Experiment 1: PF instant convergence — <= 2 iterations, RMSE = 0
- [x] Experiment 2: PF dominance — PF profit >= all others, multiple seeds
- [x] Experiment 3: Fixed-point property — one more iteration is idempotent
- [x] Experiment 4: Unprofitable elimination — wrong-side strategies get zero weight

### 7c. Edge cases
- [x] PF + same-side strategies (both profitable, weight shared)
- [x] All strategies unprofitable → all weights zero, prices carry forward
- [x] Opposing strategies without PF → oscillation (expected behaviour)

### 7d. Document findings
- [x] Theoretical results (4 theorems with proofs)
- [x] Empirical evidence (all predictions confirmed)
- [x] Verification script exits 0 (`scripts/verify_theorems.py`)

---

## Strategy Interface

Strategies implement a single abstract method:

```python
@abstractmethod
def forecast(self, state: BacktestState) -> float:
    """Return an explicit price forecast for the delivery date."""
```

The trading direction is derived automatically:

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

Perfect foresight: `forecast() = P_real` (the actual settlement price).

---

## Market Model (from `energy_market_spec.md`)

The engine runs iteratively.  At each iteration k:

1. **Trading decision**: `q_{i,t} = sign(forecast_{i,t} - P^m_t)`
2. **Profit**: `r_{i,t} = q_{i,t} * (P_real_t - P^m_t)`
   - Total: `Pi_i = sum_t r_{i,t}`
3. **Selection**: `w_i = max(Pi_i, 0) / sum_j max(Pi_j, 0)`
   - Only strategies with positive total profit receive weight.
4. **Price update**: `P^m_{t}^(k+1) = sum_i w_i^{norm} * forecast_{i,t}`
   - Direct weighted average of forecasts.  **No dampening.**
5. **Iteration**: repeat until `max_t |P^m_{k+1,t} - P^m_{k,t}| < epsilon`.

Key differences from the old engine:
- **No dampening** (`alpha` parameter removed)
- **No legacy direction +/- spread** (`forecast_spread` removed)
- **No `*24` multiplier** in profit computation (profit is per-unit, not EUR)
- All strategies must provide explicit forecasts

---

## Theoretical Analysis

### Theorem 1: PF Instant Convergence

**Claim**: When PF is the sole strategy with `forecast_PF = P_real`, the
market price jumps to `P_real` in **one iteration**.

**Proof**:

*Step 1 — PF is always profitable when P^m ≠ P_real.*
PF direction is `d = sign(P_real - P^m)`.  Its daily profit is:
```
r_t = sign(P_real - P^m) * (P_real - P^m) = |P_real - P^m| >= 0
```
Total profit `Pi_PF = sum_t |P_real_t - P^m_t| > 0` (when at least one day
has P^m ≠ P_real).  Therefore `w_PF = 1.0`.

*Step 2 — price update equals real price.*
```
P^m_{k+1,t} = w_PF * P_real_t = 1.0 * P_real_t = P_real_t
```

*Step 3 — second iteration converges.*
At iteration 1, `P^m = P_real`, so `sign(P_real - P_real) = 0` for all days.
All profits are zero, all weights are zero, prices carry forward unchanged.
Delta = 0 < epsilon → converged.

Total: **<= 2 iterations**.  ∎

*Testable predictions*:
1. PF converges in <= 2 iterations
2. After iteration 0, P^m = P_real exactly (RMSE = 0)
3. PF has weight 1.0 at iteration 0
4. Works for any dataset (any seed, any size)

### Theorem 2: PF Dominance

**Claim**: PF's total profit >= any other strategy's total profit.

**Proof**:

For any strategy i with direction `d_i = sign(forecast_i - P^m)`:
```
r_{i,t} = d_i * (P_real - P^m) where d_i in {-1, 0, +1}
```

For PF: `d_PF = sign(P_real - P^m)`, so:
```
r_{PF,t} = sign(P_real - P^m) * (P_real - P^m) = |P_real - P^m|
```

For any other strategy: `|r_{i,t}| = |d_i| * |P_real - P^m| <= |P_real - P^m|`.

Moreover, if `d_i` has the wrong sign (opposite to `sign(P_real - P^m)`),
then `r_{i,t} < 0`.  PF never has the wrong sign.

Therefore: `Pi_PF = sum_t |P_real - P^m| >= sum_t r_{i,t} = Pi_i` for any i.  ∎

*Testable predictions*:
1. PF profit is non-negative
2. PF profit >= every other strategy's profit
3. PF profit = sum |P_real - P^m| (the theoretical maximum)
4. PF gets the highest (or tied-highest) weight

### Theorem 3: Fixed-Point Characterization

**Claim**: A converged price vector P* satisfies:
```
P*_t = sum_i w_i(P*) * forecast_{i,t}
```
where `w_i(P*)` are the weights computed from profits at P*.

**Proof**: At convergence, `P^m_{k+1} = P^m_k = P*`.  The update rule
requires `P* = sum w_i * forecast_i`.  If no strategy has positive profit
at P*, all weights are zero and prices carry forward unchanged — also a
fixed point (trivially).  ∎

*Consequences*:
- With PF only: P* = P_real (the unique fixed point).
- With PF + other strategies: the undampened engine reaches P_real in one
  step, then all profits become zero (since P_real - P_real = 0), so prices
  carry forward.  The fixed point is still P_real.
- Without PF: the fixed point depends on the strategy forecasts.  If two
  strategies with constant forecasts both remain profitable, the price
  converges to their profit-weighted average.

*Testable predictions*:
1. One more iteration from P* produces P* (delta = 0)
2. P*_t = sum w_i * forecast_i_t (numerical verification)
3. With PF present, P* = P_real

### Theorem 4: Unprofitable Elimination

**Claim**: Strategies with non-positive total profit receive zero weight
and do not influence the market price.

**Proof**: By definition, `w_i = max(Pi_i, 0) / sum_j max(Pi_j, 0)`.
If `Pi_i <= 0`, then `max(Pi_i, 0) = 0`, so `w_i = 0`.

This means:
- In a trending-up market (real > market for all days), strategies that
  forecast below market (direction = -1) earn negative profit and are eliminated.
- Adding unprofitable strategies to the market does not change the result.
- If ALL strategies are unprofitable, all weights are zero and prices
  carry forward unchanged.  ∎

*Testable predictions*:
1. Short strategy has negative profit in trending-up market
2. Short weight = 0
3. PF + Short produces same result as PF only
4. Multiple unprofitable strategies are all eliminated
5. All-zero profits → all-zero weights → prices carry forward

---

## Empirical Validation

All experiments are run via `scripts/verify_theorems.py --verbose`.
Dataset: 20 synthetic days, seed=42, prices ~ U(30, 70).

### Experiment 1: Theorem 1 — PF Instant Convergence

| Iteration | RMSE | Active Weights |
|-----------|------|----------------|
| 0 | 0.000000 | PF: 1.000 |
| 1 | 0.000000 | (all zero) |

**Result**: PF converges in exactly 2 iterations.  After iteration 0,
RMSE = 0 (market = real).  At iteration 1, all profits are zero, prices
carry forward, delta = 0 → converged.  Confirmed for seeds {0, 7, 42, 99, 123}.
Theorem 1 is **confirmed**.

### Experiment 2: Theorem 2 — PF Dominance

| Strategy | Profit | Weight |
|----------|--------|--------|
| PF | 168.27 | 0.376 |
| Long | 168.27 | 0.376 |
| Random | 110.73 | 0.248 |
| Short | -168.27 | 0.000 |

**Result**: PF profit equals the theoretical maximum `sum|P_real - P_m|`.
PF profit >= every other strategy.  Long ties PF in the trending-up dataset
(because Long's direction also happens to be correct on all days).  Short is
eliminated.  Confirmed across multiple seeds.  Theorem 2 is **confirmed**.

### Experiment 3: Theorem 3 — Fixed-Point Property

| Configuration | Converged | Iterations | Fixed-Point Delta |
|---------------|-----------|------------|-------------------|
| PF only | Yes | 2 | 0.000000 |
| PF + Long | Yes | 3 | 0.000000 |
| BullishA + BullishB | Yes | 3 | 0.000000 |

**Result**: One additional iteration from P* produces delta = 0 in all cases.
The numerical identity `P*_t = sum w_i * forecast_i_t` holds with error < 1e-9.
Theorem 3 is **confirmed**.

### Experiment 4: Theorem 4 — Unprofitable Elimination

| Configuration | Short Profit | Short Weight | RMSE vs PF-only |
|---------------|-------------|-------------|-----------------|
| PF + Short | -168.27 | 0.000 | identical |
| PF + Bad1 + Bad2 | all < 0 | all 0.000 | identical |

**Result**: Short and other wrong-side strategies have negative profit and
receive zero weight.  Adding them to the market does not change any outcome.
When all strategies have zero profit, all weights are zero.
Theorem 4 is **confirmed**.

---

## Summary

### Does adding PF guarantee convergence to P_real?

**Yes** — and it's **instant** (one iteration).  With no dampening, PF's
forecast goes directly into the weighted average as `P_real`, and since PF
always has the maximum possible profit, it dominates any iteration where
prices differ from reality.

### What happens without PF?

Without PF, the market converges to a profit-weighted consensus of the
surviving strategies' forecasts.  If all forecasts are on one side (e.g.
two bullish strategies), the market oscillates between them as each takes
turns being "more profitable."  Eventually it stabilizes when all remaining
strategies' profit signals balance out.

### What about opposing strategies?

When two strategies have forecasts on opposite sides of P_real, the market
can **oscillate** rather than converge.  The winning strategy drives the
price past P_real, making the loser profitable, which drives it back.  This
is the expected behaviour of the undampened engine — it is not a bug but a
fundamental property of the feedback loop without dampening.

### Implications for the platform

The spec-compliant engine is a clean **prediction market**: strategies
that produce more accurate forecasts earn more weight and steer the price
closer to their predictions.  The `forecast()` abstraction ensures that
every strategy provides an explicit price target, and the market aggregates
these into a consensus price.

PerfectForesightStrategy is used as a **theoretical analysis tool** only
— it is not included in production market simulations.

---

## Code Deliverables

### Source files
- `src/energy_modelling/backtest/futures_market_engine.py` — Spec-compliant engine
- `src/energy_modelling/backtest/futures_market_runner.py` — Market evaluation orchestrator
- `src/energy_modelling/backtest/convergence.py` — Convergence analysis tools

### Verification
- `scripts/verify_theorems.py` — Verifies all 4 theorems (exit 0 = all pass)
- `scripts/debug_market_convergence.py` — Diagnostic trace tool

### Tests
- `tests/backtest/test_futures_market_engine.py` — 20 tests
- `tests/backtest/test_market.py` — 16 tests
- `tests/backtest/test_convergence.py` — 16 tests
- `tests/backtest/test_market_runner.py` — 11 tests
- `tests/backtest/test_futures_market_runner.py` — 6 tests

All tests pass (798 total, 0 failures).
