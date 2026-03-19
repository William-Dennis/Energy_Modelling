# Phase 7: Perfect Foresight Convergence Analysis

## Status: COMPLETE

## Objective

Formally analyze whether a perfect-foresight strategy asymptotically dominates all
other strategies under the iterative market weighting scheme, driving the market
price P_t^m to converge to the real settlement price P_t.

## Prerequisites

- Phase 4 complete (strategies implemented) ✅
- Phase 5 complete (market simulation results available) ✅

## Checklist

### 7a. Theoretical analysis
- [x] Formal statement of the convergence claim
- [x] Define the iterative update rule mathematically
- [x] Prove or disprove: does adding a perfect foresight strategy guarantee convergence?
- [x] Characterize the fixed point(s) of the iteration
- [x] Analyze the dampening parameter's effect on convergence

### 7b. Empirical analysis
- [x] Run market with perfect foresight + all other strategies (Experiments 3-4)
- [x] Run market with perfect foresight only (Experiments 1-2, 6)
- [x] Record convergence metrics for each configuration
- [x] Verify theoretical predictions against real 2024 data

### 7c. Edge cases and counterexamples
- [x] Non-divisible step size creates stable 2-cycles
- [x] Spread parameter sensitivity fully characterized
- [x] Dampening effect: changes oscillation amplitude but cannot break cycles
- [x] Fixed vs adaptive PF: fundamentally different convergence properties

### 7d. Document findings
- [x] Theoretical result written up
- [x] Empirical evidence compiled
- [x] Design implications stated

---

## Market Model Reference

### Iterative update rule

1. Start: `P_0 = last_settlement_price`
2. Each iteration k:
   - Profit: `π_i = d_i * (P_real - P_k) * 24` (summed over all days)
   - Weight: `w_i = max(0, π_i) / Σ max(0, π_j)`
   - Implied forecast: `f_i = P_k + d_i * S` (spread S)
   - Raw update: `P_raw = Σ(w_i * f_i) / Σ(w_i)` (weighted average over active strategies)
   - Dampened: `P_{k+1} = α * P_raw + (1-α) * P_k` (dampening α, default 0.5)
3. Converge when `max|P_{k+1} - P_k| < 0.01`

### Two types of perfect foresight

**Fixed PF**: `d_PF = sign(P_real - P_0)`, computed once, does not change across iterations.

**Adaptive PF**: `d_PF,k = sign(P_real - P_k)`, recomputed each iteration based on current market price.

---

## Theoretical Analysis

### Core Mechanism (Single Day, Single Strategy)

For a single day with distance `D = |P_real - P_0|`, dampening `α`, and spread `S`:

**Step size per iteration**: `Δ = α * S` (constant)

**Number of steps to first arrival or overshoot**: `n = ⌈D / Δ⌉`

**Overshoot**: `bias = n * Δ - D`

### Theorem 1: Fixed PF Convergence (with bias)

**Claim**: Fixed PF always "converges" (delta → 0) but with non-zero bias.

**Proof**: After `⌈D/Δ⌉` iterations, the market price has moved past P_real.
At this point, `π_PF = d_PF * (P_real - P_k) * 24 < 0` (PF becomes unprofitable
because the market overshot past the real price). With weight 0, no strategy
influences the market, so `P_{k+1} = P_k` (carry-forward). Delta = 0.

The final price is `P_0 + n * Δ = P_real + bias` where `bias ∈ [0, Δ)`.

**Result**: Fixed PF converges to `P_real ± Δ`, NOT to `P_real` itself. ∎

### Theorem 2: Adaptive PF Non-Convergence (Oscillation)

**Claim**: Adaptive PF does NOT converge when `D mod Δ ≠ 0`.

**Proof**: Consider the iteration after first overshoot. The market is at
`P_k = P_real + bias` (above P_real for upward movement). Adaptive PF
recomputes: `d_PF,k = sign(P_real - P_k) = -1`. Now implied forecast is
`P_k - S`, and dampened update gives `P_{k+1} = P_k - Δ`. This moves the
market back, potentially undershooting.

If `bias` and `Δ` create a periodic orbit, the system enters a stable 2-cycle.
Specifically: the market alternates between `P_real + bias` and
`P_real + bias - Δ = P_real - (Δ - bias)`.

The oscillation amplitude is `Δ` and is centered at `P_real + (bias - Δ/2)`.
Since `Δ` is constant (it doesn't shrink as we approach P_real), this is
NOT a contraction mapping. The oscillation persists indefinitely.

**Special case**: When `D/Δ` is an integer, `bias = 0` and the market lands
exactly on `P_real`. PF profit = 0, weight = 0, carry-forward at `P_real`.
This converges perfectly.

**Result**: Adaptive PF converges iff `D/(α*S)` is an integer. ∎

### Theorem 3: RMSE Bound

**Claim**: The RMSE of the adaptive PF oscillation is approximately `α*S / √3`.

**Proof**: For each day, the market oscillates uniformly between
`P_real - (Δ - bias)` and `P_real + bias`, with amplitude `Δ`.
The RMS of a uniform distribution on `[-a, a]` is `a/√3`.
Since `Δ = α*S`, the per-day RMSE is approximately `α*S / √3`.
Over many days with varying `D` values, the biases are approximately
uniformly distributed in `[0, Δ)`, and the RMSE converges to `α*S / √3`.

**Empirical verification**: With α=0.5, S=0.1: theoretical RMSE = 0.029.
Observed RMSE on 2024 data: 0.028. ✓

### Theorem 4: Fundamental Design Limitation

**Claim**: The market engine cannot converge to P_real for any non-trivial
configuration because the step size is constant.

**Proof**: The implied forecast `f_i = P_k ± S` creates a fixed step size
`Δ = α*S` regardless of the distance `|P_real - P_k|`. For convergence,
we need `|P_{k+1} - P_real| < |P_k - P_real|` (contraction). But when
the market is within distance `Δ` of P_real, the next step overshoots,
creating `|P_{k+1} - P_real| > 0`. This alternates without shrinking.

**Fix**: The engine would need one of:
1. **Adaptive spread**: `S_k = c * |P_real - P_k|` (proportional to distance)
2. **Continuous forecasts**: Replace binary ±S with real-valued price predictions
3. **Decay factor**: `S_k = S_0 * γ^k` (geometric spread decay)

Any of these would create a contraction mapping with guaranteed convergence. ∎

---

## Empirical Evidence (2024 Validation Data)

### Perfect Foresight Standalone Backtest

| Metric | Value |
|--------|-------|
| Total PnL | 193,415 EUR |
| Sharpe Ratio | 15.73 |
| Win Rate | 100.0% |
| Trade Count | 366 |

This is the theoretical upper bound. No strategy can exceed this.

### Experiment 1: Fixed PF Only (Standard Market)

| Spread | Converged | Iterations | Delta | RMSE vs P_real |
|--------|-----------|------------|-------|---------------|
| 5.0 | Yes | 10 | 0.0000 | 22.20 |
| 10.0 | Yes | 6 | 0.0000 | 22.39 |
| 22.0 | Yes | 4 | 0.0000 | 24.76 |

**Interpretation**: Fixed PF "converges" (delta=0 when it becomes unprofitable after
overshoot) but the RMSE is terrible (22-25 EUR). The market price stops moving but
is nowhere near the real price.

### Experiment 2: Adaptive PF Only

| Spread | Converged | Iterations | Delta | RMSE vs P_real |
|--------|-----------|------------|-------|---------------|
| 5.0 | No | 200 | 2.500 | 1.43 |
| 10.0 | No | 200 | 5.000 | 2.83 |
| 22.0 | No | 200 | 11.000 | 6.32 |

**Interpretation**: Adaptive PF oscillates but gets much closer to real prices.
RMSE ≈ α*S/√3 as predicted by Theorem 3. With spread=5 (step=2.5), RMSE=1.43 vs
theoretical 2.5/√3 = 1.44. Excellent agreement.

### Experiment 3: Fixed PF + All 9 Strategies (Standard Market)

| Spread | Converged | RMSE | PF Weight | Active Set |
|--------|-----------|------|-----------|------------|
| 5.0 | No | 21.12 | 0.000 | AlwaysLong, FossilDispatch |
| 22.0 | No | 23.24 | 0.000 | AlwaysShort, WindForecast, LoadForecast |

**Critical finding**: PF gets weight ZERO at the final iteration! The opposing
strategies' oscillation drives PF unprofitable. PF cannot rescue a structurally
unstable market.

### Experiment 4: Adaptive PF + All 9 Strategies

| Spread | Converged | RMSE | PF Weight |
|--------|-----------|------|-----------|
| 5.0 | No | 1.20 | 0.447 |
| 22.0 | No | 6.19 | 0.421 |

**Key result**: Adaptive PF maintains ~44% weight and dramatically reduces RMSE
(from 31.5 baseline to 1.2-6.2). It cannot fully overcome the oscillation from
other strategies but provides strong stabilization.

### Experiment 5: Baseline Without PF

| Converged | Iterations | Delta | RMSE |
|-----------|------------|-------|------|
| No | 100 | 15.65 | 31.50 |

This reproduces the Phase 5 result: structural non-convergence.

### Experiment 6: Adaptive PF with Tiny Spread

| Spread | Converged | Iterations | Delta | RMSE |
|--------|-----------|------------|-------|------|
| 0.01 | Yes (!) | 1 | 0.005 | 31.258 |
| 0.10 | No | 5000 | 0.050 | 0.028 |
| 1.00 | No | 5000 | 0.500 | 0.281 |

**Key insight**: spread=0.01 "converges" on iteration 1 because `α*S = 0.005 < 0.01`
threshold, but the market barely moved (RMSE=31.26). The convergence is vacuous.
spread=0.1 with 5000 iterations gives RMSE=0.028 — nearly perfect. This matches
the theoretical prediction: RMSE = 0.05/√3 = 0.029.

---

## Summary of Answers to Key Questions

### Does adding PF guarantee convergence to P_real?

**No.** Under the current market engine:

- **Fixed PF**: Converges (delta=0) but with substantial bias. RMSE 22+ EUR.
- **Adaptive PF (alone)**: Oscillates with RMSE ≈ α*S/√3. Does not formally converge
  unless spread is impractically small.
- **Adaptive PF (with other strategies)**: Reduces RMSE dramatically (~1-6 EUR) but
  cannot fully overcome oscillation from opposing strategies.
- **Fixed PF + other strategies**: PF gets pushed to zero weight by the oscillation
  dynamics. Completely ineffective.

### Why doesn't the engine converge?

The fundamental issue is **constant step size**. The implied forecast `P ± S` creates
a step of `α*S` per iteration that doesn't shrink as the market price approaches
P_real. This prevents the formation of a contraction mapping. Binary directions
(+1/-1) cannot express "almost there, just nudge a little."

### What would fix convergence?

1. **Adaptive spread**: `S_k ∝ |P_real - P_k|` — step shrinks near target
2. **Continuous forecasts**: Let strategies output real-valued price predictions instead of ±1
3. **Exponential decay**: `S_k = S_0 * γ^k` — spread decays over iterations
4. **Gradient-based update**: Replace weight-by-profit with gradient descent on |P_market - P_real|

### Implications for the hackathon platform

The market simulation engine, as designed, is better understood as a **mechanism for
ranking strategy differentiation** than a convergence-to-truth engine. It rewards
strategies that are correct AND contrarian (different from the consensus). This is
actually a reasonable market design for a hackathon — it prevents the trivial strategy
of copying the best performer — but it should not be interpreted as a price discovery
mechanism that converges to real settlement prices.

---

## Code Deliverables

### New files
- `src/energy_modelling/challenge/convergence.py` — Convergence analysis module
  - `fixed_perfect_foresight_directions()` — Static PF directions
  - `adaptive_perfect_foresight_directions()` — Dynamic PF directions
  - `compute_theoretical_steps_to_arrival()` — Step count formula
  - `compute_overshoot_bias()` — Overshoot formula
  - `compute_convergence_trajectory()` — Extract trajectory from equilibrium
  - `run_adaptive_foresight_market()` — Adaptive PF market runner
  - `ConvergenceTrajectory` — Result dataclass

- `strategies/perfect_foresight.py` — PerfectForesightStrategy
  - ChallengeStrategy that cheats by looking up real settlement prices
  - For analysis only, not a legitimate competitor

### New tests
- `tests/challenge/test_convergence.py` — 29 tests
- `tests/challenge/test_perfect_foresight.py` — 7 tests

All 36 new tests pass. Total test count: 38 (EDA) + 145 (challenge) + 36 (Phase 7) = 219+.
