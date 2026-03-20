# Phase 8a: Oscillation Problem Characterisation

## Status: PENDING

## Objective

Formally characterise the non-convergence (oscillation) observed in the
spec-compliant synthetic futures market engine when run against real strategy
forecasts on 2024 and 2025 evaluation data.

## Observed Failure

Both market runs fail to converge within 20 iterations:

| Year | Converged | Final Delta (EUR/MWh) | Iterations | Strategies |
|------|-----------|-----------------------|------------|------------|
| 2024 | No | 81.42 | 20 (max) | 67 |
| 2025 | No | 42.49 | 20 (max) | 67 |

The convergence threshold is 0.01 EUR/MWh.  The final deltas are 4-8 orders
of magnitude above this threshold.

---

## 2024 Oscillation Structure

### Stable 3-Iteration Limit Cycle

From iteration 10 onward, the 2024 market locks into a repeating 3-step cycle
with near-identical deltas:

| Cycle Phase | Iterations | Delta (EUR/MWh) | Dominant Cluster | Active Strategies |
|-------------|------------|-----------------|------------------|-------------------|
| A (jump up) | 10, 13, 16, 19 | ~81.4 | Stacked Ridge Meta (0.043), Lag2Reversion (0.039), KNN (0.027) | 51 |
| B (drop) | 11, 14, 17 | ~57.8 | Lasso Calendar (0.062), Lasso Regression (0.062), Mean Forecast (0.062) | 36 |
| C (settle) | 12, 15, 18 | ~25.5 | Lasso Calendar (0.114), Lasso Regression (0.114), Median Forecast (0.106) | 10 |

The cycle repeats with sub-EUR precision: iter 13 delta = 81.42, iter 16 delta = 81.42,
iter 19 delta = 81.42.  This is a mathematical limit cycle, not a transient.

### Profit Magnitudes Reveal the Mechanism

At iteration 13 (Phase A, Stacked Ridge dominant):
- Stacked Ridge Meta: +967.8 EUR
- Lag2Reversion: +889.6 EUR
- Mean Forecast Regression: **-949.5 EUR** (worst, eliminated)
- Elastic Net: **-783.0 EUR** (eliminated)

At iteration 14 (Phase B, Lasso dominant):
- Lasso Calendar Augmented: **+6,363.4 EUR**
- Lasso Regression: **+6,363.4 EUR**
- Bayesian Ridge: +6,262.8 EUR
- Quarter Seasonal: **-5,195.1 EUR** (worst, eliminated)

The 6.6x profit ratio (6,363 vs 968) between phases shows the Lasso cluster
forecasts far more accurately when prices are at Stacked-Ridge-era levels,
creating a powerful seesaw.

### Worst Oscillation Days (2024, iter 13 to 14)

| Date | |Delta| EUR/MWh | P(iter 13) | P(iter 14) | P(real) | P(last settle) |
|------|----------------------|------------|------------|---------|-----------------|
| 2024-12-15 | 57.8 | 102.6 | 44.8 | 43.8 | 99.4 |
| 2024-10-14 | 54.1 | 24.8 | 78.9 | 110.0 | 11.1 |
| 2024-01-13 | 46.2 | 99.0 | 52.8 | 73.5 | 106.0 |
| 2024-11-24 | 40.8 | 70.8 | 30.0 | 3.9 | 72.7 |
| 2024-03-25 | 40.4 | 38.3 | 78.7 | 85.3 | 27.1 |

The worst day (2024-12-15) swings 57.8 EUR/MWh between consecutive iterations.
At iter 13, the market overshoots to 102.6 (near last_settlement of 99.4),
then at iter 14, Lasso corrects it down to 44.8 (close to real price of 43.8).

### December 2024: The Key Problem Period

December 2024 exhibits extreme real-price volatility:
- Min: 21.0 EUR/MWh
- Max: 395.7 EUR/MWh
- Mean: 108.2 EUR/MWh
- Std: 72.9 EUR/MWh

The Dec 11-13 spike (settlement hit 395.7 on Dec 12) creates massive forecast
disagreement between strategy clusters:
- 2024-12-13: market = 384.2, real = 177.9, prev = 395.7 (market anchored to extreme)
- 2024-12-12: market = 259.3, real = 395.7, prev = 266.8 (market undershoots spike)

---

## 2025 Oscillation Structure

### More Chaotic, Slower Lock-In

The 2025 run shows a different pattern:
- Early iterations (0-3): rapidly changing structure, delta spike to 43.4 at iter 3
- Mid iterations (4-9): irregular oscillation, no clear cycle, delta range 7.5-56.5
- Late iterations (12-19): settles into a 3-step cycle similar to 2024

Late-cycle structure (2025, iters 15-19):

| Iter | Delta | Top Strategy | Weight |
|------|-------|--------------|--------|
| 15 | 59.9 | Stacked Ridge Meta | 0.086 |
| 16 | 42.5 | Elastic Net | 0.067 |
| 17 | 19.0 | Elastic Net | 0.123 |
| 18 | 59.9 | Stacked Ridge Meta | 0.087 |
| 19 | 42.5 | Elastic Net | 0.065 |

Notable: Always Short appears in iter 3 weights at 0.124 — unusual for a
naive strategy, indicating that at iter-2 prices, the market was biased
sufficiently upward that shorting everything was profitable.

### Worst Oscillation Days (2025, iter 18 to 19)

| Date | |Delta| EUR/MWh | P(iter 18) | P(iter 19) |
|------|----------------------|------------|------------|
| 2025-03-24 | 42.5 | 74.5 | 117.0 |
| 2025-03-31 | 38.1 | 29.5 | 67.6 |
| 2025-12-06 | 37.1 | 117.1 | 80.0 |
| 2025-02-25 | 36.3 | 96.9 | 133.2 |
| 2025-12-03 | 36.3 | 99.9 | 136.1 |

---

## Market Accuracy Despite Non-Convergence

Despite failing to converge, the final market prices are useful:

| Metric | 2024 Market | 2024 Prev-Day Baseline |
|--------|------------|------------------------|
| MAE | 19.39 | 22.02 |
| RMSE | 28.39 | 31.26 |
| Bias | 0.01 | N/A |

The market improves on the naive prev-day baseline by 2.6 EUR/MWh MAE (12% reduction),
and is essentially unbiased.  Iteration 0 alone achieves MAE of 15.10 — better than
the final result, suggesting that **the oscillation degrades accuracy after iter 0**.

---

## Root Cause Analysis

Five interacting causes produce the oscillation:

### 1. No Dampening (Spec Constraint)

The spec-compliant engine performs a hard jump:
```
P^m_{t}^(k+1) = sum_i w_i * forecast_{i,t}
```
There is no blending with the previous price (`P^m_{t}^(k)`).  In control theory,
this is equivalent to a gain of 1.0 with no feedback attenuation.  Any pair of
strategy clusters with divergent forecasts will oscillate indefinitely.

### 2. Divergent Strategy Clusters

The 67 strategies split into two forecast "poles":
- **Regression/ML cluster**: Lasso, Elastic Net, Ridge, Bayesian Ridge, Mean/Median Forecast — these produce forecast values close to the historical mean
- **Momentum/calendar cluster**: Stacked Ridge Meta, Lag2Reversion, Weekly Cycle, KNN — these produce forecasts closer to recent prices

On volatile days (Dec 2024), these poles can differ by 50-80 EUR/MWh.

### 3. Winner-Take-All Weight Dynamics

The profit normalisation gives dominant weight to whichever cluster was most
profitable in the *previous* iteration's price regime.  With 51 active strategies
in Phase A vs 10 in Phase C, the weight concentration swings dramatically:
- Phase A: top strategy weight 0.043 (diffuse among 51)
- Phase C: top strategy weight 0.114 (concentrated in 10)

### 4. High-Volatility Outlier Days

December 2024 real prices span 21.0 to 395.7 EUR/MWh — a 19:1 ratio within a single
month.  These extreme days create extreme forecast disagreement and extreme profit
differentials, amplifying the oscillation on those specific dates.

### 5. Last Settlement Price as Initial Price

The engine initialises `P^m_0 = last_settlement_price`.  On gap days (e.g. Dec 13:
last_settlement = 395.7, real = 177.9), this starting point is 217.8 EUR/MWh away
from reality, giving an enormous initial profit signal to whichever cluster forecasts
correctly on that day.

---

## Formal Characterisation

Let F_A and F_B be the mean forecasts of two strategy clusters.  Under the undampened
engine, if:

1. At price P near F_A: cluster B is more profitable (B forecasts closer to real)
2. At price P near F_B: cluster A is more profitable (A forecasts closer to real)

Then the map P -> update(P) has no fixed point, and the system oscillates between
the basins of attraction of F_A and F_B.  The oscillation is a **limit cycle** of
the discrete dynamical system defined by the market update rule.

The 3-step cycle in 2024 arises because the transition from F_A to F_B is not
instantaneous — the weight reallocation takes 2 iterations (Phase B: partial
correction, Phase C: full correction) before Phase A resets the cycle.

---

## Key Questions for Phase 8b-8e

1. Can dampening break the limit cycle without violating the spec's intent?
2. Can weight reform prevent winner-take-all dynamics?
3. Can better initialisation reduce the amplitude of the first oscillation?
4. Can iteration-level smoothing achieve convergence to a stable price?
5. What is the optimal intervention — or is a combination needed?

## Files

- `data/results/market_2024.pkl` — 2024 market result (20 iterations, non-converged)
- `data/results/market_2025.pkl` — 2025 market result (20 iterations, non-converged)
- `src/energy_modelling/backtest/futures_market_engine.py` — current spec-compliant engine
- `docs/energy_market_spec.md` — canonical market specification
