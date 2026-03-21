# Synthetic Futures Market Model

## Variables

* ( P_t ) : Real price at day ( t )
* ( P_t^m ) : Market price at day ( t ), shown at day ( t-1 )
* ( \hat{P}_{i,t} ) : Forecast by strategy ( i ) for day ( t )
* ( q_{i,t} \in {-1, +1} ) : Trading decision of strategy ( i ) at day ( t )
* ( r_{i,t} ) : Profit of strategy ( i ) at day ( t )
* ( \Pi_i ) : Total profit of strategy ( i ) over one iteration
* ( w_i ) : Weight of strategy ( i ) for market price aggregation
* ( N ) : Number of strategies
* ( T ) : Number of days in one iteration

---

## Step 1: Trading Decision

[
q_{i,t} = \begin{cases}
+1 & \text{if } \hat{P}*{i,t} > P*{t-1}^m \
-1 & \text{if } \hat{P}*{i,t} < P*{t-1}^m
\end{cases}
]

---

## Step 2: Profit Calculation

[
r_{i,t} = q_{i,t} \cdot (P_t - P_{t-1}^m)
]

[
\Pi_i = \sum_{t=1}^{T} r_{i,t}
]

---

## Step 3: Selection and Weighting

Only strategies with positive total profit are included:

[
w_i = \max(\Pi_i, 0)
]

Weights are normalized for aggregation:

[
w_i^{norm} = \frac{w_i}{\sum_{j=1}^N w_j}
]

---

## Step 4: Market Price Update

Market price is updated as the weighted average of surviving strategy forecasts:

[
P_t^{m,(k+1)} = \sum_{i=1}^{N} w_i^{norm} \cdot \hat{P}_{i,t}
]

---

## Step 5: Iteration

Repeat Steps 1-4 for ( k = 1, \dots, N_{iter} ) iterations.

[
P_t^{m,(k+1)} = F(P_t^{m,(k)})
]

---

## Key Properties

1. **Profit is based on real price:** ( P_t ) determines rewards, avoiding circularity.
2. **Directional accuracy:** Strategies are rewarded for predicting the correct movement relative to the market price.
3. **Selection pressure:** Only profitable strategies influence the next market price.
4. **Market price as ensemble forecast:** ( P_t^m ) approximates a consensus of best strategies.
5. **Fixed point:** A converged ( P_t^m ) satisfies

[
P_t^{m,*} = \sum_i w_i^* \hat{P}_{i,t}(P_t^{m,*})
]

6. **Limitation:** Trades do not impact market price; this is a prediction-market-style aggregation rather than full market microstructure.

---

## Implementation Extensions (2026-03-21)

The specification above defines the **undampened** model. The production
engine (`src/energy_modelling/backtest/futures_market_engine.py`) extends
it with **EMA dampening** in Step 4, which was found necessary to achieve
convergence when many strategies with opposing forecasts are present.

### Extended Step 4: EMA-Dampened Price Update

The production price update replaces the direct weighted average with an
exponential moving average (EMA) blend of the old and new prices:

```
P_t^{m,(k+1)} = ema_alpha * (sum_i w_i^norm * P_hat_i_t) + (1 - ema_alpha) * P_t^{m,(k)}
```

where `ema_alpha` controls the update speed:

| `ema_alpha` | Behaviour |
|-------------|-----------|
| `1.0` | **Undampened** — identical to the spec (Step 4 above) |
| `0.1` | **Production default** — slow, stable convergence |
| `0.0` | No update (prices frozen) |

### Why dampening is needed

Phase 7 proved that the undampened model converges instantly when a
perfect-foresight strategy is present (Theorem 1). However, Phase 8
showed that with 67 real strategies, the undampened model enters a
3-step limit cycle (oscillation delta ~81 EUR/MWh on 2024 data).

Phase 8 recommended `running_avg_k=5` (iteration-level running average),
but this was never implemented. Instead, Phase 9 adopted EMA dampening
(`ema_alpha=0.1`), which achieves convergence with similar MAE results.

### Relationship to the spec

The spec remains the canonical mathematical model. The EMA extension is
an engineering choice for the iterative solver — it does not change the
fixed-point equation (Property 5). When `ema_alpha=1.0`, the production
engine is exactly spec-compliant.

### References

- Phase 7: `docs/phases/phase_7_convergence_analysis.md` (undampened theorems)
- Phase 8: `docs/phases/phase_8_oscillation_research.md` (oscillation research)
- Phase 9: `docs/phases/phase_9_ema_price_update.md` (EMA experiments)
- Engine: `src/energy_modelling/backtest/futures_market_engine.py` (implementation)
