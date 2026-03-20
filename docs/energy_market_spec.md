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
