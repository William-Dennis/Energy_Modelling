# EDA Part 4: Signal Extraction and Strategy Ideas

This document translates the analytical findings from Parts 2 and 3 into concrete, testable trading signals. For each signal we provide economic motivation, implementation sketch, and an indicative performance estimate on the training set. The goal is to help students identify which ideas are worth refining and which are dead ends.

---

## 4.1 Signal Taxonomy

Signals can be grouped by their source and interpretation:

| Group | Examples | Economic Logic |
|-------|----------|----------------|
| **Calendar effects** | Day of week, month | Market microstructure, demand patterns |
| **Supply signals** | Wind forecast, solar forecast | Renewable output drives marginal cost |
| **Demand signals** | Load forecast | Demand drives price ceiling |
| **Relative price signals** | Price vs MA, DE-FR spread | Mean reversion, arbitrage |
| **Fuel cost signals** | Gas trend, carbon trend | Marginal cost of dispatchable generation |
| **Market stress** | Price volatility | Tail risk and liquidity signals |
| **ML combinations** | Net demand, multi-feature models | Non-linear combinations |

---

## 4.2 Signal 1: Day-of-Week Effect

### Motivation

Day-ahead electricity futures settle at different prices for weekday versus weekend delivery. Weekend days (Saturday, Sunday) have dramatically lower industrial and commercial demand. When the settlement rolls from a weekend contract to a Monday contract, the price almost always rises because Monday demand is much higher.

### Pattern

| Day | % Positive | Economic Reason |
|-----|-----------|-----------------|
| Monday | **90.4%** | Weekend→workday demand step-up |
| Tuesday | 61.3% | Normal workday, mild positive bias |
| Wednesday | 50.2% | Essentially coin-flip |
| Thursday | 47.1% | Slight negative bias |
| Friday | 38.7% | Workday→weekend, demand drops |
| Saturday | **13.0%** | Low-demand weekend (night after night) |
| Sunday | 26.8% | Weekend, but slightly better than Sat |

### Implementation

```python
def act(self, state: ChallengeState) -> int | None:
    dow = state.delivery_date.weekday()  # 0=Mon, 6=Sun
    if dow == 0:   return  1   # Monday: almost always long
    if dow == 1:   return  1   # Tuesday: positive bias
    if dow == 4:   return -1   # Friday: negative bias
    if dow == 5:   return -1   # Saturday: strongly short
    if dow == 6:   return -1   # Sunday: short
    return None                # Wed/Thu: skip
```

### Indicative Performance (Training, 2019–2023)

| Metric | Value |
|--------|-------|
| Total PnL | **+471,218 EUR** |
| Sharpe Ratio | **+6.59** |
| Mean daily PnL (all days) | +258 EUR |
| N trades | 1,826 |
| Max Drawdown | Very low (~−1,231 EUR peak-to-trough) |

### Validation (2024)

| Metric | Value |
|--------|-------|
| Total PnL (2024) | +93,213 EUR |
| Sharpe Ratio | +8.23 |

The DOW signal **persists strongly on the validation set**, suggesting it is a genuine structural market feature rather than overfitting.

### Risk Notes

- The DOW effect is real but arises from market microstructure, not forecasting skill. A regulator or market design change could theoretically remove it.
- The signal is so strong that it dominates most combined strategies. Students should verify they are not simply re-discovering the DOW effect when they believe they have found a different signal.

---

## 4.3 Signal 2: Renewable Forecast (Wind)

### Motivation

In the merit-order model, renewable generation (wind, solar) has near-zero marginal cost. High renewable supply displaces expensive gas/coal generation, lowering the market clearing price. Higher wind forecast → lower prices → short signal.

### Implementation

```python
def fit(self, train_data: pd.DataFrame) -> None:
    self.wind_median = (
        train_data['forecast_wind_onshore_mw_mean']
        + train_data['forecast_wind_offshore_mw_mean']
    ).median()

def act(self, state: ChallengeState) -> int | None:
    total_wind = (
        state.features['forecast_wind_onshore_mw_mean']
        + state.features['forecast_wind_offshore_mw_mean']
    )
    return -1 if total_wind > self.wind_median else 1
```

### Signal Strength

| Feature | Corr with Direction |
|---------|---------------------|
| `forecast_wind_offshore_mw_mean` | −0.212 |
| `forecast_wind_onshore_mw_mean` | −0.186 |
| Combined total wind forecast | ~−0.23 |

### Indicative Performance (Validation 2024)

| Metric | Value |
|--------|-------|
| Total PnL | +56,542 EUR |
| Sharpe | +4.01 |

The wind signal alone is materially profitable but substantially weaker than the DOW signal. Wind is best used as a **supplementary signal** to refine the DOW-based direction.

### Refinement Ideas

- Use **wind forecast error** (today's forecast minus yesterday's realised generation) as a surprise signal.
- Combine offshore + onshore forecasts (offshore has slightly stronger signal in this dataset).
- Apply a **residual approach**: after accounting for DOW, does wind still add independent signal?

```python
# Net residual wind signal after DOW
train['dow_expected'] = train.groupby('dow')['target_direction'].transform('mean')
train['residual_target'] = train['target_direction'] - train['dow_expected']
wind_corr = train['forecast_wind_offshore_mw_mean'].corr(train['residual_target'])
print(f"Wind corr after DOW residual: {wind_corr:.4f}")
# Typically ~−0.12 (still meaningful but weaker)
```

---

## 4.4 Signal 3: Net Demand Forecast

### Motivation

The single most economically principled signal is **net demand**: total electricity demand minus renewable supply. This is the demand that must be met by dispatchable (price-setting) generation. Higher net demand → higher marginal cost generator is called → higher price.

### Construction

```python
train['net_demand'] = (
    train['load_forecast_mw_mean']
    - train['forecast_wind_onshore_mw_mean']
    - train['forecast_wind_offshore_mw_mean']
    - train['forecast_solar_mw_mean']
)
```

| Feature | Corr with Direction |
|---------|---------------------|
| `load_forecast_mw_mean` alone | +0.235 |
| `forecast_wind_offshore_mw_mean` alone | −0.212 |
| **Net demand (all four terms)** | **~+0.30** |

The combined net demand signal outperforms any individual feature.

### Normalisation

Because absolute demand levels changed dramatically across years (energy crisis), consider a **z-score normalisation** over a rolling window:

```python
train_sorted = train.sort_values('delivery_date')
nd = train_sorted['net_demand']
rolling_mean = nd.rolling(90, min_periods=20).mean()
rolling_std  = nd.rolling(90, min_periods=20).std()
train_sorted['net_demand_zscore'] = (nd - rolling_mean) / rolling_std
```

This makes the signal regime-neutral: a high net demand relative to the recent past is the true information, not a high absolute level.

---

## 4.5 Signal 4: Price Mean-Reversion

### Motivation

Day-ahead electricity prices tend to exhibit **mean-reversion** over multi-day horizons. Extreme price spikes (2022 crisis) eventually correct; periods of very low prices (negative prices, low demand) recover. This creates a reversal signal.

### Construction

```python
train_sorted = train.sort_values('delivery_date')
train_sorted['price_ma20'] = train_sorted['price_mean'].rolling(20, min_periods=5).mean()
train_sorted['price_zscore'] = (
    (train_sorted['price_mean'] - train_sorted['price_ma20'])
    / train_sorted['price_mean'].rolling(20, min_periods=5).std()
)
```

**Correlation of `price_zscore` with direction:** approximately **−0.17**

When price is more than 1 standard deviation above its 20-day average, a downward move is more likely; when it is below, an upward move is more likely.

### Threshold Strategy

```python
def act(self, state: ChallengeState) -> int | None:
    zscore = compute_price_zscore(state)  # from historical data
    if zscore > 1.0:  return -1   # elevated price → short
    if zscore < -1.0: return  1   # depressed price → long
    return None                   # neutral zone → skip
```

### Caution

Pure mean-reversion strategies can fail catastrophically in trending markets (e.g. the 2022 uptrend). Always combine with a trend filter or enforce a skip zone when in a strong regime.

---

## 4.6 Signal 5: Gas Price Trend

### Motivation

Natural gas sets the marginal price in the DE-LU market for a large fraction of hours. When gas prices rise over several consecutive days, electricity prices typically follow. This creates a **momentum signal** based on gas price trends.

### Construction

```python
train_sorted = train.sort_values('delivery_date')
train_sorted['gas_trend_3d'] = train_sorted['gas_price_usd_mean'].diff(3)

# Correlation
train_sorted['gas_trend_3d'].corr(train_sorted['target_direction'])
# ~+0.05 to +0.10 depending on period
```

The gas trend signal is weaker than wind or load (correlation ~0.05–0.10) but provides **fundamental macroeconomic context** that is uncorrelated with the other signals.

### Regime Filter

Gas was the dominant pricing signal during the 2021–2022 crisis but became less relevant in 2023 as gas prices normalised. Consider using a regime indicator to weight this signal appropriately.

---

## 4.7 Signal 6: DE-FR Price Spread (Arbitrage)

### Motivation

Germany and France are tightly interconnected via transmission lines. When DE-LU prices deviate significantly above French prices, capacity flows increase, and DE prices tend to be pulled back toward French levels (arbitrage). This creates a **convergence signal**.

### Construction

```python
train['de_fr_spread'] = train['price_mean'] - train['price_fr_eur_mwh_mean']
# Correlation with direction: −0.132
# Mean spread: −11.60 EUR/MWh (DE typically slightly more expensive)
```

| Spread Quintile | % Positive Days |
|----------------|-----------------|
| Q1 (most negative, DE cheapest) | ~52% |
| Q2 | ~49% |
| Q3 | ~48% |
| Q4 | ~46% |
| Q5 (most positive, DE most expensive) | ~44% |

The signal exists but is modest. It is most useful as a **filter**: when DE-LU is clearly cheaper than France, avoid shorting; when DE-LU is much more expensive, avoid going long.

---

## 4.8 Signal Combination Framework

No single signal dominates in all market conditions. A robust strategy should combine signals from complementary sources:

### Scoring Function Approach

```python
def compute_score(state, train_data):
    score = 0.0
    
    # Signal 1: DOW (strongest)
    dow = state.delivery_date.weekday()
    dow_bias = {0: +3.0, 1: +1.0, 2: 0.0, 3: -0.5,
                4: -1.5, 5: -3.0, 6: -2.0}
    score += dow_bias.get(dow, 0.0)
    
    # Signal 2: Wind forecast
    total_wind = (state.features['forecast_wind_onshore_mw_mean']
                  + state.features['forecast_wind_offshore_mw_mean'])
    wind_zscore = (total_wind - train_data['total_wind'].mean()) \
                  / train_data['total_wind'].std()
    score -= wind_zscore * 0.8
    
    # Signal 3: Load forecast
    load_zscore = (state.features['load_forecast_mw_mean']
                   - train_data['load_forecast_mw_mean'].mean()) \
                  / train_data['load_forecast_mw_mean'].std()
    score += load_zscore * 0.8
    
    return score

def act(self, state):
    score = compute_score(state, self.train_data)
    if score > 0.5:  return  1
    if score < -0.5: return -1
    return None
```

### Feature Correlation Matrix (Subset)

| | load_fcast | wind_off_fcast | wind_on_fcast | de_fr_spread | price_vs_ma20 |
|--|-----|-----|-----|-----|-----|
| **load_fcast** | 1.00 | −0.31 | −0.37 | +0.09 | +0.12 |
| **wind_off_fcast** | −0.31 | 1.00 | +0.64 | −0.11 | −0.05 |
| **wind_on_fcast** | −0.37 | +0.64 | 1.00 | −0.13 | −0.07 |
| **de_fr_spread** | +0.09 | −0.11 | −0.13 | 1.00 | +0.38 |
| **price_vs_ma20** | +0.12 | −0.05 | −0.07 | +0.38 | 1.00 |

Load forecast is relatively independent of wind forecasts (though moderately negatively correlated). Price spread and price-vs-MA are correlated with each other but add orthogonal information to wind/load.

---

## 4.9 Machine Learning Considerations

### Feature Engineering Checklist

Before training any ML model, ensure:

- [ ] Remove all label columns (`settlement_price`, `target_direction`, `pnl_*`, `price_change_eur_mwh`) from input features
- [ ] Add calendar features: `dow`, `month`, `quarter`, `is_holiday`
- [ ] Add lagged targets: `target_direction_t-1`, `price_change_t-1` (limited signal but free information)
- [ ] Add derived features: `net_demand`, `de_fr_spread`, `price_vs_ma20`, `gas_trend_3d`
- [ ] Normalise all continuous features with a **look-ahead-free** scaler (fit on train, apply to val/test)
- [ ] Handle missing gas/carbon values (forward-fill)

### Cross-Validation Strategy

A standard k-fold CV is **incorrect** here because of time dependency. Use:

1. **Walk-forward validation:** Train on years 1–N, test on year N+1, then advance one year
2. **Purged time series CV:** Ensure no data from the test period leaks into training

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=4, gap=0)
for train_idx, val_idx in tscv.split(train):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    # Fit model, evaluate on val
```

### Recommended Models

| Model | Pros | Cons |
|-------|------|------|
| Logistic Regression | Interpretable, fast, robust | Cannot capture non-linear interactions |
| Random Forest | Handles non-linearity, feature importance | Can overfit, need careful depth control |
| Gradient Boosting (XGBoost / LightGBM) | Best raw performance in tabular settings | Risk of overfitting to noise |
| Ridge/Lasso | Well-regularised, excellent for small datasets | Linear only |

Start with logistic regression or ridge as a baseline before moving to more complex models.

---

## 4.10 Low-Signal Features to Avoid

| Signal | Reason to Avoid |
|--------|----------------|
| Raw gas price level | Near-zero correlation; dominated by 2022 regime |
| Raw carbon price level | Near-zero correlation |
| Solar forecast | Near-zero correlation (demand also rises with sun) |
| Simple momentum (t−1 direction) | Correlation ~+0.02; essentially zero |
| Nuclear generation | Structurally zero from April 2023; not predictive |
| Raw temperature | Collinear with load; adds nothing beyond load |

---

## 4.11 Validation on Out-of-Sample Data (2024)

Key benchmarks to beat on the 2024 validation set:

| Strategy | 2024 PnL | 2024 Sharpe | Notes |
|----------|----------|-------------|-------|
| Always Long | ~−2,000 EUR | ~−0.1 | Baseline (given) |
| Wind Contrarian | +56,542 EUR | +4.01 | Median-split on wind forecast |
| **DOW Strategy** | **+93,213 EUR** | **+8.23** | Very strong out-of-sample |

A combined DOW + wind strategy should aim to exceed the DOW-only benchmark.

---

## 4.12 Summary: Top 5 Actionable Signals

1. **Day of week** — The dominant signal; exploit the Mon/Sat asymmetry explicitly.
2. **Net demand forecast** (`load − wind_onshore − wind_offshore`) — The strongest single economic indicator (~+0.30 correlation).
3. **Price vs. 20-day MA** — Mean-reversion proxy; be careful in trending regimes.
4. **DE-FR price spread** — Moderate arbitrage signal; best used as a filter.
5. **Gas price 3-day trend** — Weak but uncorrelated with the above; adds marginal diversification.

---

**Next:** [Part 5 — EDA Scoring Rubric](05_eda_scoring_rubric.md)
