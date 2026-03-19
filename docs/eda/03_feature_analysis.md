# EDA Part 3: Feature Analysis

This document provides a detailed analysis of each feature group in the dataset: generation, weather, load, neighbouring prices, cross-border flows, commodity prices, and day-ahead forecasts. For each group we report summary statistics, correlation with the target direction, and interpretive notes grounded in energy market fundamentals.

---

## 3.1 Correlation Summary

The table below ranks all features by their absolute Pearson correlation with `target_direction` (training set, 2019–2023). Higher magnitude = stronger linear association.

| Rank | Feature | Correlation | Direction |
|------|---------|-------------|-----------|
| 1 | `load_forecast_mw_mean` | **+0.235** | Higher load forecast → price likely to rise |
| 2 | `forecast_wind_offshore_mw_mean` | **−0.212** | Higher offshore wind → price likely to fall |
| 3 | `gen_wind_onshore_mw_mean` | **+0.210** | High yesterday wind → higher prices today\* |
| 4 | `weather_wind_speed_10m_kmh_mean` | **+0.207** | High wind speed → higher prices today\* |
| 5 | `gen_fossil_gas_mw_mean` | **−0.195** | High gas generation (yesterday) → price fall |
| 6 | `flow_nl_net_import_mw_mean` | **−0.192** | Heavy Dutch imports → lower prices |
| 7 | `gen_fossil_brown_coal_lignite_mw_mean` | **−0.187** | High lignite gen → lower price change |
| 8 | `forecast_wind_onshore_mw_mean` | **−0.186** | Higher onshore wind forecast → price likely to fall |
| 9 | `gen_wind_offshore_mw_mean` | **+0.153** | High yesterday offshore wind → higher prices today\* |
| 10 | `gen_fossil_hard_coal_mw_mean` | **−0.148** | High coal gen → lower prices |
| 11 | `price_min` | **−0.143** | Higher yesterday minimum → price likely to fall |
| 12 | `load_actual_mw_mean` | **−0.133** | High yesterday load → price likely to fall |
| 13 | `price_vs_ma20`\*\* | **−0.173** | Price above 20-day MA → mean-reversion tendency |
| 14 | `flow_fr_net_import_mw_mean` | **−0.099** | Heavy French imports → lower prices |
| 15 | `price_mean` | **−0.092** | Higher yesterday price → price likely to fall |

\*The positive correlation of **lagged** wind with direction is counterintuitive at first: it reflects the fact that high wind **yesterday** often coincides with Monday (wind is typically higher in winter, and Mondays almost always go up). This is a collinearity with the DOW effect, not a direct causal wind → price rise relationship.

\*\*`price_vs_ma20` is a derived feature: `price_mean − rolling_20day_mean(price_mean)`.

---

## 3.2 Generation Features

### Summary Statistics (Training Set)

| Feature | Mean (MW) | Std (MW) | Min | Max |
|---------|-----------|----------|-----|-----|
| `gen_solar_mw_mean` | 5,612 | 3,728 | 0 | 15,361 |
| `gen_wind_onshore_mw_mean` | 11,727 | 8,703 | 629 | 44,010 |
| `gen_wind_offshore_mw_mean` | 3,476 | 2,207 | 256 | 9,302 |
| `gen_fossil_gas_mw_mean` | 6,247 | 3,824 | 384 | 22,001 |
| `gen_fossil_hard_coal_mw_mean` | 3,616 | 2,479 | 77 | 11,940 |
| `gen_fossil_brown_coal_lignite_mw_mean` | 10,628 | 3,322 | 3,117 | 17,170 |
| `gen_nuclear_mw_mean` | 5,411 | 2,921 | 0 | 9,506 |

### Nuclear Phase-Out (2023)

Germany completed its nuclear phase-out in April 2023. The average nuclear generation by year demonstrates this structural break:

| Year | Avg Nuclear Gen (MW) |
|------|---------------------|
| 2019 | 8,112 |
| 2020 | 6,937 |
| 2021 | 7,471 |
| 2022 | 3,755 |
| 2023 | **778** |

All 259 rows with `gen_nuclear_mw_mean == 0` are from 2023. This makes nuclear generation **not useful as a signal for 2024+ predictions** since it remains zero. Strategies that rely on this feature must account for the structural break.

### Wind Generation vs. Direction

Wind generation (lagged) shows a counterintuitive **positive** correlation (+0.21) with tomorrow's direction. This is primarily a collinearity with the day-of-week effect: high-wind periods occur more in winter months, and the DOW uplift on Mondays confounds the relationship. 

The **forecast** wind (`forecast_wind_onshore_mw_mean`, `forecast_wind_offshore_mw_mean`) shows the correct **negative** relationship (−0.186, −0.212): high renewable generation is forecast to suppress prices.

```python
# Check wind forecast vs direction
train.groupby('target_direction')['forecast_wind_onshore_mw_mean'].describe()
# Short (direction=-1): mean forecast = 13,102 MW
# Long  (direction=+1): mean forecast =  9,900 MW
# => Higher forecast wind associated with price falls
```

### Solar Generation

Solar has essentially **zero correlation** (±0.01) with direction. This is expected: solar output follows a strong seasonal pattern, but so does demand, and the two largely cancel each other out in the daily settlement price direction.

---

## 3.3 Weather Features

### Summary Statistics (Training Set)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `weather_temperature_2m_degc_mean` | 9.78°C | 6.99 | −12.49 | 27.49 |
| `weather_wind_speed_10m_kmh_mean` | 12.05 km/h | 5.49 | 2.53 | 35.80 |
| `weather_shortwave_radiation_wm2_mean` | 132 W/m² | 93.6 | 3.79 | 344.9 |

### Temperature

Temperature has near-zero correlation with direction (+0.004). While temperature drives heating/cooling demand, this effect is already captured by the load features. Raw temperature is unlikely to add unique predictive power.

### Wind Speed

Wind speed (+0.207 correlation) is a reasonable proxy for generation potential. However, the **forecast wind generation** variables are strictly better because they already translate wind speed into expected MW output using actual turbine curves and forecasting models.

### Shortwave Radiation

Shortwave radiation (a proxy for solar irradiance) has essentially zero correlation (+0.0002) with direction. This is consistent with the solar generation finding: more sunlight does not reliably predict price direction.

---

## 3.4 Load Features

### Load Actual vs. Load Forecast

| Feature | Mean (MW) | Std (MW) | Correlation with Direction |
|---------|-----------|----------|--------------------------|
| `load_actual_mw_mean` | 55,087 | 6,239 | **−0.133** |
| `load_forecast_mw_mean` | 55,091 | 6,141 | **+0.235** |

The **sign flip** between lagged actual load and same-day forecast is striking:

- **Yesterday's high load (lagged actual)** → *slightly negative* correlation. High load yesterday tends to coincide with higher prices yesterday, which may mean-revert downward today.
- **Today's high load forecast** → *positive* correlation. Higher expected demand today → market expects higher prices today.

This is a good example of why the timing group matters: the same conceptual variable (load) has opposite predictive signs depending on whether it is lagged or same-day.

```python
# Load forecast by direction
train.groupby('target_direction')['load_forecast_mw_mean'].mean()
# direction = -1: 53,739 MW (lower load forecast when prices fall)
# direction = +1: 56,629 MW (higher load forecast when prices rise)
# Difference: ~2,890 MW (about 5% of the mean)
```

---

## 3.5 Neighbouring Country Prices

### Summary Statistics (Training Set)

| Country | Mean (EUR/MWh) | Std | Corr with DE Direction |
|---------|---------------|-----|------------------------|
| France (`price_fr`) | 100.19 | 98.46 | −0.044 |
| Netherlands (`price_nl`) | 92.32 | 93.13 | −0.062 |
| Austria (`price_at`) | 93.01 | 93.12 | −0.059 |
| Poland (`price_pl`) | 95.52 | 92.53 | −0.095 |
| Czech Republic (`price_cz`) | 93.51 | 93.60 | −0.068 |
| Denmark Zone 1 (`price_dk_1`) | 93.00 | 91.95 | −0.090 |

All neighbouring prices have small negative correlations with DE-LU direction. This suggests a mild mean-reversion signal: when neighbouring prices are high, DE-LU prices are somewhat more likely to fall.

### DE-LU vs. France Spread

The **DE-FR spread** (DE price minus FR price) is a commonly cited signal in European energy markets:

```
Mean spread: −11.60 EUR/MWh
Std:          31.80 EUR/MWh
Q25:          −9.90 EUR/MWh
Median:        −1.47 EUR/MWh
Q75:           +0.91 EUR/MWh
```

The spread is **negative on average** (DE prices are on average ~11.60 EUR/MWh cheaper than France over this period), with a Pearson correlation with DE direction of **−0.132**. When the spread is positive (DE more expensive than France), there is a tendency for DE prices to fall back toward French levels; when the spread is very negative (DE much cheaper), DE prices are somewhat more likely to rise.

```python
# Compute the spread
train['de_fr_spread'] = train['price_mean'] - train['price_fr_eur_mwh_mean']

# Mean spread by direction
train.groupby('target_direction')['de_fr_spread'].mean()
# direction = -1: +9.5 EUR/MWh  (DE expensive → likely to fall)
# direction = +1: −17.3 EUR/MWh (DE cheap   → likely to rise)
```

---

## 3.6 Cross-Border Flow Features

| Feature | Mean (MW) | Std | Corr with Direction |
|---------|-----------|-----|---------------------|
| `flow_fr_net_import_mw_mean` | −2,027 | 2,658 | **−0.099** |
| `flow_nl_net_import_mw_mean` | −617 | 1,921 | **−0.192** |

Negative values indicate net export from DE-LU to the neighbouring zone.

The Netherlands import flow has the stronger signal (−0.192): when DE is exporting heavily to the Netherlands, it suggests DE prices were low yesterday relative to Dutch prices, and a reversal (price rise) is slightly more likely tomorrow.

**Interpretation:** Flows are a market-integration signal. High imports to DE (positive values) suggest prices in DE are cheap relative to neighbours; high exports (negative values) suggest DE prices are elevated.

---

## 3.7 Commodity Price Features

### Gas Price

```
Gas price mean:  49.0 USD/MMBtu
Gas price std:   52.3 USD/MMBtu
Gas price min:    3.5 USD/MMBtu  (2020 COVID period)
Gas price max:  339.2 USD/MMBtu  (2022 energy crisis peak)
Correlation with direction: +0.013 (near zero)
```

Gas price itself has near-zero correlation with direction. However, the **trend** in gas prices may be more informative. A rising gas price over several days typically pressures electricity prices upward (gas sets the marginal price in many hours).

```python
# Gas price trend
train_sorted = train.sort_values('delivery_date')
train_sorted['gas_change_3d'] = train_sorted['gas_price_usd_mean'].diff(3)
print(train_sorted['gas_change_3d'].corr(train_sorted['target_direction']))
# Positive correlation: rising gas prices → electricity price rise
```

### Carbon Price

```
Carbon price mean:  22.2 USD/tCO2
Carbon price std:    7.0 USD/tCO2
Correlation with direction: +0.006 (near zero)
```

Similar to gas: the level of carbon price has minimal predictive power for daily direction. Carbon price dynamics operate on a longer time scale. During the 2022 energy crisis, carbon prices were extremely elevated (~35–38 USD/tCO2) but daily movements showed no consistent directional bias.

---

## 3.8 DE-LU Own Price Statistics

| Feature | Correlation with Direction |
|---------|--------------------------|
| `price_mean` | −0.092 |
| `price_max` | −0.053 |
| `price_min` | **−0.143** |
| `price_std` | +0.058 |

`price_min` has the strongest correlation among the own-price features. When yesterday's minimum price was very low (e.g. due to a negative-price hour), it is more likely that prices recover (upward direction) today.

`price_std` (yesterday's intraday volatility) has a small **positive** correlation: high volatility days (larger intraday swings) are very slightly more likely to be followed by an upward day — possibly because high-volatility periods correlate with high-demand winter days.

---

## 3.9 Day-Ahead Forecast Summary

The four forecast features deserve special attention because they describe **today's** conditions, not yesterday's:

| Feature | Mean (MW) | Corr with Direction |
|---------|-----------|---------------------|
| `load_forecast_mw_mean` | 55,091 | **+0.235** |
| `forecast_wind_onshore_mw_mean` | 11,604 | **−0.186** |
| `forecast_wind_offshore_mw_mean` | 2,826 | **−0.212** |
| `forecast_solar_mw_mean` | 5,610 | −0.004 |

- **Load forecast** is the strongest single feature in the dataset (+0.235).
- **Wind forecasts** (both onshore and offshore) are strong inverse signals: high forecast renewables depress day-ahead prices.
- **Solar forecast** adds little — solar is already accounted for in price expectations.

A simple combined signal based on `load_forecast - forecast_wind_onshore - forecast_wind_offshore` (a "net demand" proxy) is a strong feature:

```python
train['net_demand_forecast'] = (
    train['load_forecast_mw_mean']
    - train['forecast_wind_onshore_mw_mean']
    - train['forecast_wind_offshore_mw_mean']
)
print(train['net_demand_forecast'].corr(train['target_direction']))
# ~+0.30 (stronger than any individual feature)
```

---

## 3.10 Feature Collinearity Notes

Several features are highly correlated with each other:

- `price_fr`, `price_nl`, `price_at`, `price_cz`, `price_pl`, `price_dk_1` all move together (European market coupling).
- `gen_fossil_gas_mw_mean` and `price_mean` are positively correlated (gas sets marginal price).
- `weather_wind_speed_10m_kmh_mean` and `gen_wind_onshore_mw_mean` are correlated (wind speed drives wind generation).
- `forecast_wind_onshore_mw_mean` and `gen_wind_onshore_mw_mean` tend to correlate (weather persistence).

When building ML models, avoid feeding all correlated price columns simultaneously; the resulting feature importance will be diluted. Consider principal component analysis or careful feature selection.

---

## 3.11 Key Takeaways

| Signal | Type | Strength | Notes |
|--------|------|----------|-------|
| Day of week | Calendar | Very strong | 90% Mondays up, 87% Saturdays down |
| Load forecast | Same-day | Strong (+0.235) | Net demand is better than load alone |
| Wind forecast (offshore) | Same-day | Strong (−0.212) | High renewables → lower prices |
| Wind forecast (onshore) | Same-day | Strong (−0.186) | Use alongside offshore |
| Net demand forecast | Derived | Strongest (~+0.30) | `load - wind_onshore - wind_offshore` |
| Gas generation (lagged) | Lagged | Moderate (−0.195) | Confounded by 2022 crisis |
| NL flow (lagged) | Lagged | Moderate (−0.192) | Market-integration proxy |
| Price vs MA20 | Derived | Moderate (−0.173) | Mean-reversion proxy |
| DE-FR spread | Derived | Moderate (−0.132) | Relative price arbitrage |
| Nuclear generation | Lagged | Weak (−0.030) | Structurally zero from 2023 onwards |
| Solar forecast | Same-day | Negligible | Already priced in |
| Momentum (t−1 direction) | Lagged | Negligible (+0.021) | No serial correlation in direction |

---

**Next:** [Part 4 — Signal Extraction and Strategy Ideas](04_signal_extraction.md)
