# Phase 1: EDA Audit

## Status: COMPLETE

## Objective

Formally audit the existing 12 EDA sections in `_eda.py`, document what each section
does well, identify gaps, and prioritize which analyses to add in Phase 2.

## Prerequisites

- Phase 0 complete (codebase consolidated, tests green)

## Checklist

### 1a. Review existing 12 sections
- [x] Section 1: Dataset Overview — assess completeness
- [x] Section 2: Day-Ahead Price Time Series — assess
- [x] Section 3: Generation Mix — assess
- [x] Section 4: Load & Forecasts — assess
- [x] Section 5: Neighbour Prices & Cross-Border Flows — assess
- [x] Section 6: Carbon & Gas Prices — assess
- [x] Section 7: Weather Overview — assess
- [x] Section 8: Key Correlations with Price — assess
- [x] Section 9: Price Distribution & Temporal Patterns — assess
- [x] Section 10: Negative Price Analysis — assess
- [x] Section 11: Correlation Heatmap — assess
- [x] Section 12: Renewable Share vs Price — assess

### 1b. Identify analysis gaps
- [x] Autocorrelation / time-series stationarity
- [x] Regime detection (high-price vs low-price)
- [x] Lagged feature importance (what predicts tomorrow's direction?)
- [x] Forecast error analysis (DA forecast accuracy)
- [x] Price change distribution (the actual trading signal)
- [x] Volatility clustering
- [x] Seasonal decomposition
- [x] Cross-border flow impact on price direction
- [x] Other gaps discovered during audit

### 1c. Write audit report
- [x] Document findings per section in this file
- [x] Rank gaps by trading-signal relevance
- [x] Create prioritized list for Phase 2

## Key Questions to Answer

1. Which existing analyses are most relevant to predicting price direction?
2. What critical signal-generating analyses are completely missing?
3. Which sections are surface-level descriptive vs actionable for trading?
4. What data transformations would reveal hidden patterns?

---

## Critical Context: EDA vs Trading Signal Disconnect

The EDA dashboard operates on the **hourly** parquet dataset (`dataset_de_lu.parquet`),
while strategies trade on the **daily** challenge CSV with:
- 25 lagged features (D-1 daily means of hourly data)
- 4 same-day forecast features (DA forecasts for day D)
- Label: `target_direction` = sign(settlement_price_D - settlement_price_{D-1})

**Implication**: Any hourly EDA insight only matters if it survives daily aggregation
and a 1-day lag. The most critical missing analysis is on the **daily price change**
itself — the exact quantity strategies must predict.

---

## Audit Findings — Section by Section

### Section 1: Dataset Overview (`_section_overview`, lines 200-229)

**What it does**: Row count, column count, date range, hourly frequency. Descriptive
stats table (count, mean, std, min, 25%, 50%, 75%, max). Missing data per column.

**Quality**: GOOD — standard data profiling. The missing-data check is useful.

**Trading relevance**: LOW — provides data-quality confidence but no trading signals.
Descriptive stats on raw hourly values don't map directly to daily direction prediction.

**Gaps**:
- No stationarity check (ADF test on price level vs price change)
- No mention of which columns become challenge features vs labels
- No train/validation/test split awareness (all years treated equally)

---

### Section 2: Day-Ahead Price Time Series (`_section_price_ts`, lines 232-263)

**What it does**: Line chart of price at hourly/daily/weekly/monthly aggregation.
Five summary metrics: mean, median, std, min, max.

**Quality**: ADEQUATE — good for visual overview, aggregation toggle is nice.

**Trading relevance**: LOW-MEDIUM — shows price levels and trends, but strategies
predict **price changes** not levels. The 2022 energy crisis spike is visible but
the chart doesn't show whether momentum or mean-reversion dominates.

**Gaps**:
- No **price change** (returns) plot — this is the actual signal
- No rolling volatility overlay
- No regime annotation (e.g., 2022 crisis period highlighted)
- No autocorrelation analysis (is tomorrow's price change predictable from today's?)
- No year-over-year comparison view

---

### Section 3: Generation Mix (`_section_generation`, lines 266-306)

**What it does**: Stacked area chart of 17 generation types over time. Renewable
share % line chart below.

**Quality**: GOOD — clean stacked area with custom color scheme. Display order is
logical (renewables first, then fossils, then nuclear/other).

**Trading relevance**: MEDIUM — generation mix drives price (merit order), and 7
generation columns appear as lagged features. Renewable share is highly correlated
with price (negative).

**Gaps**:
- No **residual load** (load - renewables) which is the key merit-order driver
- No fossil dispatch by fuel type over time (shows marginal generator switching)
- No generation vs price overlay (which fuel sets the price?)
- The renewable share chart is useful but doesn't show **change** in share, which
  would better predict price direction

---

### Section 4: Load & Forecasts (`_section_load`, lines 309-344)

**What it does**: Two charts — load actual vs forecast, and wind/solar DA forecasts.
Both with daily/weekly/monthly aggregation.

**Quality**: ADEQUATE — plots the right things but misses the most important analysis.

**Trading relevance**: HIGH potential, currently LOW delivery — load forecast and
renewable forecasts are among the 4 same-day features in the challenge. But the
section doesn't analyze **forecast errors**, which would reveal whether forecast
accuracy varies by season, regime, or magnitude (a direct trading signal).

**Gaps**:
- **No forecast error analysis** (forecast - actual). This is critical:
  - Systematic forecast bias = directional trading signal
  - Error magnitude = confidence signal
  - Error by season/weather = conditional strategy
- No load forecast vs actual scatter plot
- No solar/wind forecast accuracy metrics (MAE, RMSE, bias)
- No comparison of forecast errors across different conditions

---

### Section 5: Neighbour Prices & Cross-Border Flows (`_section_neighbours`, lines 347-396)

**What it does**: Two-column layout. Left: DE-LU price vs 9 neighbour zone prices.
Right: net cross-border flows (2 borders: FR, NL).

**Quality**: GOOD — shows price convergence/divergence across zones. The flow sign
convention (+ = import) is clearly labeled.

**Trading relevance**: MEDIUM — 6 neighbour prices and 2 flow columns are lagged
features. Price spread with neighbours can indicate arbitrage pressure.

**Gaps**:
- No **price spread** (DE-LU minus neighbour) analysis — spreads predict flow
  direction which affects next-day price
- No flow vs price scatter (does high import predict lower price?)
- No analysis of which neighbours lead or lag DE-LU price changes
- Only 2 of ~6+ possible flow borders are shown (data may have more)

---

### Section 6: Carbon & Gas Prices (`_section_commodities`, lines 399-418)

**What it does**: Single line chart of carbon and gas prices over time with
aggregation toggle.

**Quality**: MINIMAL — just a time series plot, nothing more.

**Trading relevance**: MEDIUM — gas price is a key marginal cost driver (gas plants
set the clearing price when they're marginal). Carbon price affects all fossil costs.
Both are lagged features.

**Gaps**:
- No gas-price vs electricity-price scatter/correlation
- No analysis of gas-price changes driving electricity price changes
- No fuel-switching threshold analysis (at what gas/carbon price does coal vs gas flip?)
- No **spark spread** (electricity - gas) or **dark spread** (electricity - coal)
  analysis, which directly captures generator profitability and dispatch decisions

---

### Section 7: Weather Overview (`_section_weather`, lines 421-442)

**What it does**: Single weather variable selector, daily mean time series.

**Quality**: MINIMAL — one variable at a time, just a line chart.

**Trading relevance**: LOW-MEDIUM — temperature, wind speed, and solar radiation
are lagged features. Temperature drives load (heating/cooling), wind/solar drive
renewable generation. But viewing them individually as time series is not very
informative for trading.

**Gaps**:
- No weather vs price correlation/scatter
- No conditional analysis (price behavior in hot vs cold days)
- No wind speed vs wind generation validation
- No multi-variable view (temperature + wind + solar together)
- No extreme weather event identification

---

### Section 8: Key Correlations with Price (`_section_correlations`, lines 445-474)

**What it does**: Horizontal bar chart of Pearson correlation between each feature
and `price_eur_mwh`. Sorted, color-coded by sign/magnitude.

**Quality**: GOOD — clear visualization, includes all relevant columns. The
color-coding on RdBu scale is effective.

**Trading relevance**: MEDIUM — shows which features move with price, but:
- Correlations are on **levels** not **changes** (a feature correlated with price
  level doesn't necessarily predict price *direction*)
- Pearson is linear only; non-linear relationships are missed
- No lag structure (should correlate D-1 features with D price change)

**Gaps**:
- No **correlation of changes** (diff() of features vs diff() of price)
- No **lagged correlation** (feature at D-1 vs price change at D)
- No non-linear correlation (Spearman rank, mutual information)
- No correlation stability over time (does feature importance shift by regime?)

---

### Section 9: Price Distribution & Temporal Patterns (`_section_distributions`, lines 477-576)

**What it does**: Four sub-charts:
1. Price histogram with box marginal
2. Price by year (box plots)
3. Hourly price profile (mean, median, +/-1 std band)
4. Monthly + day-of-week price profiles

**Quality**: GOOD — this is the richest section. The hourly profile with uncertainty
bands is well done.

**Trading relevance**: LOW-MEDIUM — Shows price level patterns (higher in winter,
peak hours, weekdays). But trading is on **daily** settlement price **changes**.
The hourly profile collapses into a single daily mean.

**Gaps**:
- No **price change distribution** — this is the critical gap. We need:
  - Histogram of daily price changes (settlement_D - settlement_{D-1})
  - Is the change distribution symmetric? Fat-tailed? Skewed?
  - What % of days are positive vs negative changes? (base rates)
- No **volatility by month/season** (std of daily changes over time)
- No **consecutive direction runs** (is there momentum? mean-reversion?)
- Day-of-week analysis is on price levels, should also show direction win rates

---

### Section 10: Negative Price Analysis (`_section_negative`, lines 579-630)

**What it does**: Count of negative-price hours, % of total, most negative price.
Bar charts of negative hours by year and by hour-of-day. Renewable share comparison
(negative vs non-negative hours).

**Quality**: GOOD — useful niche analysis. The renewable share comparison is
a nice touch showing the renewable-overgeneration driver.

**Trading relevance**: LOW — negative prices occur at the hourly level. The daily
settlement (mean of 24 hours) is rarely negative. Knowing that negative hours
cluster at midday with high solar doesn't directly predict daily direction. However,
days with many negative hours tend to have lower settlements, which could be a
(weakly) predictive signal.

**Gaps**:
- No analysis of daily settlements on days with negative hours vs without
- No connection to trading signal (does yesterday having negative hours predict
  today's price direction?)
- Count is useful but the connection to the daily prediction problem is unstated

---

### Section 11: Correlation Heatmap (`_section_heatmap`, lines 633-659)

**What it does**: Full pairwise correlation matrix of all generation, weather, load,
commodity, and price columns. Plotly imshow with RdBu colorscale.

**Quality**: ADEQUATE — comprehensive but dense (30+ columns). Hard to extract
actionable insights from a 30x30 heatmap.

**Trading relevance**: LOW — same issues as Section 8 (correlations on levels, not
changes). Additionally, the full heatmap is dominated by known collinearities
(e.g., solar gen ~ solar radiation, wind gen ~ wind speed) which are not useful
for trading.

**Gaps**:
- Very redundant with Section 8 (both show price correlations)
- No hierarchical clustering to group correlated features
- No feature selection guidance (which features are redundant?)
- Should show correlations of **changes**, or at least **daily** aggregated data

---

### Section 12: Renewable Share vs Price (`_section_scatter`, lines 662-677)

**What it does**: Scatter plot of renewable share % (x) vs price (y), colored by
year, 10k sample.

**Quality**: ADEQUATE — shows the expected negative relationship. Year coloring
reveals structural shifts (2022 energy crisis lifted the entire curve).

**Trading relevance**: LOW — this is a level-vs-level relationship. For trading,
we'd need **change in renewable share vs price change**.

**Gaps**:
- No **delta-vs-delta** version (change in renewables vs change in price)
- No residual load vs price scatter (more direct merit-order relationship)
- No conditional coloring by season or regime
- The relationship appears non-linear (price floor near 0 at high renewable share)
  but no non-linear fit is shown

---

## Summary Table

| Section | Quality | Trading Relevance | Key Gap |
|---------|---------|-------------------|---------|
| 1. Dataset Overview | GOOD | LOW | No stationarity check, no train/val split awareness |
| 2. Price Time Series | ADEQUATE | LOW-MEDIUM | No price *change* analysis, no volatility, no autocorrelation |
| 3. Generation Mix | GOOD | MEDIUM | No residual load, no generation-vs-price overlay |
| 4. Load & Forecasts | ADEQUATE | LOW (HIGH potential) | **No forecast error analysis** — the biggest miss |
| 5. Neighbour Prices | GOOD | MEDIUM | No price spread analysis, no lead/lag |
| 6. Carbon & Gas | MINIMAL | MEDIUM | No spark/dark spread, no gas-vs-elec correlation |
| 7. Weather | MINIMAL | LOW-MEDIUM | Single-variable only, no weather-vs-price |
| 8. Correlations | GOOD | MEDIUM | Correlations on levels not changes, no lag structure |
| 9. Distributions | GOOD | LOW-MEDIUM | **No price change distribution** — critical miss |
| 10. Negative Prices | GOOD | LOW | Hourly phenomenon, weak connection to daily trading |
| 11. Heatmap | ADEQUATE | LOW | Redundant with S8, too dense, levels not changes |
| 12. Renewable vs Price | ADEQUATE | LOW | Level-vs-level, should be change-vs-change |

---

## Overall Assessment

The existing EDA is a **solid descriptive dashboard** that answers "what does the data
look like?" It covers all major feature groups (generation, load, weather, neighbors,
commodities) with clean visualizations.

However, it is **almost entirely descriptive** and **operates on price levels**. For
a trading challenge where the task is to predict **daily price direction**, the
dashboard has three critical blind spots:

1. **The price change distribution is never shown.** Strategies predict direction;
   we don't even know the base rate (% of up vs down days), the distribution shape,
   or whether there are momentum/mean-reversion patterns.

2. **Forecast errors are never analyzed.** The 4 same-day forecast features are
   arguably the most powerful predictors, and forecast error patterns (systematic
   bias, seasonal variation) would directly suggest trading strategies.

3. **All correlations are on levels, not changes.** A feature correlated with price
   level (e.g., gas price) is different from one that predicts price *direction*.
   The lagged structure (D-1 features → D direction) is never examined.

---

## Gap Priority Ranking for Phase 2

| Priority | Analysis | Rationale |
|----------|----------|-----------|
| **P1** | **Price change (returns) distribution** | This IS the trading signal. Need histogram, base rates (% up/down days), fat tails, momentum/mean-reversion tests. Without this, strategies are designed blind. |
| **P2** | **Autocorrelation & direction persistence** | Do up days follow up days (momentum)? Or do extreme moves revert? ACF of daily price changes + runs test. Directly suggests momentum vs mean-reversion strategies. |
| **P3** | **Forecast error analysis** | DA forecasts (load, solar, wind) are same-day features. Systematic forecast errors → directional signal. Error by season/magnitude → conditional strategies. |
| **P4** | **Lagged feature → direction correlation** | Which D-1 features best predict D's direction? Rank features by predictive power (point-biserial correlation, mutual information with target_direction). This is feature selection for Phase 4. |
| **P5** | **Volatility clustering & regime detection** | Identify high-vol vs low-vol periods. Strategy performance likely varies by regime. GARCH-like conditional volatility or rolling-window std of daily changes. |
| **P6** | **Residual load analysis** | Residual load = load - renewables. This is the key merit-order variable. Plot residual load vs price, check if residual load change predicts price direction. |
| **P7** | **Spark spread & dark spread** | Electricity - gas (spark) and electricity - coal (dark) spreads capture generator profitability. Changes in spreads indicate dispatch shifts → price direction. |
| **P8** | **Seasonal decomposition of price changes** | STL decomposition of daily settlement prices → trend + seasonal + residual. The residual is the unpredictable component; seasonal is exploitable. |
| **P9** | **Cross-border spread → direction** | DE-LU vs neighbour price spread changes. If DE-LU was cheaper than FR yesterday, does that predict upward pressure today (convergence)? |
| **P10** | **Day-of-week & month direction win rates** | Extension of existing temporal patterns but for direction rather than price level. Are Mondays more likely up? Is January bullish? |

---

## Answers to Key Questions

### 1. Which existing analyses are most relevant to predicting price direction?
- **Section 8 (Correlations)** and **Section 5 (Neighbour Prices)** are closest,
  but both need transformation from levels to changes.

### 2. What critical signal-generating analyses are completely missing?
- Price change distribution (P1)
- Autocorrelation / direction persistence (P2)
- Forecast error analysis (P3)
- Lagged feature importance for direction (P4)

### 3. Which sections are surface-level descriptive vs actionable for trading?
- **Surface-level**: Sections 1, 7, 10, 11, 12
- **Has potential but needs rework**: Sections 2, 4, 5, 6, 8, 9
- **Solid foundation**: Section 3, 9 (temporal patterns)

### 4. What data transformations would reveal hidden patterns?
- `diff()` on all features and price → change-based correlations
- `shift(1)` to create proper lagged predictors → feature importance ranking
- Residual load = load - renewable generation → merit-order proxy
- Spark spread = electricity - gas → generator profitability signal
- Rolling window statistics (volatility, Sharpe) → regime identification
