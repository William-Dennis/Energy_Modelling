# Phase 1: EDA Audit

## Status: NOT STARTED

## Objective

Formally audit the existing 12 EDA sections in `_eda.py`, document what each section
does well, identify gaps, and prioritize which analyses to add in Phase 2.

## Prerequisites

- Phase 0 complete (codebase consolidated, tests green)

## Checklist

### 1a. Review existing 12 sections
- [ ] Section 1: Dataset Overview — assess completeness
- [ ] Section 2: Day-Ahead Price Time Series — assess
- [ ] Section 3: Generation Mix — assess
- [ ] Section 4: Load & Forecasts — assess
- [ ] Section 5: Neighbour Prices & Cross-Border Flows — assess
- [ ] Section 6: Carbon & Gas Prices — assess
- [ ] Section 7: Weather Overview — assess
- [ ] Section 8: Key Correlations with Price — assess
- [ ] Section 9: Price Distribution & Temporal Patterns — assess
- [ ] Section 10: Negative Price Analysis — assess
- [ ] Section 11: Correlation Heatmap — assess
- [ ] Section 12: Renewable Share vs Price — assess

### 1b. Identify analysis gaps
- [ ] Autocorrelation / time-series stationarity
- [ ] Regime detection (high-price vs low-price)
- [ ] Lagged feature importance (what predicts tomorrow's direction?)
- [ ] Forecast error analysis (DA forecast accuracy)
- [ ] Price change distribution (the actual trading signal)
- [ ] Volatility clustering
- [ ] Seasonal decomposition
- [ ] Cross-border flow impact on price direction
- [ ] Other gaps discovered during audit

### 1c. Write audit report
- [ ] Document findings per section in this file
- [ ] Rank gaps by trading-signal relevance
- [ ] Create prioritized list for Phase 2

## Key Questions to Answer

1. Which existing analyses are most relevant to predicting price direction?
2. What critical signal-generating analyses are completely missing?
3. Which sections are surface-level descriptive vs actionable for trading?
4. What data transformations would reveal hidden patterns?

## Audit Findings

*(To be filled in during execution)*

| Section | Quality | Trading Relevance | Gap |
|---------|---------|-------------------|-----|
| 1. Dataset Overview | | | |
| 2. Price Time Series | | | |
| 3. Generation Mix | | | |
| 4. Load & Forecasts | | | |
| 5. Neighbour Prices | | | |
| 6. Carbon & Gas | | | |
| 7. Weather | | | |
| 8. Correlations | | | |
| 9. Price Distribution | | | |
| 10. Negative Prices | | | |
| 11. Heatmap | | | |
| 12. Renewable vs Price | | | |

## Gap Priority Ranking

*(To be filled in during execution)*

| Priority | Analysis | Rationale |
|----------|----------|-----------|
| | | |
