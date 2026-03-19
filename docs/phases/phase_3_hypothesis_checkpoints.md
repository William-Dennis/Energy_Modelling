# Phase 3: Hypothesis Checkpoints

## Status: COMPLETE

## Objective

Identify natural hypotheses that emerge from the EDA analysis (Phases 1-2) and
translate them into concrete, testable trading strategy ideas. Each hypothesis
should have a clear signal, expected edge, and implementation approach.

## Prerequisites

- Phase 2 complete (deepened EDA with pattern discoveries)

## Checklist

### 3a. Extract hypotheses from EDA
- [x] List all patterns discovered in EDA that suggest predictable price direction
- [x] For each pattern, state the hypothesis formally
- [x] Estimate the expected edge (based on EDA statistics)
- [x] Identify the feature(s) driving the signal

### 3b. Prioritize by expected Sharpe
- [x] Rank hypotheses by signal strength (from EDA correlation/importance analysis)
- [x] Consider implementation complexity
- [x] Consider robustness (does it hold across years?)
- [x] Select top candidates for Phase 4 implementation

### 3c. Define strategy specifications
- [x] For each selected hypothesis, define entry signal, features, parameters, expected behavior

---

## Key EDA Findings (Data-Driven)

### Base Rates (2019-2024, 2192 days)
- **53.3% of days are down**, 46.7% up — persistent downward bias
- Mean up move: 22.93 EUR, mean down move: 20.08 EUR — up moves are larger
- Kurtosis: 8.84 — heavy tails, extreme days are common
- Base rate is stable across train (46.8% up) and validation (46.4% up)

### Day-of-Week Effect (strongest pattern found)
- **Monday: 90.7% up** — virtually certain (stable 85-94% across all 6 years)
- **Tuesday: 59.6% up** — moderate edge
- **Saturday: 14.4% up** (85.6% down) — strong reverse signal
- **Sunday: 26.5% up** (73.5% down) — strong reverse signal
- **Friday: 38.7% up** — moderate down bias
- Mid-week (Wed-Thu): ~50%, no edge

### Autocorrelation
- Lag 1: -0.023 (insignificant) — no simple momentum or mean-reversion
- **Lag 2: -0.277** (significant) — strong 2-day mean reversion
- **Lag 7: +0.297** (significant) — weekly cycle (same-day-of-week persistence)
- Transition matrix nearly symmetric: after up → 46.9% up, after down → 46.6% up

### Feature Importance (D-1 features → D direction)
Top predictors:
1. `load_forecast_mw_mean`: +0.234 (higher load forecast → more likely up)
2. `forecast_wind_offshore_mw_mean`: -0.218 (higher wind forecast → more likely down)
3. `gen_wind_onshore_mw_mean`: +0.211 (higher yesterday wind → more likely up today — counterintuitive)
4. `weather_wind_speed_10m_kmh_mean`: +0.211 (same as above)
5. `gen_fossil_gas_mw_mean`: -0.196 (higher gas gen → more likely down)
6. `forecast_wind_onshore_mw_mean`: -0.189 (higher wind forecast → down)

Near-zero predictors: solar forecast (0.002), solar generation (0.010), carbon price (0.008)

### Volatility Regimes
- 2022 had 5.6x the volatility of 2019 (std 65 vs 12)
- High-vol regime: 48.5% up, moves ~35 EUR avg
- Low-vol regime: 45.0% up, moves ~10 EUR avg
- Direction prediction slightly easier in low-vol (stronger down bias)

---

## Hypotheses

### H1: Day-of-Week Calendar Strategy

**EDA evidence**:
- *Section 13 — Price Change Distribution*: Day-of-week % up breakdown
- *Section 19 — Day-of-Week Edge Stability*: Year-by-year validation that Monday/Saturday edges persist

**Pattern observed**: Monday has 90.7% up rate (stable 85-94% across all years).
Saturday has 14.4% up (85.6% down). Sunday 26.5% up. The pattern is the strongest
and most robust signal in the entire dataset.

**Hypothesis**: A strategy that goes long on Monday, short on Saturday/Sunday, and
adapts behavior on other days will significantly outperform naive strategies.

**Signal**:
- Monday → long (+1)
- Saturday → short (-1)
- Sunday → short (-1)
- Tuesday → long (+1) (59.6% up, marginal edge)
- Friday → short (-1) (38.7% up, 61.3% down)
- Wednesday, Thursday → skip (None) (~50/50, no edge)

**Features used**: Day of week (derived from `delivery_date` in ChallengeState)
**Expected edge**: ~65% overall win rate (weighted by trading days)
**Robustness**: Extremely robust — Monday pattern holds 85-94% in every single year
**Implementation complexity**: LOW
**Priority**: 1 (implement first)

**Economic intuition**: Monday settlement = mean of Monday's 24 hourly prices.
Sunday settlement = mean of Sunday's 24 hourly prices. Weekend prices are
structurally lower (low industrial demand). Monday-Sunday transition almost always
positive; Friday-Saturday transition almost always negative.

---

### H2: Wind Forecast Contrarian

**EDA evidence**:
- *Section 15 — Forecast Error Analysis*: Wind forecast error magnitude and direction
- *Section 16 — Feature Importance*: `forecast_wind_offshore_mw_mean` correlation -0.218
- *Section 23 — Wind Quintile Analysis*: Up-rate decreases monotonically with wind quintile

**Pattern observed**: `forecast_wind_offshore_mw_mean` has -0.218 correlation with
direction. `forecast_wind_onshore_mw_mean` has -0.189. When DA wind forecasts are
high, price tends to go down (more supply → lower price).

**Hypothesis**: Going short when wind forecasts exceed a threshold and long when
they are below it captures the renewable-supply effect on prices.

**Signal**:
- `forecast_wind_offshore_mw_mean + forecast_wind_onshore_mw_mean` > threshold → short (-1)
- Below threshold → long (+1)
- Threshold: median of training data (fit during `fit()`)

**Features used**: `forecast_wind_offshore_mw_mean`, `forecast_wind_onshore_mw_mean`
**Expected edge**: ~55% win rate (correlation 0.2 → modest directional edge)
**Robustness**: Wind impact on price is structural (merit order)
**Implementation complexity**: LOW
**Priority**: 2

---

### H3: Load Forecast Level

**EDA evidence**:
- *Section 15 — Forecast Error Analysis*: Load forecast error distribution
- *Section 16 — Feature Importance*: `load_forecast_mw_mean` correlation +0.234 (strongest single feature)

**Pattern observed**: `load_forecast_mw_mean` has +0.234 correlation with direction
(the strongest single feature). Higher load forecast → more likely up.

**Hypothesis**: When load forecast is high relative to recent history, price is more
likely to increase (higher demand → higher clearing price).

**Signal**:
- `load_forecast_mw_mean` > rolling mean of last N training days → long (+1)
- Below → short (-1)

**Features used**: `load_forecast_mw_mean`
**Expected edge**: ~55% win rate
**Robustness**: Load-price relationship is fundamental
**Implementation complexity**: LOW
**Priority**: 3

---

### H4: Lag-2 Mean Reversion

**EDA evidence**:
- *Section 13 — Price Change Distribution*: Heavy tails (kurtosis 8.84) — large moves are common targets for reversion
- *Section 14 — Autocorrelation & Direction Persistence*: Lag-2 ACF = -0.277 (strongly significant)

**Pattern observed**: ACF at lag 2 is -0.277 (strongly significant). Two days after
a large move, price tends to reverse.

**Hypothesis**: If the price change 2 days ago was large and positive, today's price
is more likely to drop (and vice versa).

**Signal**:
- Track last 2 price changes. If change 2 days ago was > threshold → short (-1)
- If change 2 days ago was < -threshold → long (+1)
- Otherwise → skip (None)

**Features used**: `price_mean` (yesterday's mean), `last_settlement_price`, historical tracking
**Expected edge**: Modest, but the lag-2 ACF is strong
**Robustness**: ACF structure appears consistent but needs year-by-year check
**Implementation complexity**: MEDIUM (needs to track state across days)
**Priority**: 4

---

### H5: Weekly Cycle Exploitation

**EDA evidence**:
- *Section 14 — Autocorrelation & Direction Persistence*: Lag-7 ACF = +0.297 (significant positive)
- *Section 13 — Price Change Distribution*: Day-of-week decomposition shows structural weekly pattern

**Pattern observed**: ACF at lag 7 is +0.297 (significant positive). The price change
on the same day of the week tends to repeat direction.

**Hypothesis**: If last week's same day had a positive change, this week's same day
will also tend to be positive.

**Signal**:
- Track the price change from 7 days ago. If positive → long (+1). If negative → short (-1).

**Features used**: Historical price change tracking (7-day lag)
**Expected edge**: Moderate (ACF 0.297 is meaningful)
**Robustness**: The weekly cycle is a structural feature of electricity markets
**Implementation complexity**: MEDIUM
**Priority**: 5

---

### H6: Fossil Dispatch Contrarian

**EDA evidence**:
- *Section 16 — Feature Importance*: `gen_fossil_gas_mw_mean` correlation -0.196
- *Section 18 — Residual Load Analysis*: Residual load (load minus renewables) links fossil dispatch to price

**Pattern observed**: `gen_fossil_gas_mw_mean` (-0.196), `gen_fossil_brown_coal_lignite_mw_mean`
(-0.185), `gen_fossil_hard_coal_mw_mean` (-0.139) all negatively correlated with direction.

**Hypothesis**: When yesterday had high fossil generation, today's price is more likely
to fall (high fossil dispatch signals expensive generation, which tends to mean-revert
as conditions normalize).

**Signal**:
- Combined fossil generation (gas + coal + lignite) > training median → short (-1)
- Below → long (+1)

**Features used**: `gen_fossil_gas_mw_mean`, `gen_fossil_hard_coal_mw_mean`, `gen_fossil_brown_coal_lignite_mw_mean`
**Expected edge**: ~54% (moderate)
**Robustness**: Fossil dispatch as price driver is fundamental
**Implementation complexity**: LOW
**Priority**: 6

---

### H7: Composite Signal (Multi-Feature)

**EDA evidence**:
- *Section 16 — Feature Importance*: Top 6 features have moderate but uncorrelated predictive power
- *Section 17 — Volatility & Regime Analysis*: Regime-dependent signal strength suggests combining features
- *Section 24 — Strategy Correlation Insights*: Decomposition of how individual strategy signals combine

**Pattern observed**: Multiple features have moderate predictive power. Combining
them should improve signal-to-noise ratio.

**Hypothesis**: A weighted sum of the top features (load forecast, wind forecast,
fossil generation) provides a stronger directional signal than any individual feature.

**Signal**:
- Compute z-score of each feature relative to training stats
- Weighted sum: `w_load * z_load + w_wind * z_wind + w_fossil * z_fossil`
- If composite > 0 → long (+1), if < 0 → short (-1)
- Weights: proportional to feature-direction correlation from training

**Features used**: `load_forecast_mw_mean`, `forecast_wind_*_mw_mean`, `gen_fossil_*_mw_mean`
**Expected edge**: ~57% (combining uncorrelated signals)
**Robustness**: Depends on weight stability
**Implementation complexity**: MEDIUM
**Priority**: 7

---

## Strategy Selection for Phase 4

| Priority | Hypothesis | Strategy Name | Complexity |
|----------|-----------|---------------|------------|
| 1 | H1: DOW Calendar | `day_of_week.py` | LOW |
| 2 | H2: Wind Forecast | `wind_forecast.py` | LOW |
| 3 | H3: Load Forecast | `load_forecast.py` | LOW |
| 4 | H4: Lag-2 Reversion | `lag2_reversion.py` | MEDIUM |
| 5 | H5: Weekly Cycle | `weekly_cycle.py` | MEDIUM |
| 6 | H6: Fossil Dispatch | `fossil_dispatch.py` | LOW |
| 7 | H7: Composite | `composite_signal.py` | MEDIUM |

**Phase 4 implementation order**: H1 first (strongest, simplest), then H2-H3 (feature-based),
then H4-H5 (time-series-based), then H6-H7 (ensemble).

---

## EDA Section → Strategy Cross-Reference

| EDA Section | Dashboard Header | Strategies Supported |
|-------------|-----------------|---------------------|
| 13 | Price Change Distribution | H1 (DOW % up), H4 (heavy tails → reversion), H5 (weekly pattern) |
| 14 | Autocorrelation & Direction Persistence | H4 (lag-2 = -0.277), H5 (lag-7 = +0.297) |
| 15 | Forecast Error Analysis | H2 (wind forecast), H3 (load forecast) |
| 16 | Feature Importance for Price Direction | H2, H3, H6, H7 (all feature-based strategies) |
| 17 | Volatility & Regime Analysis | General regime awareness; informs H7 weight stability |
| 18 | Residual Load Analysis | H6 (fossil dispatch linked to residual load) |
| 19 | Day-of-Week Edge Stability | H1 validation (year-by-year robustness) |
| 20 | Feature Drift (Train vs Validation) | All strategies (monitors signal stability) |
| 21 | Quarterly Direction Rates | General (base rate stability check) |
| 22 | Volatility Regime Performance | H4, H7 (regime-dependent signal strength) |
| 23 | Wind Quintile Analysis | H2 validation (monotonic up-rate decline with wind) |
| 24 | Strategy Correlation Insights | H7 (decomposition of composite signal) |
