# EDA Part 1: Dataset Overview

This document provides a structural overview of the DE-LU Day-Ahead Power Futures challenge dataset. It covers the data layout, column definitions, availability rules, split structure, and data quality. Understanding the dataset's shape and meaning is the essential first step before any signal analysis.

---

## 1.1 Context and Objective

The dataset supports a **directional trading challenge** on the DE-LU (Germany + Luxembourg) electricity day-ahead futures market. Each row represents one calendar day. The task is to predict the direction of the next day's settlement price change:

- **+1** (long) → price is expected to rise
- **−1** (short) → price is expected to fall
- **None** (skip) → no trade taken

Profit and loss is computed as:

```
PnL = (settlement_price − last_settlement_price) × direction × 24
```

The quantity is fixed at 1 MW, so students choose only direction.

---

## 1.2 File Structure

| File | Rows | Columns | Size |
|------|------|---------|------|
| `daily_public.csv` | 2,192 | 37 | ~1.2 MB |
| `daily_public_glossary.csv` | 36 | 3 | ~3.5 KB |

The main dataset spans **2019-01-01 to 2024-12-31** (6 years of daily data).

---

## 1.3 Dataset Splits

| Split | Date Range | Rows | Purpose |
|-------|------------|------|---------|
| `train` | 2019-01-01 → 2023-12-31 | 1,826 | Strategy development and fitting |
| `validation` | 2024-01-01 → 2024-12-31 | 366 | Out-of-sample signal validation |
| `hidden_test` | 2025-01-01 → 2025-12-31 | 365 | Final leaderboard evaluation (not in public file) |

The `split` column in the dataset labels each row. Students must be careful not to train on validation data.

---

## 1.4 Column Groups

All 37 columns fall into four timing groups defined in the glossary:

### Label Columns *(not available on hidden test)*

These are target/outcome columns. They are present in the public file for training but must **not** be used as input features — they are only known after the decision is made.

| Column | Description |
|--------|-------------|
| `settlement_price` | DA futures settlement price (EUR/MWh) for the delivery day |
| `price_change_eur_mwh` | Settlement price minus prior day's settlement |
| `target_direction` | Sign of price change: +1 or −1 |
| `pnl_long_eur` | EUR PnL of a 1 MW long position (×24 h) |
| `pnl_short_eur` | EUR PnL of a 1 MW short position (×24 h) |

### Reference Columns *(always available)*

| Column | Description |
|--------|-------------|
| `delivery_date` | Calendar date for delivery |
| `split` | Dataset split label |
| `last_settlement_price` | Prior day's settlement (the entry price for any trade) |

### Lagged Realised Columns *(t−1 data; available at decision time for day t)*

These are **yesterday's** observed/realised values. They are known at the time the strategy makes its decision.

| Category | Columns |
|----------|---------|
| **Generation** | `gen_solar_mw_mean`, `gen_wind_onshore_mw_mean`, `gen_wind_offshore_mw_mean`, `gen_fossil_gas_mw_mean`, `gen_fossil_hard_coal_mw_mean`, `gen_fossil_brown_coal_lignite_mw_mean`, `gen_nuclear_mw_mean` |
| **Load** | `load_actual_mw_mean` |
| **Weather** | `weather_temperature_2m_degc_mean`, `weather_wind_speed_10m_kmh_mean`, `weather_shortwave_radiation_wm2_mean` |
| **Neighbouring prices** | `price_fr_eur_mwh_mean`, `price_nl_eur_mwh_mean`, `price_at_eur_mwh_mean`, `price_pl_eur_mwh_mean`, `price_cz_eur_mwh_mean`, `price_dk_1_eur_mwh_mean` |
| **Cross-border flows** | `flow_fr_net_import_mw_mean`, `flow_nl_net_import_mw_mean` |
| **Commodity prices** | `carbon_price_usd_mean`, `gas_price_usd_mean` |
| **DE-LU price stats** | `price_mean`, `price_max`, `price_min`, `price_std` |

### Same-Day Forecast Columns *(day-ahead forecasts available before market close)*

These are **today's** day-ahead forecasts, published before the decision window. They are particularly valuable because they describe conditions for the delivery day itself.

| Column | Description |
|--------|-------------|
| `load_forecast_mw_mean` | Day-ahead load forecast for DE-LU (MW) |
| `forecast_solar_mw_mean` | Day-ahead solar generation forecast (MW) |
| `forecast_wind_onshore_mw_mean` | Day-ahead onshore wind forecast (MW) |
| `forecast_wind_offshore_mw_mean` | Day-ahead offshore wind forecast (MW) |

---

## 1.5 Data Quality

Missing values are minimal (<1% of rows):

| Column | Missing Values | Notes |
|--------|----------------|-------|
| `carbon_price_usd_mean` | 15 | Market holidays or data gaps |
| `gas_price_usd_mean` | 14 | Market holidays or data gaps |
| `weather_*` (3 columns) | 1 each | Single row gap |
| `price_std` | 1 | Single row gap |

**Recommended handling:** Forward-fill or backward-fill for commodity prices (they do not trade on public holidays). Weather data can be linearly interpolated.

No columns are entirely null. No duplicate delivery dates exist.

---

## 1.6 Structural Observations for Students

1. **The label columns are your target** — `target_direction` is what you are predicting. `pnl_long_eur` and `pnl_short_eur` let you compute the financial outcome directly.

2. **Lagged vs. same-day timing** — Realised generation and price data are from *yesterday*, but forecasts are from *today*. This timing distinction matters: a strategy using `gen_wind_onshore_mw_mean` is using yesterday's wind, while one using `forecast_wind_onshore_mw_mean` is using today's forecast.

3. **Nuclear generation changes structurally in 2023** — Germany completed its nuclear phase-out in April 2023. The `gen_nuclear_mw_mean` column is zero for all of 2023. This structural break affects supply-side features.

4. **Price levels vary dramatically across years** — Average settlement prices ranged from ~30 EUR/MWh in 2020 to ~235 EUR/MWh in 2022 (energy crisis). Raw price levels may not be directly comparable across years without normalisation.

5. **Holiday effects** — Carbon and gas price columns have 14–15 missing values corresponding to market holidays. The underlying delivery date is never missing.

---

## 1.7 Recommended First Steps in Python

```python
import pandas as pd

# Load data
df = pd.read_csv("data/backtest/daily_public.csv", parse_dates=["delivery_date"])

# Check shape
print(df.shape)  # (2192, 37)

# Split distribution
print(df['split'].value_counts())
# train         1826
# validation     366

# Check for missing values
print(df.isnull().sum()[df.isnull().sum() > 0])

# Separate train and validation
train = df[df['split'] == 'train'].copy()
val   = df[df['split'] == 'validation'].copy()

# Add useful date features
for d in [train, val]:
    d['year']    = d['delivery_date'].dt.year
    d['month']   = d['delivery_date'].dt.month
    d['dow']     = d['delivery_date'].dt.day_of_week  # 0=Mon, 6=Sun
    d['quarter'] = d['delivery_date'].dt.quarter
```

---

## 1.8 Glossary Reference

The complete column glossary is available at `data/backtest/daily_public_glossary.csv`. It lists every column's `timing_group` and a short `description`. Cross-referencing the glossary with this document is the recommended starting point for any feature engineering.

---

**Next:** [Part 2 — Target Variable Analysis](02_target_analysis.md)
