# Kaggle Dataset: DE-LU Electricity Market

This document combines the **dataset description** (for the Kaggle page) and the
**upload guide** (for maintainers) into a single reference.

---

## Part 1 — Dataset Description

> Copy the content below into the Kaggle dataset description field.

### DE-LU Electricity Market — Hourly Prices, Generation, Load, Weather & Flows 2019-2025

Comprehensive hourly dataset for the **DE-LU bidding zone** (Germany + Luxembourg, unified since October 2018) combining 10 data sources into a single analysis-ready file.

**61,369 rows x 75 columns | 2019-01-01 00:00 UTC -> 2025-12-31 23:00 UTC**

#### What's included

| # | Group | Columns | Source |
|---|-------|---------|--------|
| 1 | Day-ahead price — DE-LU | `price_eur_mwh` | ENTSO-E Transparency Platform — Day Ahead Prices (A.44) |
| 2 | Generation by fuel type | `gen_biomass_mw`, `gen_fossil_gas_mw`, `gen_solar_mw`, `gen_wind_onshore_mw`, `gen_wind_offshore_mw`, `gen_nuclear_mw`, `gen_fossil_hard_coal_mw`, `gen_fossil_brown_coal_lignite_mw`, ... (19 types) | ENTSO-E Transparency Platform — Actual Generation per Production Type (A.75) |
| 3 | Total load | `load_actual_mw`, `load_forecast_mw` | ENTSO-E Transparency Platform — Total Load Actual and Forecast (A.65) |
| 4 | Wind & solar DA forecast | `forecast_solar_mw`, `forecast_wind_onshore_mw`, `forecast_wind_offshore_mw` | ENTSO-E Transparency Platform — Wind and Solar Forecast (A.69) |
| 5 | Neighbour zone DA prices | `price_fr_eur_mwh`, `price_nl_eur_mwh`, `price_at_eur_mwh`, `price_pl_eur_mwh`, `price_cz_eur_mwh`, `price_dk_1_eur_mwh`, `price_dk_2_eur_mwh`, `price_be_eur_mwh`, `price_se_4_eur_mwh` | ENTSO-E Transparency Platform — Day Ahead Prices (A.44) for 9 neighbouring zones |
| 6 | Cross-border physical flows | `flow_{zone}_export_mw`, `flow_{zone}_import_mw`, `flow_{zone}_net_import_mw` for FR, NL, AT, PL, CZ, DK_1, DK_2, BE, SE_4 (27 columns) | ENTSO-E Transparency Platform — Physical Flows (A.11) |
| 7 | Day-ahead NTC | `ntc_dk_1_export_mw`, `ntc_dk_1_import_mw`, `ntc_dk_2_export_mw`, `ntc_dk_2_import_mw`, `ntc_nl_export_mw`, `ntc_nl_import_mw` | ENTSO-E Transparency Platform — Day Ahead Transfer Capacities (A.61) |
| 8 | EU ETS carbon price proxy | `carbon_price_usd` | WisdomTree Carbon ETC (CARB.L) via Yahoo Finance — USD-denominated EUA futures proxy |
| 9 | Natural gas price proxy | `gas_price_usd` | ICE Dutch TTF Natural Gas Futures (TTF=F) via Yahoo Finance |
| 10 | ERA5 reanalysis weather | `weather_temperature_2m_degc`, `weather_relative_humidity_2m_pct`, `weather_wind_speed_10m_kmh`, `weather_wind_speed_100m_kmh`, `weather_shortwave_radiation_wm2`, `weather_direct_normal_irradiance_wm2`, `weather_precipitation_mm` | Open-Meteo Historical Weather API (ERA5 reanalysis) at 51.5 N, 10.5 E |

#### Key facts

- **61,369 rows**, hourly resolution, all timestamps in UTC
- **75 columns** — every column carries a unit suffix (`_mw`, `_eur_mwh`, `_degc`, `_pct`, `_kmh`, `_wm2`, `_mm`, `_usd`)
- **<1% missing** for 68 of 75 columns; NTC for DK_2 (~26% missing) and NL (~23% missing) have known API data gaps — these are documented and left as NaN rather than imputed
- **Price range**: -500 to +936 EUR/MWh — includes real negative price events (excess renewables) and the 2021-2022 energy crisis spike
- **Nuclear generation** (`gen_nuclear_mw`) goes to zero from April 2023 following Germany's final phase-out
- **Leap years** handled correctly (2020, 2024 have 8,784 rows; non-leap years have 8,760 or 8,761 rows due to DST transitions at UTC boundary)

#### Column dictionary (selected)

| Column | Unit | Description |
|--------|------|-------------|
| `price_eur_mwh` | EUR/MWh | DE-LU day-ahead electricity price |
| `gen_solar_mw` | MW | Actual solar generation |
| `gen_wind_onshore_mw` | MW | Actual onshore wind generation |
| `gen_wind_offshore_mw` | MW | Actual offshore wind generation |
| `gen_nuclear_mw` | MW | Nuclear generation (zero from April 2023) |
| `gen_fossil_gas_mw` | MW | Natural gas generation |
| `gen_fossil_brown_coal_lignite_mw` | MW | Lignite (brown coal) generation |
| `load_actual_mw` | MW | Total actual system load |
| `load_forecast_mw` | MW | Day-ahead total load forecast |
| `forecast_solar_mw` | MW | Day-ahead solar generation forecast |
| `forecast_wind_onshore_mw` | MW | Day-ahead onshore wind forecast |
| `carbon_price_usd` | USD | EU ETS carbon price proxy (CARB.L, daily -> hourly forward-filled) |
| `gas_price_usd` | USD | TTF natural gas futures price (daily -> hourly forward-filled) |
| `weather_temperature_2m_degc` | deg C | 2m air temperature at grid centre (51.5 N, 10.5 E) |
| `weather_shortwave_radiation_wm2` | W/m2 | Global horizontal irradiance |
| `weather_wind_speed_100m_kmh` | km/h | Wind speed at 100m (hub height proxy) |

#### Suggested uses

- **Day-ahead price forecasting** — regression/ML with features from all 10 sources
- **Renewable generation forecasting** — solar and wind with weather drivers
- **Cross-border flow analysis** — DE-LU interconnection utilisation patterns
- **Energy crisis analysis** — 2021-2022 gas/carbon price spike vs. electricity prices
- **Nuclear phase-out impact study** — before/after April 2023 generation mix shift
- **Negative price frequency analysis** — renewable oversupply events

#### Data quality notes

- ENTSO-E generation data has sporadic **consumption sub-columns** for some fuel types — these were dropped if >50% missing across the full period
- `gen_nuclear_mw` and `gen_fossil_coal_derived_gas_mw` report NaN when the plant type is not operating; these NaNs are zero-filled (absence of generation, not missing data)
- NTC columns for FR, PL, BE, SE_4, AT, CZ were dropped entirely due to >50% missing; DK_1, DK_2, NL retained despite partial gaps
- Carbon and gas prices are **daily data forward-filled to hourly** — there is no intraday variation in these columns
- Weather point is a single representative grid cell for Germany; it does not capture spatial heterogeneity across the large DE-LU zone

#### Data sources & licences

| Source | Data | Licence |
|--------|------|---------|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Prices, generation, load, forecasts, flows, NTC | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Open-Meteo](https://open-meteo.com/) / [Copernicus ERA5](https://www.copernicus.eu/en/access-data) | Historical weather reanalysis | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Yahoo Finance](https://finance.yahoo.com/) via [yfinance](https://github.com/ranaroussi/yfinance) | CARB.L (WisdomTree Carbon ETC), TTF=F (ICE TTF Gas Futures) | Yahoo Finance Terms of Service — for research and educational use only |

**Please cite appropriately when using this dataset.**

#### Reproducibility

This dataset was built with a fully reproducible open-source pipeline:

- **ENTSO-E**: [`entsoe-py`](https://github.com/EnergieID/entsoe-py) Python client
- **Weather**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- **Commodity prices**: [`yfinance`](https://github.com/ranaroussi/yfinance)
- **Stack**: Python 3.11, pandas, pyarrow, uv

The pipeline fetches, cleans, and joins all sources year-by-year and stores both Parquet (primary) and CSV formats.

---

## Part 2 — Upload Guide

### Prerequisites

1. **Kaggle account** — create one at https://www.kaggle.com if needed.
2. **Kaggle API token** — download from https://www.kaggle.com/settings -> "API" -> "Create New Token".
   Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. **kaggle CLI** — already included in this project's dependencies. Verify with:
   ```
   uv run kaggle --version
   ```

### Step 1 — Regenerate the dataset

Ensure all raw data is fresh and the processed files are current:

```
# Re-run all data fetching (skips cached files — fast)
uv run collect-data --years 2019 --years 2020 --years 2021 --years 2022 --years 2023 --years 2024 --years 2025 --step all --kaggle

# Or force a full re-fetch from APIs (slow — ~30 min total due to ENTSO-E generation queries)
uv run collect-data --years 2019 ... --step all --kaggle --force
```

To scan year availability across all 10 sources (dry-run, uses cached files):

```
uv run python scripts/scan_years.py
```

### Step 2 — Validate the processed files

Run this before uploading. Checks should all pass.

```python
uv run python - <<'EOF'
import pandas as pd, json, os, sys

parquet = "data/processed/dataset_de_lu.parquet"
csv_path = "data/processed/dataset_de_lu.csv"
meta_path = "data/processed/dataset_metadata.json"

# 1. Files exist
for f in [parquet, csv_path, meta_path]:
    assert os.path.exists(f), f"MISSING: {f}"
    print(f"  OK  {f}  ({os.path.getsize(f)/1024/1024:.1f} MB)")

df = pd.read_parquet(parquet)

# 2. Shape sanity
assert df.shape[0] > 50000, f"Too few rows: {df.shape[0]}"
assert df.shape[1] > 50,    f"Too few columns: {df.shape[1]}"
print(f"\n  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# 3. Index is UTC datetime
assert str(df.index.dtype) == "datetime64[ns, UTC]", "Index is not UTC datetime"
print(f"  Date range: {df.index.min()} -> {df.index.max()}")

# 4. No all-NaN columns
all_nan = [c for c in df.columns if df[c].isna().all()]
assert not all_nan, f"All-NaN columns: {all_nan}"

# 5. No constant-value columns
constant = [c for c in df.columns if df[c].nunique() <= 1]
assert not constant, f"Constant columns: {constant}"

# 6. All columns have unit suffixes
valid_suffixes = ["_mw", "_eur_mwh", "_degc", "_pct", "_kmh", "_wm2", "_mm", "_usd"]
bad_suffix = [c for c in df.columns if not any(c.endswith(s) for s in valid_suffixes)]
assert not bad_suffix, f"Missing unit suffix: {bad_suffix}"

# 7. Target column exists and has reasonable values
assert "price_eur_mwh" in df.columns
assert df["price_eur_mwh"].notna().mean() > 0.99, "price_eur_mwh has too many NaNs"
print(f"  Price range: {df['price_eur_mwh'].min():.1f} .. {df['price_eur_mwh'].max():.1f} EUR/MWh")

# 8. Missing data report
missing = (df.isna().mean() * 100).round(2)
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\n  Columns with missing data:")
for col, pct in missing.items():
    flag = "  WARNING >25%" if pct > 25 else ""
    print(f"    {col}: {pct}%{flag}")

# 9. Metadata valid
meta = json.load(open(meta_path))
assert meta["quality"]["total_rows"] == df.shape[0]
print(f"\n  Metadata: {len(meta['quality']['columns'])} columns documented")

print("\nAll checks passed. Dataset is ready to upload.")
EOF
```

Expected output summary:
- **~19 MB** Parquet, **~37 MB** CSV
- **61,369 rows x 75 columns**
- Date range: 2018-12-31 23:00 UTC to 2025-12-31 23:00 UTC
- Columns with >5% missing: only `ntc_dk_2_*` (~26%) and `ntc_nl_*` (~23%) — acceptable, documented

### Step 3 — Prepare the dataset folder

Kaggle expects a directory containing the files to upload plus a `dataset-metadata.json`.

```
mkdir -p kaggle_upload
copy data\processed\dataset_de_lu.csv kaggle_upload\
copy data\processed\dataset_de_lu.parquet kaggle_upload\
```

Create `kaggle_upload/dataset-metadata.json` (Kaggle's required format — different from our internal metadata):

```json
{
  "title": "DE-LU Electricity Market — Hourly Prices, Generation, Load, Weather, Flows 2019-2025",
  "id": "YOUR_KAGGLE_USERNAME/de-lu-electricity-market",
  "licenses": [{"name": "CC BY 4.0"}],
  "keywords": [
    "electricity", "energy", "germany", "day-ahead prices",
    "time series", "forecasting", "entsoe", "renewable energy"
  ],
  "collaborators": [],
  "data": [
    {
      "description": "Hourly DE-LU electricity market dataset — primary file (Parquet)",
      "name": "dataset_de_lu.parquet",
      "totalBytes": 20000000,
      "columns": []
    },
    {
      "description": "Hourly DE-LU electricity market dataset — Kaggle-ready CSV",
      "name": "dataset_de_lu.csv",
      "totalBytes": 37000000,
      "columns": []
    }
  ]
}
```

Replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username.

### Step 4 — Create the dataset on Kaggle (first time only)

```
uv run kaggle datasets create -p kaggle_upload --dir-mode zip
```

Check the upload succeeded:

```
uv run kaggle datasets status YOUR_KAGGLE_USERNAME/de-lu-electricity-market
```

### Step 5 — Write the dataset description on Kaggle

After creation, go to the dataset page and click "Edit" to add a description.
Copy the content from **Part 1** above into the description field.

### Step 6 — Update the dataset (subsequent releases)

After regenerating the data (step 1) and validating (step 2):

```
uv run kaggle datasets version -p kaggle_upload -m "Refresh to include YYYY data" --dir-mode zip
```

### Step 7 — Make the dataset public

1. Go to the dataset page
2. Click the **Settings** tab
3. Under **Visibility**, change from **Private** to **Public**
4. Confirm

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `kaggle: command not found` | Run `uv run kaggle` instead of bare `kaggle` |
| `401 Unauthorized` | Check `~/.kaggle/kaggle.json` exists and has correct credentials |
| `403 Forbidden` on dataset create | Ensure your `id` in `dataset-metadata.json` matches your username |
| Upload stalls | Files > 500 MB require `--dir-mode zip` — already included above |
| Column values look wrong | Re-run step 1 with `--force` to rebuild from APIs, then re-validate |
| NTC columns all-NaN on Kaggle preview | Expected — DK_2 and NL NTC are ~23-26% missing, documented in metadata |
