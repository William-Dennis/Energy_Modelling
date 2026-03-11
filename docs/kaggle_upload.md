# Kaggle Dataset Upload Guide

This guide walks through validating and uploading the DE-LU electricity market dataset to Kaggle.
Do not skip the validation steps — uploading a bad dataset is worse than not uploading one.

---

## Prerequisites

1. **Kaggle account** — create one at https://www.kaggle.com if needed.
2. **Kaggle API token** — download from https://www.kaggle.com/settings → "API" → "Create New Token".
   Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. **kaggle CLI** — already included in this project's dependencies. Verify with:
   ```
   uv run kaggle --version
   ```

---

## Step 1 — Regenerate the dataset (always do this before uploading)

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

To force re-fetching from all APIs:

```
uv run python scripts/scan_years.py --force
```

---

## Step 2 — Validate the processed files

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
- All checks passed

---

## Step 3 — Prepare the dataset folder

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
    "electricity",
    "energy",
    "germany",
    "day-ahead prices",
    "time series",
    "forecasting",
    "entsoe",
    "renewable energy"
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

---

## Step 4 — Create the dataset on Kaggle (first time only)

```
uv run kaggle datasets create -p kaggle_upload --dir-mode zip
```

This:
- Creates a new private dataset on Kaggle
- Zips and uploads the files
- Returns a URL like `https://www.kaggle.com/datasets/YOUR_USERNAME/de-lu-electricity-market`

Check the upload succeeded:

```
uv run kaggle datasets status YOUR_KAGGLE_USERNAME/de-lu-electricity-market
```

---

## Step 5 — Write the dataset description on Kaggle

After creation, go to the dataset page and click "Edit" to add a description. Use this as a starting point:

```markdown
## DE-LU Electricity Market Dataset — 2019 to 2025

Hourly dataset for the **DE-LU bidding zone** (Germany + Luxembourg) combining
10 data sources into a single analysis-ready file with 75 columns.

### What's included

| Group | Columns | Source |
|-------|---------|--------|
| Day-ahead price (DE-LU) | `price_eur_mwh` | ENTSO-E A.44 |
| Generation by fuel type (19 types) | `gen_*_mw` | ENTSO-E A.75 |
| Total load — actual + forecast | `load_actual_mw`, `load_forecast_mw` | ENTSO-E A.65 |
| Wind/solar DA forecasts | `forecast_solar_mw`, `forecast_wind_*_mw` | ENTSO-E A.69 |
| Neighbour zone DA prices (9 zones) | `price_fr_eur_mwh`, ... | ENTSO-E A.44 |
| Cross-border flows (9 borders) | `flow_*_export/import/net_mw` | ENTSO-E A.11 |
| Day-ahead NTC (3 borders) | `ntc_*_export/import_mw` | ENTSO-E A.61 |
| EU ETS carbon price proxy | `carbon_price_usd` | Yahoo Finance CARB.L |
| Gas price proxy | `gas_price_usd` | Yahoo Finance TTF=F |
| ERA5 reanalysis weather (7 vars) | `weather_*` | Open-Meteo |

### Key facts

- **61,369 rows** (2019-01-01 00:00 UTC to 2025-12-31 23:00 UTC, hourly)
- **75 columns** — all with unit suffixes (`_mw`, `_eur_mwh`, `_degc`, etc.)
- **<1% missing** for 68 of 75 columns; NTC columns for DK_2 and NL have ~23-26% missing
- Price range: −500 to +936 EUR/MWh (includes negative price events)
- Nuclear generation goes to zero from April 2023 (Germany phase-out)

### Suggested use

Day-ahead electricity price forecasting (regression), renewable integration analysis,
cross-border flow modelling, energy market EDA.

### Licences

- ENTSO-E data: CC BY 4.0
- Open-Meteo/ERA5: CC BY 4.0 (Copernicus Climate Change Service)
- Yahoo Finance data (carbon, gas): Yahoo Finance Terms of Service — for research/educational use
```

---

## Step 6 — Update the dataset (subsequent releases)

After regenerating the data (step 1) and validating (step 2):

```
uv run kaggle datasets version -p kaggle_upload -m "Refresh to include YYYY data" --dir-mode zip
```

---

## Step 7 — Make the dataset public

Once you are satisfied with the description and have confirmed the data looks correct on Kaggle:

1. Go to the dataset page
2. Click the **Settings** tab
3. Under **Visibility**, change from **Private** to **Public**
4. Confirm

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `kaggle: command not found` | Run `uv run kaggle` instead of bare `kaggle` |
| `401 Unauthorized` | Check `~/.kaggle/kaggle.json` exists and has correct credentials |
| `403 Forbidden` on dataset create | Ensure your `id` in `dataset-metadata.json` matches your username |
| Upload stalls | Files > 500 MB require `--dir-mode zip` — already included above |
| Column values look wrong | Re-run step 1 with `--force` to rebuild from APIs, then re-validate |
| NTC columns all-NaN on Kaggle preview | Expected — DK_2 and NL NTC are ~23-26% missing, documented in metadata |
