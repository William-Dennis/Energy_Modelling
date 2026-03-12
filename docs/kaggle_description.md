# DE-LU Electricity Market — Hourly Prices, Generation, Load, Weather & Flows 2019–2025

Comprehensive hourly dataset for the **DE-LU bidding zone** (Germany + Luxembourg, unified since October 2018) combining 10 data sources into a single analysis-ready file.

**61,369 rows × 75 columns | 2019-01-01 00:00 UTC → 2025-12-31 23:00 UTC**

---

## What's included

| # | Group | Columns | Source |
|---|-------|---------|--------|
| 1 | Day-ahead price — DE-LU | `price_eur_mwh` | ENTSO-E Transparency Platform — Day Ahead Prices (A.44) |
| 2 | Generation by fuel type | `gen_biomass_mw`, `gen_fossil_gas_mw`, `gen_solar_mw`, `gen_wind_onshore_mw`, `gen_wind_offshore_mw`, `gen_nuclear_mw`, `gen_fossil_hard_coal_mw`, `gen_fossil_brown_coal_lignite_mw`, … (19 types) | ENTSO-E Transparency Platform — Actual Generation per Production Type (A.75) |
| 3 | Total load | `load_actual_mw`, `load_forecast_mw` | ENTSO-E Transparency Platform — Total Load Actual and Forecast (A.65) |
| 4 | Wind & solar DA forecast | `forecast_solar_mw`, `forecast_wind_onshore_mw`, `forecast_wind_offshore_mw` | ENTSO-E Transparency Platform — Wind and Solar Forecast (A.69) |
| 5 | Neighbour zone DA prices | `price_fr_eur_mwh`, `price_nl_eur_mwh`, `price_at_eur_mwh`, `price_pl_eur_mwh`, `price_cz_eur_mwh`, `price_dk_1_eur_mwh`, `price_dk_2_eur_mwh`, `price_be_eur_mwh`, `price_se_4_eur_mwh` | ENTSO-E Transparency Platform — Day Ahead Prices (A.44) for 9 neighbouring zones |
| 6 | Cross-border physical flows | `flow_{zone}_export_mw`, `flow_{zone}_import_mw`, `flow_{zone}_net_import_mw` for FR, NL, AT, PL, CZ, DK_1, DK_2, BE, SE_4 (27 columns) | ENTSO-E Transparency Platform — Physical Flows (A.11) |
| 7 | Day-ahead NTC | `ntc_dk_1_export_mw`, `ntc_dk_1_import_mw`, `ntc_dk_2_export_mw`, `ntc_dk_2_import_mw`, `ntc_nl_export_mw`, `ntc_nl_import_mw` | ENTSO-E Transparency Platform — Day Ahead Transfer Capacities (A.61) |
| 8 | EU ETS carbon price proxy | `carbon_price_usd` | WisdomTree Carbon ETC (CARB.L) via Yahoo Finance — USD-denominated EUA futures proxy |
| 9 | Natural gas price proxy | `gas_price_usd` | ICE Dutch TTF Natural Gas Futures (TTF=F) via Yahoo Finance |
| 10 | ERA5 reanalysis weather | `weather_temperature_2m_degc`, `weather_relative_humidity_2m_pct`, `weather_wind_speed_10m_kmh`, `weather_wind_speed_100m_kmh`, `weather_shortwave_radiation_wm2`, `weather_direct_normal_irradiance_wm2`, `weather_precipitation_mm` | Open-Meteo Historical Weather API (ERA5 reanalysis) at 51.5°N, 10.5°E |

---

## Key facts

- **61,369 rows**, hourly resolution, all timestamps in UTC
- **75 columns** — every column carries a unit suffix (`_mw`, `_eur_mwh`, `_degc`, `_pct`, `_kmh`, `_wm2`, `_mm`, `_usd`)
- **<1% missing** for 68 of 75 columns; NTC for DK_2 (~26% missing) and NL (~23% missing) have known API data gaps — these are documented and left as NaN rather than imputed
- **Price range**: −500 to +936 EUR/MWh — includes real negative price events (excess renewables) and the 2021–2022 energy crisis spike
- **Nuclear generation** (`gen_nuclear_mw`) goes to zero from April 2023 following Germany's final phase-out
- **Leap years** handled correctly (2020, 2024 have 8,784 rows; non-leap years have 8,760 or 8,761 rows due to DST transitions at UTC boundary)

---

## Column dictionary (selected)

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
| `carbon_price_usd` | USD | EU ETS carbon price proxy (CARB.L, daily → hourly forward-filled) |
| `gas_price_usd` | USD | TTF natural gas futures price (daily → hourly forward-filled) |
| `weather_temperature_2m_degc` | °C | 2m air temperature at grid centre (51.5°N, 10.5°E) |
| `weather_shortwave_radiation_wm2` | W/m² | Global horizontal irradiance |
| `weather_wind_speed_100m_kmh` | km/h | Wind speed at 100m (hub height proxy) |

---

## Suggested uses

- **Day-ahead price forecasting** — regression/ML with features from all 10 sources
- **Renewable generation forecasting** — solar and wind with weather drivers
- **Cross-border flow analysis** — DE-LU interconnection utilisation patterns
- **Energy crisis analysis** — 2021–2022 gas/carbon price spike vs. electricity prices
- **Nuclear phase-out impact study** — before/after April 2023 generation mix shift
- **Negative price frequency analysis** — renewable oversupply events

---

## Data quality notes

- ENTSO-E generation data has sporadic **consumption sub-columns** for some fuel types — these were dropped if >50% missing across the full period
- `gen_nuclear_mw` and `gen_fossil_coal_derived_gas_mw` report NaN when the plant type is not operating; these NaNs are zero-filled (absence of generation, not missing data)
- NTC columns for FR, PL, BE, SE_4, AT, CZ were dropped entirely due to >50% missing; DK_1, DK_2, NL retained despite partial gaps
- Carbon and gas prices are **daily data forward-filled to hourly** — there is no intraday variation in these columns
- Weather point is a single representative grid cell for Germany; it does not capture spatial heterogeneity across the large DE-LU zone

---

## Data sources & licences

| Source | Data | Licence |
|--------|------|---------|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Prices, generation, load, forecasts, flows, NTC | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — © ENTSO-E |
| [Open-Meteo](https://open-meteo.com/) / [Copernicus ERA5](https://www.copernicus.eu/en/access-data) | Historical weather reanalysis | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Contains modified Copernicus Climate Change Service information |
| [Yahoo Finance](https://finance.yahoo.com/) via [yfinance](https://github.com/ranaroussi/yfinance) | CARB.L (WisdomTree Carbon ETC), TTF=F (ICE TTF Gas Futures) | Yahoo Finance Terms of Service — for research and educational use only |

**Please cite appropriately when using this dataset.** For ENTSO-E data, attribution is required under CC BY 4.0. ERA5 data requires the Copernicus attribution statement. Yahoo Finance data may not be redistributed commercially.

---

## Reproducibility

This dataset was built with a fully reproducible open-source pipeline:

- **ENTSO-E**: [`entsoe-py`](https://github.com/EnergieID/entsoe-py) Python client
- **Weather**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- **Commodity prices**: [`yfinance`](https://github.com/ranaroussi/yfinance)
- **Stack**: Python 3.11, pandas, pyarrow, uv

The pipeline fetches, cleans, and joins all sources year-by-year and stores both Parquet (primary) and CSV formats.
