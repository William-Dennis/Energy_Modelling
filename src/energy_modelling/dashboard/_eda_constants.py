"""Shared constants and configuration for the EDA dashboard tab."""

from __future__ import annotations

from pathlib import Path

DATA_PATH = Path("data/processed/dataset_de_lu.parquet")
BACKTEST_DATA_PATH = Path("data/backtest/daily_public.csv")
PRICE_COL = "price_eur_mwh"

GEN_COLS_DISPLAY = {
    "gen_biomass_mw": "Biomass",
    "gen_fossil_brown_coal_lignite_mw": "Lignite",
    "gen_fossil_coal_derived_gas_mw": "Coal Gas",
    "gen_fossil_gas_mw": "Natural Gas",
    "gen_fossil_hard_coal_mw": "Hard Coal",
    "gen_fossil_oil_mw": "Oil",
    "gen_geothermal_mw": "Geothermal",
    "gen_hydro_pumped_storage_mw": "Pumped Storage",
    "gen_hydro_run_of_river_and_poundage_mw": "Run of River",
    "gen_hydro_water_reservoir_mw": "Water Reservoir",
    "gen_nuclear_mw": "Nuclear",
    "gen_other_mw": "Other",
    "gen_other_renewable_mw": "Other Renewable",
    "gen_solar_mw": "Solar",
    "gen_waste_mw": "Waste",
    "gen_wind_offshore_mw": "Wind Offshore",
    "gen_wind_onshore_mw": "Wind Onshore",
}

WEATHER_COLS_DISPLAY = {
    "weather_temperature_2m_degc": "Temperature (2m) [C]",
    "weather_relative_humidity_2m_pct": "Humidity (2m) [%]",
    "weather_wind_speed_10m_kmh": "Wind Speed (10m) [km/h]",
    "weather_wind_speed_100m_kmh": "Wind Speed (100m) [km/h]",
    "weather_shortwave_radiation_wm2": "GHI [W/m2]",
    "weather_direct_normal_irradiance_wm2": "DNI [W/m2]",
    "weather_precipitation_mm": "Precipitation [mm]",
}

RENEWABLE_COLS = [
    "gen_solar_mw",
    "gen_wind_onshore_mw",
    "gen_wind_offshore_mw",
    "gen_hydro_run_of_river_and_poundage_mw",
    "gen_hydro_water_reservoir_mw",
    "gen_biomass_mw",
    "gen_other_renewable_mw",
    "gen_geothermal_mw",
]

FOSSIL_COLS = [
    "gen_fossil_gas_mw",
    "gen_fossil_brown_coal_lignite_mw",
    "gen_fossil_hard_coal_mw",
    "gen_fossil_oil_mw",
    "gen_fossil_coal_derived_gas_mw",
]

NEIGHBOUR_PRICE_COLS = {
    "price_fr_eur_mwh": "France",
    "price_nl_eur_mwh": "Netherlands",
    "price_at_eur_mwh": "Austria",
    "price_pl_eur_mwh": "Poland",
    "price_cz_eur_mwh": "Czech Republic",
    "price_dk_1_eur_mwh": "Denmark West",
    "price_dk_2_eur_mwh": "Denmark East",
    "price_be_eur_mwh": "Belgium",
    "price_se_4_eur_mwh": "Sweden South",
}

GEN_COLORS = {
    "Solar": "#FFD700",
    "Wind Onshore": "#1E90FF",
    "Wind Offshore": "#4169E1",
    "Lignite": "#8B4513",
    "Hard Coal": "#333333",
    "Natural Gas": "#FF6347",
    "Nuclear": "#9370DB",
    "Biomass": "#228B22",
    "Run of River": "#00CED1",
    "Water Reservoir": "#20B2AA",
    "Pumped Storage": "#48D1CC",
    "Oil": "#696969",
    "Coal Gas": "#A0522D",
    "Waste": "#808000",
    "Other": "#C0C0C0",
    "Other Renewable": "#32CD32",
    "Geothermal": "#FF8C00",
}

_DISPLAY_ORDER = [
    "Solar",
    "Wind Onshore",
    "Wind Offshore",
    "Run of River",
    "Water Reservoir",
    "Biomass",
    "Other Renewable",
    "Geothermal",
    "Natural Gas",
    "Lignite",
    "Hard Coal",
    "Oil",
    "Coal Gas",
    "Nuclear",
    "Pumped Storage",
    "Waste",
    "Other",
]

_AGG_FREQ = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME"}

_DOW_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
