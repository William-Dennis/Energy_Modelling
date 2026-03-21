"""Join all data sources into a single dataset."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig

MAX_FFILL_HOURS = 3
MAX_MISSING_PCT = 50.0
ZERO_FILL_GEN_COLS = {"gen_nuclear_mw", "gen_fossil_coal_derived_gas_mw"}

WEATHER_UNITS: dict[str, str] = {
    "temperature_2m": "degc",
    "relative_humidity_2m": "pct",
    "wind_speed_10m": "kmh",
    "wind_speed_100m": "kmh",
    "shortwave_radiation": "wm2",
    "direct_normal_irradiance": "wm2",
    "precipitation": "mm",
}


def _add_gen_unit_suffix(col: str) -> str:
    """Rename a ``gen_*`` column by appending ``_mw`` if not already present."""
    if col.endswith("_mw"):
        return col
    return f"{col}_mw"


def _add_weather_unit_suffix(col: str) -> str:
    """Rename a ``weather_*`` column by appending the correct unit suffix."""
    raw_name = col.removeprefix("weather_")
    unit = WEATHER_UNITS.get(raw_name)
    if unit:
        return f"{col}_{unit}"
    # Unknown weather variable — just append generic
    return col


def _add_forecast_unit_suffix(col: str) -> str:
    """Rename a ``forecast_*`` column by appending ``_mw`` if not present."""
    if col.endswith("_mw"):
        return col
    return f"{col}_mw"


def load_raw_parquet(path: Path, name: str) -> pd.DataFrame:
    """Load a consolidated Parquet file, raising if missing."""
    if not path.exists():
        msg = f"Raw {name} file not found: {path}. Run the download step first."
        raise FileNotFoundError(msg)
    df = pd.read_parquet(path)
    logger.info("Loaded {} — {} rows, {} columns", name, len(df), len(df.columns))
    return df


def _load_optional_parquet(path: Path, name: str) -> pd.DataFrame | None:
    """Load a Parquet file if it exists; return None otherwise."""
    if not path.exists():
        logger.info("Optional {} file not found — skipping: {}", name, path)
        return None
    df = pd.read_parquet(path)
    if df.empty or len(df.columns) == 0:
        logger.info("Optional {} file is empty — skipping", name)
        return None
    logger.info("Loaded {} — {} rows, {} columns", name, len(df), len(df.columns))
    return df


def impute_small_gaps(df: pd.DataFrame, max_gap_hours: int = MAX_FFILL_HOURS) -> pd.DataFrame:
    """Forward-fill gaps of up to *max_gap_hours* consecutive NaN rows."""
    return df.ffill(limit=max_gap_hours)


def compute_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-column missing percentages, row count, and date range."""
    total = len(df)
    missing_pct: dict[str, float] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        missing_pct[col] = round(n_missing / total * 100, 2) if total > 0 else 0.0

    return {
        "total_rows": total,
        "date_range_start": str(df.index.min()) if total > 0 else None,
        "date_range_end": str(df.index.max()) if total > 0 else None,
        "columns": list(df.columns),
        "missing_percent": missing_pct,
    }


def _build_source_descriptions(config: DataCollectionConfig) -> dict[str, str]:
    """Build the sources dictionary for Kaggle metadata."""
    return {
        "prices": "ENTSO-E Transparency Platform (Day Ahead Prices, A.44)",
        "generation": "ENTSO-E Transparency Platform (Actual Generation per Type, A.75)",
        "load": "ENTSO-E Transparency Platform (Total Load, A.65 actual + forecast)",
        "forecasts": ("ENTSO-E Transparency Platform (Wind/Solar DA Forecast, A.69)"),
        "neighbour_prices": (
            f"ENTSO-E Transparency Platform (DA Prices for {', '.join(config.neighbour_zones)})"
        ),
        "flows": ("ENTSO-E Transparency Platform (Cross-border Physical Flows, A.11)"),
        "ntc": ("ENTSO-E Transparency Platform (Day-ahead NTC, A.61)"),
        "carbon_price": ("Yahoo Finance — CARB.L (WisdomTree Carbon ETC, USD proxy for EUA)"),
        "gas_price": ("Yahoo Finance — TTF=F / NG=F (natural gas futures proxy)"),
        "weather": (
            f"Open-Meteo Archive API (ERA5 reanalysis) at"
            f" ({config.weather_latitude}, {config.weather_longitude})"
        ),
    }


def build_kaggle_metadata(
    config: DataCollectionConfig,
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Build a Kaggle-compatible metadata dictionary."""
    return {
        "title": (
            f"DE-LU Electricity Market Dataset — Prices, Generation, Load, "
            f"Weather, Flows, Forecasts, Carbon & Gas ({config.bidding_zone})"
        ),
        "description": (
            "Comprehensive hourly dataset for the DE-LU bidding zone joining: "
            "ENTSO-E day-ahead prices, actual generation per type, total load "
            "(actual + forecast), wind/solar DA forecasts, neighbouring zone "
            "day-ahead prices, cross-border physical flows, day-ahead NTC, "
            "EU ETS carbon price proxy (CARB.L), TTF/NG gas price proxy, "
            "and ERA5 reanalysis weather data.  "
            "All timestamps are UTC.  Column names include unit suffixes."
        ),
        "sources": _build_source_descriptions(config),
        "bidding_zone": config.bidding_zone,
        "neighbour_zones": config.neighbour_zones,
        "timezone": "UTC",
        "weather_coordinates": {
            "latitude": config.weather_latitude,
            "longitude": config.weather_longitude,
        },
        "years": config.years,
        "quality": quality,
        "license": (
            "CC-BY-4.0 (ENTSO-E data), CC-BY-4.0 (Copernicus/ERA5 via Open-Meteo), "
            "Yahoo Finance Terms of Service (carbon & gas)"
        ),
    }


def _load_mandatory_sources(
    config: DataCollectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prefix mandatory data sources."""
    prices = load_raw_parquet(config.raw_dir / "prices_da.parquet", "prices")
    generation = load_raw_parquet(config.raw_dir / "generation.parquet", "generation")
    weather = load_raw_parquet(config.raw_dir / "weather.parquet", "weather")

    generation = generation.add_prefix("gen_")
    weather = weather.add_prefix("weather_")
    generation = generation.rename(columns={c: _add_gen_unit_suffix(c) for c in generation.columns})
    weather = weather.rename(columns={c: _add_weather_unit_suffix(c) for c in weather.columns})
    return prices, generation, weather


def _join_optional_sources(config: DataCollectionConfig, joined: pd.DataFrame) -> pd.DataFrame:
    """Load and join each optional data source."""
    load_df = _load_optional_parquet(config.raw_dir / "load.parquet", "load")
    if load_df is not None:
        joined = joined.join(load_df, how="outer")

    forecasts_df = _load_optional_parquet(config.raw_dir / "forecasts.parquet", "forecasts")
    if forecasts_df is not None:
        forecasts_df = forecasts_df.add_prefix("forecast_")
        forecasts_df = forecasts_df.rename(
            columns={c: _add_forecast_unit_suffix(c) for c in forecasts_df.columns}
        )
        joined = joined.join(forecasts_df, how="outer")

    nb_df = _load_optional_parquet(config.raw_dir / "neighbour_prices.parquet", "neighbour_prices")
    if nb_df is not None:
        joined = joined.join(nb_df, how="outer")

    flows_df = _load_optional_parquet(config.raw_dir / "flows.parquet", "flows")
    if flows_df is not None:
        joined = joined.join(flows_df, how="outer")

    ntc_df = _load_optional_parquet(config.raw_dir / "ntc.parquet", "ntc")
    if ntc_df is not None:
        joined = joined.join(ntc_df, how="outer")

    carbon_df = _load_optional_parquet(config.raw_dir / "carbon_price.parquet", "carbon_price")
    if carbon_df is not None:
        joined = joined.join(carbon_df, how="outer")

    gas_df = _load_optional_parquet(config.raw_dir / "gas_price.parquet", "gas_price")
    if gas_df is not None:
        joined = joined.join(gas_df, how="outer")

    return joined


def _drop_low_quality_columns(joined: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with too much missing data or constant values."""
    pre_quality = compute_data_quality(joined)
    logger.info(
        "Pre-imputation: {} rows, missing: {}",
        pre_quality["total_rows"],
        {k: f"{v}%" for k, v in pre_quality["missing_percent"].items() if v > 0},
    )
    drop_cols = [
        col for col, pct in pre_quality["missing_percent"].items() if pct > MAX_MISSING_PCT
    ]
    if drop_cols:
        joined = joined.drop(columns=drop_cols)
        logger.info(
            "Dropped {} columns with >{:.0f}% missing: {}",
            len(drop_cols),
            MAX_MISSING_PCT,
            drop_cols,
        )
    constant_cols = [col for col in joined.columns if joined[col].dropna().nunique() <= 1]
    if constant_cols:
        joined = joined.drop(columns=constant_cols)
        logger.info("Dropped {} constant-value columns: {}", len(constant_cols), constant_cols)
        drop_cols.extend(constant_cols)
    return joined, drop_cols


def _clean_and_impute(joined: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Sort, dedup, zero-fill, drop high-missing columns, and impute."""
    joined = joined.sort_index()
    joined = joined[~joined.index.duplicated(keep="first")]

    for col in ZERO_FILL_GEN_COLS:
        if col in joined.columns:
            n_filled = int(joined[col].isna().sum())
            if n_filled > 0:
                joined[col] = joined[col].fillna(0.0)
                logger.info("  Zero-filled {} NaN values in {}", n_filled, col)

    joined, drop_cols = _drop_low_quality_columns(joined)
    joined = impute_small_gaps(joined)

    post_quality = compute_data_quality(joined)
    logger.info(
        "Post-imputation: {} rows, {} columns, missing: {}",
        post_quality["total_rows"],
        len(joined.columns),
        {k: f"{v}%" for k, v in post_quality["missing_percent"].items() if v > 0},
    )
    return joined, drop_cols, post_quality


def _export_results(
    config: DataCollectionConfig,
    joined: pd.DataFrame,
    kaggle: bool,
    drop_cols: list[str],
    post_quality: dict[str, Any],
) -> Path:
    """Save parquet and optional Kaggle CSV + metadata."""
    output_path = config.processed_dir / f"dataset_{config.bidding_zone.lower()}.parquet"
    joined.to_parquet(output_path, engine="pyarrow")
    logger.info("Saved joined dataset -> {} ({} rows)", output_path, len(joined))

    if kaggle:
        csv_path = config.processed_dir / f"dataset_{config.bidding_zone.lower()}.csv"
        joined.to_csv(csv_path)
        logger.info("Saved Kaggle CSV -> {}", csv_path)

        meta_path = config.processed_dir / "dataset_metadata.json"
        metadata = build_kaggle_metadata(config, post_quality)
        metadata["dropped_columns"] = {
            "columns": drop_cols,
            "reason": f"More than {MAX_MISSING_PCT:.0f}% missing across the full time span",
        }
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info("Saved Kaggle metadata -> {}", meta_path)

    return output_path


def join_datasets(
    config: DataCollectionConfig,
    *,
    kaggle: bool = False,
) -> Path:
    """Join all data sources into one Parquet dataset.

    Set *kaggle=True* to also export CSV + metadata JSON.
    """
    config.ensure_dirs()

    prices, generation, weather = _load_mandatory_sources(config)

    logger.info("Joining datasets on UTC timestamps...")
    joined = prices.join(generation, how="outer").join(weather, how="outer")

    joined = _join_optional_sources(config, joined)
    joined, drop_cols, post_quality = _clean_and_impute(joined)

    return _export_results(config, joined, kaggle, drop_cols, post_quality)
