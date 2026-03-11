"""Join price, generation, and weather data into a single dataset.

Performs an outer join on hourly UTC timestamps, applies simple imputation
for small gaps, and produces a consolidated Parquet file.  Optionally
exports a Kaggle-ready CSV with accompanying metadata JSON.

Column handling strategy for multi-year generation data:

- **Core generation columns** (present in all or most years) are kept.
  Where a fuel type was physically absent (e.g. ``nuclear`` after April 2023
  shutdown), NaN is filled with 0.0 MW — the plant genuinely produced nothing.
- **Consumption columns** are sporadic side-reports that come and go across
  ENTSO-E reporting revisions.  ``hydro_pumped_storage_consumption`` is the
  only consistently reported one (important for modelling storage dispatch).
  Other ``*_consumption`` columns with >50 % missing are dropped.
- A configurable ``max_missing_pct`` threshold controls which columns survive.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig

# Maximum gap (in hours) that will be forward-filled. Larger gaps stay NaN.
MAX_FFILL_HOURS = 3

# Columns with more than this fraction missing get dropped before export.
MAX_MISSING_PCT = 50.0

# Generation columns where NaN means "zero output" (plant shut down / fuel
# type not reported that year) rather than truly missing data.  These get
# filled with 0.0 before the missing-% threshold is applied.
ZERO_FILL_GEN_COLS = {
    "gen_nuclear",
    "gen_fossil_coal_derived_gas",
}


def load_raw_parquet(path: Path, name: str) -> pd.DataFrame:
    """Load a consolidated Parquet file, with a clear error if missing.

    Parameters
    ----------
    path:
        Path to the Parquet file.
    name:
        Human-readable name for error messages (e.g. ``"prices"``).

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        msg = f"Raw {name} file not found: {path}. Run the download step first."
        raise FileNotFoundError(msg)
    df = pd.read_parquet(path)
    logger.info("Loaded {} — {} rows, {} columns", name, len(df), len(df.columns))
    return df


def impute_small_gaps(df: pd.DataFrame, max_gap_hours: int = MAX_FFILL_HOURS) -> pd.DataFrame:
    """Forward-fill gaps of up to ``max_gap_hours`` consecutive NaN rows.

    Larger gaps are left as NaN so downstream consumers can decide how to
    handle them.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex at hourly frequency.
    max_gap_hours:
        Maximum number of consecutive NaN values to fill per column.

    Returns
    -------
    pd.DataFrame
        DataFrame with small gaps filled.
    """
    return df.ffill(limit=max_gap_hours)


def compute_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Compute data quality statistics for the joined dataset.

    Returns
    -------
    dict
        Dictionary with per-column missing percentages, row count, and
        date range.
    """
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


def build_kaggle_metadata(
    config: DataCollectionConfig,
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Build a metadata dict suitable for Kaggle dataset description.

    Parameters
    ----------
    config:
        Pipeline configuration (for zone, coordinates, etc.).
    quality:
        Output of :func:`compute_data_quality`.

    Returns
    -------
    dict
        Metadata JSON-serialisable dictionary.
    """
    return {
        "title": (
            f"DE-LU Day-Ahead Electricity Prices, Generation Mix & Weather ({config.bidding_zone})"
        ),
        "description": (
            "Hourly dataset joining ENTSO-E day-ahead prices, actual generation "
            "per type, and ERA5 reanalysis weather data for the DE-LU bidding zone. "
            "All timestamps are UTC."
        ),
        "sources": {
            "prices": "ENTSO-E Transparency Platform (Day Ahead Prices, A.44)",
            "generation": "ENTSO-E Transparency Platform (Actual Generation per Type, A.75)",
            "weather": (
                f"Open-Meteo Archive API (ERA5 reanalysis) at"
                f" ({config.weather_latitude}, {config.weather_longitude})"
            ),
        },
        "bidding_zone": config.bidding_zone,
        "timezone": "UTC",
        "weather_coordinates": {
            "latitude": config.weather_latitude,
            "longitude": config.weather_longitude,
        },
        "years": config.years,
        "quality": quality,
        "license": "CC-BY-4.0 (ENTSO-E data), CC-BY-4.0 (Copernicus/ERA5 via Open-Meteo)",
    }


def join_datasets(
    config: DataCollectionConfig,
    *,
    kaggle: bool = False,
) -> Path:
    """Join prices, generation, and weather into one dataset.

    Parameters
    ----------
    config:
        Pipeline configuration.
    kaggle:
        If *True*, also export a CSV and ``dataset_metadata.json`` for
        Kaggle upload.

    Returns
    -------
    Path
        Path to the final joined Parquet file.
    """
    config.ensure_dirs()

    # Load raw consolidated files
    prices = load_raw_parquet(config.raw_dir / "prices_da.parquet", "prices")
    generation = load_raw_parquet(config.raw_dir / "generation.parquet", "generation")
    weather = load_raw_parquet(config.raw_dir / "weather.parquet", "weather")

    # Prefix generation columns to avoid name collisions
    generation = generation.add_prefix("gen_")

    # Prefix weather columns
    weather = weather.add_prefix("weather_")

    # Outer join on UTC timestamps
    logger.info("Joining datasets on UTC timestamps...")
    joined = prices.join(generation, how="outer").join(weather, how="outer")

    # Sort and deduplicate
    joined = joined.sort_index()
    joined = joined[~joined.index.duplicated(keep="first")]

    # --- Zero-fill known generation columns ---
    # For fuels that were physically absent in some years (e.g. nuclear after
    # the German phase-out), NaN means 0 MW — not missing data.
    for col in ZERO_FILL_GEN_COLS:
        if col in joined.columns:
            n_filled = int(joined[col].isna().sum())
            if n_filled > 0:
                joined[col] = joined[col].fillna(0.0)
                logger.info("  Zero-filled {} NaN values in {}", n_filled, col)

    # Log pre-imputation quality
    pre_quality = compute_data_quality(joined)
    logger.info(
        "Pre-imputation: {} rows, missing: {}",
        pre_quality["total_rows"],
        {k: f"{v}%" for k, v in pre_quality["missing_percent"].items() if v > 0},
    )

    # --- Drop columns with too much missing data ---
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

    # --- Drop constant-value columns (no information) ---
    constant_cols = [col for col in joined.columns if joined[col].dropna().nunique() <= 1]
    if constant_cols:
        joined = joined.drop(columns=constant_cols)
        logger.info("Dropped {} constant-value columns: {}", len(constant_cols), constant_cols)
        drop_cols.extend(constant_cols)

    # Impute small gaps
    joined = impute_small_gaps(joined)

    # Log post-imputation quality
    post_quality = compute_data_quality(joined)
    logger.info(
        "Post-imputation: {} rows, {} columns, missing: {}",
        post_quality["total_rows"],
        len(joined.columns),
        {k: f"{v}%" for k, v in post_quality["missing_percent"].items() if v > 0},
    )

    # Save Parquet
    output_path = config.processed_dir / f"dataset_{config.bidding_zone.lower()}.parquet"
    joined.to_parquet(output_path, engine="pyarrow")
    logger.info("Saved joined dataset -> {} ({} rows)", output_path, len(joined))

    # Kaggle export
    if kaggle:
        csv_path = config.processed_dir / f"dataset_{config.bidding_zone.lower()}.csv"
        joined.to_csv(csv_path)
        logger.info("Saved Kaggle CSV -> {}", csv_path)

        meta_path = config.processed_dir / "dataset_metadata.json"
        metadata = build_kaggle_metadata(config, post_quality)
        # Record dropped columns for transparency
        metadata["dropped_columns"] = {
            "columns": drop_cols,
            "reason": f"More than {MAX_MISSING_PCT:.0f}% missing across the full time span",
        }
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info("Saved Kaggle metadata -> {}", meta_path)

    return output_path
