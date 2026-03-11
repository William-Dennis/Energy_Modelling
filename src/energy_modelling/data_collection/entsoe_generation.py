"""Download actual generation per type from the ENTSO-E Transparency Platform.

Uses the ``entsoe-py`` library to query generation mix data for a given
bidding zone and date range.  Data is saved as one Parquet file per year
under the raw data directory and then consolidated.
"""

from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig


def _year_range(year: int, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) timestamps for a calendar year."""
    start = pd.Timestamp(f"{year}-01-01", tz=timezone)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=timezone)
    return start, end


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise generation DataFrame columns.

    The ``entsoe-py`` library returns a MultiIndex of
    ``(generation_type, "Actual Aggregated")`` or similar.  This function
    flattens the columns to clean snake_case names like ``wind_onshore``,
    ``solar``, ``gas``, etc.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Take only the first level (generation type name)
        df.columns = [col[0] for col in df.columns]

    # Normalise names: lower-case, replace spaces/hyphens with underscores
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")
        for c in df.columns
    ]
    return df


def fetch_generation_for_year(
    client: EntsoePandasClient,
    year: int,
    bidding_zone: str,
    timezone: str,
) -> pd.DataFrame:
    """Fetch actual generation per type for a single year.

    Parameters
    ----------
    client:
        An authenticated ``EntsoePandasClient``.
    year:
        Calendar year to fetch.
    bidding_zone:
        ENTSO-E bidding zone key (e.g. ``"DE_LU"``).
    timezone:
        IANA timezone string used to define year boundaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"``
        and one column per generation type (in MW).
    """
    start, end = _year_range(year, timezone)
    logger.info("Fetching generation mix for {} year={}", bidding_zone, year)

    df: pd.DataFrame = client.query_generation(bidding_zone, start=start, end=end)

    # Clean up column names
    df = _clean_columns(df)

    # Normalise to UTC
    dt_index = pd.DatetimeIndex(df.index)
    df.index = dt_index.tz_convert("UTC")
    df.index.name = "timestamp_utc"

    # Resample to hourly — ENTSO-E sometimes returns 15-min data
    df = df.resample("1h").mean()

    logger.info("  -> {} rows, {} columns for year {}", len(df), len(df.columns), year)
    return df


def download_generation(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download generation mix for all configured years and save to Parquet.

    Parameters
    ----------
    config:
        Pipeline configuration.
    force:
        If *True*, re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the consolidated Parquet file.
    """
    config.ensure_dirs()
    consolidated_path = config.raw_dir / "generation.parquet"

    client = EntsoePandasClient(api_key=config.entsoe_api_key)
    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"generation_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        df = fetch_generation_for_year(client, year, config.bidding_zone, config.timezone)
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    # Consolidate — align columns across years (some fuel types may appear/disappear)
    all_gen = pd.concat(frames).sort_index()
    all_gen = all_gen[~all_gen.index.duplicated(keep="first")]
    all_gen.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated generation -> {} ({} rows)", consolidated_path, len(all_gen))
    return consolidated_path
