"""Download day-ahead wind and solar generation forecasts from ENTSO-E.

Uses the ``entsoe-py`` library to query wind and solar generation forecasts
for a given bidding zone.  Data is saved as one Parquet file per year under
the raw data directory and then consolidated.
"""

from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.utils import (
    normalise_name as _normalise_name,
)
from energy_modelling.data_collection.utils import (
    year_range as _year_range,
)


def _clean_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise forecast DataFrame columns.

    The ``entsoe-py`` library may return either a MultiIndex or flat columns
    depending on the query.  This function normalises to flat snake_case
    column names prefixed with ``forecast_``.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols: list[str] = []
        for parts in df.columns:
            name = "_".join(_normalise_name(str(p)) for p in parts if str(p).strip())
            new_cols.append(name)
        df.columns = pd.Index(new_cols)
    else:
        df.columns = pd.Index([_normalise_name(str(c)) for c in df.columns])
    return df


def fetch_forecasts_for_year(
    client: EntsoePandasClient,
    year: int,
    bidding_zone: str,
    timezone: str,
) -> pd.DataFrame:
    """Fetch day-ahead wind and solar generation forecasts for a single year.

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
        and columns for each forecast type (e.g. ``"solar"``,
        ``"wind_onshore"``, ``"wind_offshore"``).
    """
    start, end = _year_range(year, timezone)
    logger.info("Fetching wind/solar forecasts for {} year={}", bidding_zone, year)

    df: pd.DataFrame = client.query_wind_and_solar_forecast(
        bidding_zone, start=start, end=end, psr_type=None
    )

    # Clean up column names
    df = _clean_forecast_columns(df)

    # Normalise to UTC
    dt_index = pd.DatetimeIndex(df.index)
    df.index = dt_index.tz_convert("UTC")
    df.index.name = "timestamp_utc"

    # Resample to hourly
    df = df.resample("1h").mean()

    logger.info("  -> {} rows, {} columns for year {}", len(df), len(df.columns), year)
    return df


def download_forecasts(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download wind/solar forecasts for all configured years and save to Parquet.

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
    consolidated_path = config.raw_dir / "forecasts.parquet"

    client = EntsoePandasClient(api_key=config.entsoe_api_key)
    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"forecasts_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        df = fetch_forecasts_for_year(client, year, config.bidding_zone, config.timezone)
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    # Consolidate — columns may vary across years
    all_fc = pd.concat(frames).sort_index()
    all_fc = all_fc[~all_fc.index.duplicated(keep="first")]
    all_fc.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated forecasts -> {} ({} rows)", consolidated_path, len(all_fc))
    return consolidated_path
