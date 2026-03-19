"""Download actual and forecast total load from the ENTSO-E Transparency Platform.

Uses the ``entsoe-py`` library to query total load (actual) and total load
day-ahead forecast for a given bidding zone.  Data is saved as one Parquet
file per year under the raw data directory and then consolidated.
"""

from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.utils import year_range as _year_range


def fetch_load_for_year(
    client: EntsoePandasClient,
    year: int,
    bidding_zone: str,
    timezone: str,
) -> pd.DataFrame:
    """Fetch actual load and day-ahead load forecast for a single year.

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
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"`` and
        columns ``"load_actual_mw"`` and ``"load_forecast_mw"``.
    """
    start, end = _year_range(year, timezone)
    logger.info("Fetching total load for {} year={}", bidding_zone, year)

    # Actual load
    actual: pd.Series = client.query_load(bidding_zone, start=start, end=end)
    if isinstance(actual, pd.DataFrame):
        # Some zones return DataFrame with multiple columns; take the first
        actual = actual.iloc[:, 0]
    actual_df = actual.to_frame(name="load_actual_mw")

    # Forecast load
    forecast: pd.Series = client.query_load_forecast(bidding_zone, start=start, end=end)
    if isinstance(forecast, pd.DataFrame):
        forecast = forecast.iloc[:, 0]
    forecast_df = forecast.to_frame(name="load_forecast_mw")

    # Join actual and forecast
    df = actual_df.join(forecast_df, how="outer")

    # Normalise to UTC
    dt_index = pd.DatetimeIndex(df.index)
    df.index = dt_index.tz_convert("UTC")
    df.index.name = "timestamp_utc"

    # Resample to hourly
    df = df.resample("1h").mean()

    logger.info("  -> {} rows for year {}", len(df), year)
    return df


def download_load(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download total load for all configured years and save to Parquet.

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
    consolidated_path = config.raw_dir / "load.parquet"

    client = EntsoePandasClient(api_key=config.entsoe_api_key)
    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"load_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        df = fetch_load_for_year(client, year, config.bidding_zone, config.timezone)
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    # Consolidate
    all_load = pd.concat(frames).sort_index()
    all_load = all_load[~all_load.index.duplicated(keep="first")]
    all_load.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated load -> {} ({} rows)", consolidated_path, len(all_load))
    return consolidated_path
