"""Download day-ahead electricity prices from the ENTSO-E Transparency Platform.

Uses the ``entsoe-py`` library to query day-ahead prices for a given bidding
zone and date range.  Data is saved as one Parquet file per year under the raw
data directory and then consolidated into a single file.
"""

from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.utils import year_range as _year_range


def fetch_prices_for_year(
    client: EntsoePandasClient,
    year: int,
    bidding_zone: str,
    timezone: str,
) -> pd.DataFrame:
    """Fetch day-ahead prices for a single year and return as a DataFrame.

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
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"`` and a
        single column ``"price_eur_mwh"``.
    """
    start, end = _year_range(year, timezone)
    logger.info("Fetching DA prices for {} year={}", bidding_zone, year)

    series: pd.Series = client.query_day_ahead_prices(bidding_zone, start=start, end=end)

    # Normalise to UTC and convert to DataFrame
    df = series.to_frame(name="price_eur_mwh")
    dt_index = pd.DatetimeIndex(df.index)
    df.index = dt_index.tz_convert("UTC")
    df.index.name = "timestamp_utc"

    # Resample to hourly — some zones may return 15-min or 30-min data
    df = df.resample("1h").mean()

    logger.info("  -> {} rows for year {}", len(df), year)
    return df


def download_prices(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download DA prices for all configured years and save to Parquet.

    For each year a raw file ``prices_da_{year}.parquet`` is saved.  A
    consolidated file ``prices_da.parquet`` covering all years is then written
    to the raw directory.

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
    consolidated_path = config.raw_dir / "prices_da.parquet"

    client = EntsoePandasClient(api_key=config.entsoe_api_key)
    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"prices_da_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        df = fetch_prices_for_year(client, year, config.bidding_zone, config.timezone)
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    # Consolidate
    all_prices = pd.concat(frames).sort_index()
    all_prices = all_prices[~all_prices.index.duplicated(keep="first")]
    all_prices.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated prices -> {} ({} rows)", consolidated_path, len(all_prices))
    return consolidated_path
