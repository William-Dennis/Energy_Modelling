"""Download day-ahead prices for neighbouring bidding zones from ENTSO-E.

Fetches day-ahead prices for each configured neighbour zone, producing
one column per zone.  Data is saved as one Parquet file per year under the
raw data directory and then consolidated.
"""

from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.utils import year_range as _year_range


def fetch_neighbour_prices_for_year(
    client: EntsoePandasClient,
    year: int,
    timezone: str,
    neighbour_zones: list[str],
) -> pd.DataFrame:
    """Fetch day-ahead prices for all neighbour zones for a single year.

    Parameters
    ----------
    client:
        An authenticated ``EntsoePandasClient``.
    year:
        Calendar year to fetch.
    timezone:
        IANA timezone string used to define year boundaries.
    neighbour_zones:
        List of ENTSO-E bidding zone keys to query.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"`` and
        one column per zone (e.g. ``"price_fr_eur_mwh"``).
    """
    start, end = _year_range(year, timezone)
    frames: list[pd.DataFrame] = []

    for zone in neighbour_zones:
        col_name = f"price_{zone.lower()}_eur_mwh"
        try:
            logger.info("  Fetching DA price for {} year={}", zone, year)
            series: pd.Series = client.query_day_ahead_prices(zone, start=start, end=end)
            zone_df = series.to_frame(name=col_name)
            dt_index = pd.DatetimeIndex(zone_df.index)
            zone_df.index = dt_index.tz_convert("UTC")
            zone_df.index.name = "timestamp_utc"
            zone_df = zone_df.resample("1h").mean()
            frames.append(zone_df)
        except Exception:
            logger.warning("  Failed to fetch prices for {} year={} — skipping", zone, year)

    if not frames:
        logger.warning("No neighbour prices retrieved for year={}", year)
        return pd.DataFrame()

    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer")

    logger.info("  -> {} rows, {} zones for year {}", len(df), len(frames), year)
    return df


def download_neighbour_prices(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download neighbour zone DA prices for all configured years.

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
    consolidated_path = config.raw_dir / "neighbour_prices.parquet"

    client = EntsoePandasClient(api_key=config.entsoe_api_key)
    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"neighbour_prices_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        logger.info("Fetching neighbour prices year={}", year)
        df = fetch_neighbour_prices_for_year(client, year, config.timezone, config.neighbour_zones)
        if df.empty:
            continue
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    if not frames:
        # Write empty file so downstream doesn't crash
        pd.DataFrame().to_parquet(consolidated_path, engine="pyarrow")
        logger.warning("No neighbour price data retrieved")
        return consolidated_path

    # Consolidate
    all_nb = pd.concat(frames).sort_index()
    all_nb = all_nb[~all_nb.index.duplicated(keep="first")]
    all_nb.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated neighbour prices -> {} ({} rows)", consolidated_path, len(all_nb))
    return consolidated_path
