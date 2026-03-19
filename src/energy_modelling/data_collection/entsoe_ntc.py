"""Download day-ahead net transfer capacities (NTC) from ENTSO-E.

Queries NTC in both directions for each neighbour border.  Data is saved
as one Parquet file per year under the raw data directory and then
consolidated.
"""

from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.utils import year_range as _year_range


def _fetch_ntc_one_direction(
    client: EntsoePandasClient,
    from_zone: str,
    to_zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Fetch NTC for a single direction, returning None on failure."""
    try:
        series: pd.Series = client.query_net_transfer_capacity_dayahead(
            from_zone, to_zone, start=start, end=end
        )
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series
    except Exception:
        logger.warning("  Failed to fetch NTC {} -> {}", from_zone, to_zone)
        return None


def fetch_ntc_for_year(
    client: EntsoePandasClient,
    year: int,
    bidding_zone: str,
    timezone: str,
    neighbour_zones: list[str],
) -> pd.DataFrame:
    """Fetch day-ahead NTC for all neighbours for a single year.

    For each neighbour N, produces two columns:
    - ``ntc_{n}_export_mw``: capacity DE_LU -> N
    - ``ntc_{n}_import_mw``: capacity N -> DE_LU

    Parameters
    ----------
    client:
        An authenticated ``EntsoePandasClient``.
    year:
        Calendar year to fetch.
    bidding_zone:
        Home bidding zone (e.g. ``"DE_LU"``).
    timezone:
        IANA timezone string used to define year boundaries.
    neighbour_zones:
        List of neighbour bidding zone keys.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"``.
    """
    start, end = _year_range(year, timezone)
    all_cols: dict[str, pd.Series] = {}

    for zone in neighbour_zones:
        zone_lower = zone.lower()
        logger.info("  Fetching NTC {} <-> {} year={}", bidding_zone, zone, year)

        # Export capacity: DE_LU -> neighbour
        export_series = _fetch_ntc_one_direction(client, bidding_zone, zone, start, end)
        # Import capacity: neighbour -> DE_LU
        import_series = _fetch_ntc_one_direction(client, zone, bidding_zone, start, end)

        if export_series is not None:
            all_cols[f"ntc_{zone_lower}_export_mw"] = export_series
        if import_series is not None:
            all_cols[f"ntc_{zone_lower}_import_mw"] = import_series

    if not all_cols:
        logger.warning("No NTC data retrieved for year={}", year)
        return pd.DataFrame()

    df = pd.DataFrame(all_cols)

    # Normalise to UTC
    dt_index = pd.DatetimeIndex(df.index)
    df.index = dt_index.tz_convert("UTC")
    df.index.name = "timestamp_utc"

    # Resample to hourly
    df = df.resample("1h").mean()

    logger.info("  -> {} rows, {} columns for year {}", len(df), len(df.columns), year)
    return df


def download_ntc(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download NTC for all configured years.

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
    consolidated_path = config.raw_dir / "ntc.parquet"

    client = EntsoePandasClient(api_key=config.entsoe_api_key)
    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"ntc_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        logger.info("Fetching NTC year={}", year)
        df = fetch_ntc_for_year(
            client, year, config.bidding_zone, config.timezone, config.neighbour_zones
        )
        if df.empty:
            continue
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    if not frames:
        pd.DataFrame().to_parquet(consolidated_path, engine="pyarrow")
        logger.warning("No NTC data retrieved")
        return consolidated_path

    # Consolidate
    all_ntc = pd.concat(frames).sort_index()
    all_ntc = all_ntc[~all_ntc.index.duplicated(keep="first")]
    all_ntc.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated NTC -> {} ({} rows)", consolidated_path, len(all_ntc))
    return consolidated_path
