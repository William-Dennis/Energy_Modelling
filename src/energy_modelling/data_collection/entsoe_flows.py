"""Download cross-border physical electricity flows from ENTSO-E.

Queries physical flows in both directions for each neighbour border and
computes net flows.  Data is saved as one Parquet file per year under the
raw data directory and then consolidated.
"""

from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig


def _year_range(year: int, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) timestamps for a calendar year in the given timezone."""
    start = pd.Timestamp(f"{year}-01-01", tz=timezone)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=timezone)
    return start, end


def _fetch_one_direction(
    client: EntsoePandasClient,
    from_zone: str,
    to_zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Fetch physical flow for a single direction, returning None on failure."""
    try:
        series: pd.Series = client.query_crossborder_flows(from_zone, to_zone, start=start, end=end)
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series
    except Exception:
        logger.warning("  Failed to fetch flow {} -> {}", from_zone, to_zone)
        return None


def fetch_flows_for_year(
    client: EntsoePandasClient,
    year: int,
    bidding_zone: str,
    timezone: str,
    neighbour_zones: list[str],
) -> pd.DataFrame:
    """Fetch cross-border physical flows for all neighbours for a single year.

    For each neighbour N, produces three columns:
    - ``flow_{n}_export_mw``: DE_LU -> N (export from DE_LU perspective)
    - ``flow_{n}_import_mw``: N -> DE_LU (import to DE_LU)
    - ``flow_{n}_net_import_mw``: import - export (positive = net import)

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
        logger.info("  Fetching flows {} <-> {} year={}", bidding_zone, zone, year)

        # Export: DE_LU -> neighbour
        export_series = _fetch_one_direction(client, bidding_zone, zone, start, end)
        # Import: neighbour -> DE_LU
        import_series = _fetch_one_direction(client, zone, bidding_zone, start, end)

        if export_series is not None:
            all_cols[f"flow_{zone_lower}_export_mw"] = export_series
        if import_series is not None:
            all_cols[f"flow_{zone_lower}_import_mw"] = import_series

    if not all_cols:
        logger.warning("No flow data retrieved for year={}", year)
        return pd.DataFrame()

    df = pd.DataFrame(all_cols)

    # Normalise to UTC
    dt_index = pd.DatetimeIndex(df.index)
    df.index = dt_index.tz_convert("UTC")
    df.index.name = "timestamp_utc"

    # Resample to hourly
    df = df.resample("1h").mean()

    # Compute net import for borders where both directions are available
    for zone in neighbour_zones:
        zone_lower = zone.lower()
        imp_col = f"flow_{zone_lower}_import_mw"
        exp_col = f"flow_{zone_lower}_export_mw"
        net_col = f"flow_{zone_lower}_net_import_mw"
        if imp_col in df.columns and exp_col in df.columns:
            df[net_col] = df[imp_col] - df[exp_col]

    logger.info("  -> {} rows, {} columns for year {}", len(df), len(df.columns), year)
    return df


def download_flows(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download cross-border flows for all configured years.

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
    consolidated_path = config.raw_dir / "flows.parquet"

    client = EntsoePandasClient(api_key=config.entsoe_api_key)
    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"flows_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        logger.info("Fetching cross-border flows year={}", year)
        df = fetch_flows_for_year(
            client, year, config.bidding_zone, config.timezone, config.neighbour_zones
        )
        if df.empty:
            continue
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    if not frames:
        pd.DataFrame().to_parquet(consolidated_path, engine="pyarrow")
        logger.warning("No cross-border flow data retrieved")
        return consolidated_path

    # Consolidate
    all_flows = pd.concat(frames).sort_index()
    all_flows = all_flows[~all_flows.index.duplicated(keep="first")]
    all_flows.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated flows -> {} ({} rows)", consolidated_path, len(all_flows))
    return consolidated_path
