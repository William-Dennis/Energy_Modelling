"""Download EU ETS carbon price proxy from Yahoo Finance.

Uses the WisdomTree Carbon ETC (``CARB.L``) as a proxy for EU ETS
(European Union Allowance) carbon prices.  ``CARB.L`` is USD-denominated,
traded on the London Stock Exchange, and tracks EUA futures with
>0.99 correlation.  Daily data is forward-filled to hourly to match the
dataset granularity.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig

CARBON_TICKER = "CARB.L"


def fetch_carbon_price(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily carbon price proxy from Yahoo Finance.

    Parameters
    ----------
    start_date:
        Start date as ``"YYYY-MM-DD"`` string.
    end_date:
        End date as ``"YYYY-MM-DD"`` string (exclusive for yfinance).

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"``
        and a single column ``"carbon_price_usd"``.  Resampled to hourly
        with forward-fill.
    """
    logger.info("Fetching carbon price ({}) {} to {}", CARBON_TICKER, start_date, end_date)

    ticker = yf.Ticker(CARBON_TICKER)
    hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)

    if hist.empty:
        logger.warning("No carbon price data returned for {} to {}", start_date, end_date)
        return pd.DataFrame(columns=pd.Index(["carbon_price_usd"]))

    # Use Close price
    df = hist[["Close"]].rename(columns={"Close": "carbon_price_usd"})

    # Normalise to UTC (yfinance returns tz-aware or tz-naive depending on ticker)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df.index.name = "timestamp_utc"

    # Resample to hourly and forward-fill (market data is daily)
    df = df.resample("1h").ffill()

    logger.info("  -> {} rows", len(df))
    return df


def download_carbon_price(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download carbon price for all configured years and save to Parquet.

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
    consolidated_path = config.raw_dir / "carbon_price.parquet"

    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"carbon_price_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"

        df = fetch_carbon_price(start_date, end_date)
        if df.empty:
            logger.warning("No carbon data for year={} — skipping", year)
            continue
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    if not frames:
        pd.DataFrame(columns=pd.Index(["carbon_price_usd"])).to_parquet(
            consolidated_path, engine="pyarrow"
        )
        logger.warning("No carbon price data retrieved")
        return consolidated_path

    # Consolidate
    all_carbon = pd.concat(frames).sort_index()
    all_carbon = all_carbon[~all_carbon.index.duplicated(keep="first")]
    all_carbon.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated carbon price -> {} ({} rows)", consolidated_path, len(all_carbon))
    return consolidated_path
