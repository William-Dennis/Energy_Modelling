"""Download TTF natural gas price proxy from Yahoo Finance.

Uses the ``TTF=F`` ticker (ICE Dutch TTF Natural Gas Futures) on Yahoo
Finance.  If ``TTF=F`` is unavailable, falls back to ``NG=F`` (Henry Hub
Natural Gas Futures, USD/MMBtu) which is less relevant for European markets
but provides a gas price proxy.

Daily data is forward-filled to hourly to match the dataset granularity.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig

# Ordered list of tickers to try — first success wins
GAS_TICKERS = ["TTF=F", "NG=F"]


def fetch_gas_price(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily gas price proxy from Yahoo Finance.

    Tries ``TTF=F`` first, then falls back to ``NG=F``.

    Parameters
    ----------
    start_date:
        Start date as ``"YYYY-MM-DD"`` string.
    end_date:
        End date as ``"YYYY-MM-DD"`` string.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"``
        and columns ``"gas_price_usd"`` and ``"gas_ticker"`` (the source
        ticker used).  Resampled to hourly with forward-fill.
    """
    for ticker_name in GAS_TICKERS:
        logger.info("Trying gas ticker {} for {} to {}", ticker_name, start_date, end_date)
        try:
            ticker = yf.Ticker(ticker_name)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            if not hist.empty and len(hist) > 5:
                logger.info("  Using {} — {} daily rows", ticker_name, len(hist))

                df = hist[["Close"]].rename(columns={"Close": "gas_price_usd"})

                # Normalise to UTC
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
                df.index.name = "timestamp_utc"

                # Resample to hourly with forward-fill
                df = df.resample("1h").ffill()

                return df
        except Exception:
            logger.warning("  Ticker {} failed", ticker_name)

    logger.warning("No gas price data available for {} to {}", start_date, end_date)
    return pd.DataFrame(columns=pd.Index(["gas_price_usd"]))


def download_gas_price(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download gas price for all configured years and save to Parquet.

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
    consolidated_path = config.raw_dir / "gas_price.parquet"

    frames: list[pd.DataFrame] = []

    for year in sorted(config.years):
        year_path = config.raw_dir / f"gas_price_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"

        df = fetch_gas_price(start_date, end_date)
        if df.empty:
            logger.warning("No gas data for year={} — skipping", year)
            continue
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    if not frames:
        pd.DataFrame(columns=pd.Index(["gas_price_usd"])).to_parquet(
            consolidated_path, engine="pyarrow"
        )
        logger.warning("No gas price data retrieved")
        return consolidated_path

    # Consolidate
    all_gas = pd.concat(frames).sort_index()
    all_gas = all_gas[~all_gas.index.duplicated(keep="first")]
    all_gas.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated gas price -> {} ({} rows)", consolidated_path, len(all_gas))
    return consolidated_path
