"""Data loading and feature engineering for the market simulation.

Loads the DE-LU hourly dataset, computes daily settlement prices, and
builds a feature matrix suitable for strategy consumption (with proper
lagging to prevent look-ahead bias).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

# Columns used for daily feature aggregation (mean).
_GENERATION_COLS = [
    "gen_solar_mw",
    "gen_wind_onshore_mw",
    "gen_wind_offshore_mw",
    "gen_fossil_gas_mw",
    "gen_fossil_hard_coal_mw",
    "gen_fossil_brown_coal_lignite_mw",
    "gen_nuclear_mw",
]

_LOAD_COLS = [
    "load_actual_mw",
    "load_forecast_mw",
]

_FORECAST_COLS = [
    "forecast_solar_mw",
    "forecast_wind_onshore_mw",
    "forecast_wind_offshore_mw",
]

_WEATHER_COLS = [
    "weather_temperature_2m_degc",
    "weather_wind_speed_10m_kmh",
    "weather_shortwave_radiation_wm2",
]

_NEIGHBOR_PRICE_COLS = [
    "price_fr_eur_mwh",
    "price_nl_eur_mwh",
    "price_at_eur_mwh",
    "price_pl_eur_mwh",
    "price_cz_eur_mwh",
    "price_dk_1_eur_mwh",
]

_FLOW_COLS = [
    "flow_fr_net_import_mw",
    "flow_nl_net_import_mw",
]

_COMMODITY_COLS = [
    "carbon_price_usd",
    "gas_price_usd",
]

_ALL_FEATURE_COLS = (
    _GENERATION_COLS
    + _LOAD_COLS
    + _FORECAST_COLS
    + _WEATHER_COLS
    + _NEIGHBOR_PRICE_COLS
    + _FLOW_COLS
    + _COMMODITY_COLS
)


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load the DE-LU hourly CSV dataset.

    Parameters
    ----------
    path:
        Path to ``dataset_de_lu.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"``
        and all original columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Dataset not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc").sort_index()
    return df


def compute_daily_settlement(df: pd.DataFrame) -> pd.Series:
    """Compute the daily settlement price from hourly DA prices.

    The settlement price for each calendar day is the arithmetic mean
    of the ``price_eur_mwh`` column over that day's 24 hours.

    Parameters
    ----------
    df:
        Hourly DataFrame as returned by :func:`load_dataset`.

    Returns
    -------
    pd.Series
        Daily settlement prices indexed by ``datetime.date``, named
        ``"settlement_price"``.
    """
    daily = df["price_eur_mwh"].groupby(df.index.date).mean()
    daily.name = "settlement_price"
    return daily


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a daily feature matrix from the hourly dataset.

    Aggregates hourly data to daily resolution and lags all features
    by one day to ensure no look-ahead bias: features for delivery
    day D are computed from data up to and including day D-1.

    Parameters
    ----------
    df:
        Hourly DataFrame as returned by :func:`load_dataset`.

    Returns
    -------
    pd.DataFrame
        Daily feature matrix indexed by ``datetime.date``.  Includes
        aggregated generation, load, weather, cross-border flows,
        and neighbouring zone prices.
    """
    # Select only columns that exist in the DataFrame
    available = [c for c in _ALL_FEATURE_COLS if c in df.columns]

    # Aggregate to daily means
    daily = df[available].groupby(df.index.date).mean()

    # Add price stats (mean, max, min, std) from the day
    daily["price_mean"] = df["price_eur_mwh"].groupby(df.index.date).mean()
    daily["price_max"] = df["price_eur_mwh"].groupby(df.index.date).max()
    daily["price_min"] = df["price_eur_mwh"].groupby(df.index.date).min()
    daily["price_std"] = df["price_eur_mwh"].groupby(df.index.date).std()

    # Rename columns to indicate aggregation
    rename = {c: f"{c}_mean" for c in available}
    daily = daily.rename(columns=rename)

    # Lag by one day: shift forward so that row for date D contains
    # features computed from date D-1's data.
    daily = daily.shift(1)

    # Drop the first row (NaN from the shift)
    daily = daily.dropna(how="all")

    # Ensure index contains date objects
    daily.index = [d if isinstance(d, date) else d.date() for d in daily.index]

    return daily
