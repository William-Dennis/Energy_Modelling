"""Data loading and feature engineering for the market simulation.

Loads the DE-LU hourly dataset, computes daily settlement prices, and
builds a feature matrix suitable for strategy consumption.

Feature timing follows a mixed-information contract:

- realised or observed series are lagged by one day
- day-ahead forecast series remain aligned to the delivery day

This matches a day-ahead trading setting where forecasts for delivery day D
are available before the auction closes, while realised values for D are not.
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

_REALISED_LOAD_COLS = ["load_actual_mw"]

_FORECAST_LOAD_COLS = ["load_forecast_mw"]

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

_REALISED_FEATURE_COLS = (
    _GENERATION_COLS
    + _REALISED_LOAD_COLS
    + _WEATHER_COLS
    + _NEIGHBOR_PRICE_COLS
    + _FLOW_COLS
    + _COMMODITY_COLS
)

_FORECAST_FEATURE_COLS = _FORECAST_LOAD_COLS + _FORECAST_COLS

_ALL_FEATURE_COLS = _REALISED_FEATURE_COLS + _FORECAST_FEATURE_COLS

_PRICE_STAT_COLS = ["price_mean", "price_max", "price_min", "price_std"]


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

    Aggregates hourly data to daily resolution using two timing groups:

    - realised features are shifted forward by one day, so row ``D`` only
      contains realised information from ``D-1`` or earlier
    - day-ahead forecast features remain on row ``D``, because those values
      are assumed to be known before trading delivery day ``D``

    Parameters
    ----------
    df:
        Hourly DataFrame as returned by :func:`load_dataset`.

    Returns
    -------
    pd.DataFrame
        Daily feature matrix indexed by ``datetime.date``.
    """

    def _aggregate_mean(columns: list[str]) -> pd.DataFrame:
        available = [column for column in columns if column in df.columns]
        if not available:
            return pd.DataFrame(index=pd.Index(sorted(set(df.index.date))))
        daily = df[available].groupby(df.index.date).mean()
        return daily.rename(columns={column: f"{column}_mean" for column in available})

    realised_daily = _aggregate_mean(list(_REALISED_FEATURE_COLS))

    if "price_eur_mwh" in df.columns:
        price_group = df["price_eur_mwh"].groupby(df.index.date)
        realised_daily["price_mean"] = price_group.mean()
        realised_daily["price_max"] = price_group.max()
        realised_daily["price_min"] = price_group.min()
        realised_daily["price_std"] = price_group.std()

    forecast_daily = _aggregate_mean(list(_FORECAST_FEATURE_COLS))

    realised_daily = realised_daily.shift(1)
    daily = realised_daily.join(forecast_daily, how="outer")
    daily = daily.dropna(how="all")

    # Ensure index contains date objects
    daily.index = [d if isinstance(d, date) else d.date() for d in daily.index]

    return daily
