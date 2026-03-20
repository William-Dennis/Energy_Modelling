"""Pure EDA computation functions (no Streamlit dependency).

Independently testable analyses on pandas/numpy data grouped by priority:
P1-Price changes, P2-Autocorrelation, P3-Forecast errors,
P4-Lagged correlations, P5-Volatility, P6-Residual load.
Advanced analyses live in ``eda_analysis_advanced`` and are re-exported here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Re-export advanced analyses so callers can import everything from this module.
from energy_modelling.dashboard.eda_analysis_advanced import (  # noqa: F401
    day_of_week_edge_by_year,
    direction_by_group,
    feature_drift,
    quarterly_direction_rates,
    volatility_regime_performance,
    wind_quintile_analysis,
)

# Re-export clean_hourly_data from its canonical location so that existing
# imports (dashboard, tests) continue to work unchanged.
from energy_modelling.futures_market.data import clean_hourly_data  # noqa: F401

# ---------------------------------------------------------------------------
# P1: Price Change Distribution
# ---------------------------------------------------------------------------


def compute_daily_settlement(hourly_prices: pd.Series) -> pd.Series:
    """Aggregate hourly prices to daily mean settlement prices.

    Parameters
    ----------
    hourly_prices:
        Hourly price series with DatetimeIndex.

    Returns
    -------
    Daily mean settlement prices indexed by date.
    """
    daily = hourly_prices.groupby(hourly_prices.index.date).mean()
    daily.index = pd.DatetimeIndex(daily.index)
    daily.name = "settlement_price"
    return daily


def compute_price_changes(daily_settlements: pd.Series) -> pd.Series:
    """Compute day-over-day price changes (the trading signal).

    Parameters
    ----------
    daily_settlements:
        Daily settlement price series.

    Returns
    -------
    Daily price changes (NaN for first day is dropped).
    """
    changes = daily_settlements.diff().dropna()
    changes.name = "price_change"
    return changes


def direction_base_rates(price_changes: pd.Series) -> dict[str, Any]:
    """Compute base rates of price direction (up/down/zero).

    Parameters
    ----------
    price_changes:
        Daily price change series.

    Returns
    -------
    Dictionary with counts, percentages, and mean move sizes.
    """
    n_total = len(price_changes)
    n_up = int((price_changes > 0).sum())
    n_down = int((price_changes < 0).sum())
    n_zero = int((price_changes == 0).sum())

    up_moves = price_changes[price_changes > 0]
    down_moves = price_changes[price_changes < 0]

    return {
        "n_total": n_total,
        "n_up": n_up,
        "n_down": n_down,
        "n_zero": n_zero,
        "pct_up": n_up / n_total * 100 if n_total > 0 else 0.0,
        "pct_down": n_down / n_total * 100 if n_total > 0 else 0.0,
        "pct_zero": n_zero / n_total * 100 if n_total > 0 else 0.0,
        "mean_up_move": float(up_moves.mean()) if len(up_moves) > 0 else 0.0,
        "mean_down_move": float(down_moves.abs().mean()) if len(down_moves) > 0 else 0.0,
        "median_change": float(price_changes.median()),
        "skewness": float(price_changes.skew()),
        "kurtosis": float(price_changes.kurtosis()),
    }


# ---------------------------------------------------------------------------
# P2: Autocorrelation & Direction Persistence
# ---------------------------------------------------------------------------


def autocorrelation(price_changes: pd.Series, max_lag: int = 20) -> pd.Series:
    """Compute autocorrelation of price changes at each lag.

    Parameters
    ----------
    price_changes:
        Daily price change series.
    max_lag:
        Maximum lag to compute.

    Returns
    -------
    Series indexed by lag (1..max_lag) with autocorrelation values.
    """
    acf_values = []
    for lag in range(1, max_lag + 1):
        acf_values.append(price_changes.autocorr(lag=lag))
    return pd.Series(acf_values, index=range(1, max_lag + 1), name="autocorrelation")


def _collect_streaks(
    directions: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Walk direction signs and return (up_streaks, down_streaks)."""
    up_streaks: list[int] = []
    down_streaks: list[int] = []
    current_streak = 1

    for i in range(1, len(directions)):
        if directions[i] == directions[i - 1] and directions[i] != 0:
            current_streak += 1
        else:
            if directions[i - 1] > 0:
                up_streaks.append(current_streak)
            elif directions[i - 1] < 0:
                down_streaks.append(current_streak)
            current_streak = 1

    # Don't forget the last streak
    if len(directions) > 0:
        if directions[-1] > 0:
            up_streaks.append(current_streak)
        elif directions[-1] < 0:
            down_streaks.append(current_streak)

    return up_streaks, down_streaks


def compute_direction_streaks(price_changes: pd.Series) -> dict[str, Any]:
    """Compute statistics on consecutive same-direction runs.

    Parameters
    ----------
    price_changes:
        Daily price change series.

    Returns
    -------
    Dictionary with max/mean streak lengths for up and down.
    """
    directions = np.sign(price_changes.values)
    up_streaks, down_streaks = _collect_streaks(directions)

    return {
        "max_up_streak": max(up_streaks) if up_streaks else 0,
        "max_down_streak": max(down_streaks) if down_streaks else 0,
        "mean_up_streak": float(np.mean(up_streaks)) if up_streaks else 0.0,
        "mean_down_streak": float(np.mean(down_streaks)) if down_streaks else 0.0,
        "n_up_runs": len(up_streaks),
        "n_down_runs": len(down_streaks),
    }


# ---------------------------------------------------------------------------
# P3: Forecast Error Analysis
# ---------------------------------------------------------------------------


def compute_forecast_errors(actual: pd.Series, forecast: pd.Series) -> pd.DataFrame:
    """Compute forecast errors: error, absolute error, percentage error.

    Parameters
    ----------
    actual:
        Actual observed values.
    forecast:
        Forecasted values (same index as actual).

    Returns
    -------
    DataFrame with columns: error, abs_error, pct_error.
    error = forecast - actual (positive means forecast too high).
    pct_error = error / actual * 100 (NaN where actual is zero).
    """
    error = forecast - actual
    abs_error = error.abs()
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_error = (error / actual * 100).where(actual != 0, np.nan)

    return pd.DataFrame(
        {"error": error, "abs_error": abs_error, "pct_error": pct_error},
        index=actual.index,
    )


# ---------------------------------------------------------------------------
# P4: Lagged Feature -> Direction Correlation
# ---------------------------------------------------------------------------


def lagged_direction_correlation(features: pd.DataFrame, direction: pd.Series) -> pd.Series:
    """Correlate each feature with the direction signal.

    Both features and direction should be aligned on the same index
    (i.e., features are already lagged if needed).

    Parameters
    ----------
    features:
        DataFrame of feature columns.
    direction:
        Series of direction values (+1, -1, 0).

    Returns
    -------
    Series of Pearson correlations, indexed by feature name,
    sorted by absolute value (descending).
    """
    combined = features.join(direction, how="inner")
    dir_name = direction.name or "direction"
    correlations = combined.corr()[dir_name].drop(dir_name)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# P5: Volatility & Regime Detection
# ---------------------------------------------------------------------------


def rolling_volatility(price_changes: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling standard deviation of price changes.

    Parameters
    ----------
    price_changes:
        Daily price change series.
    window:
        Rolling window size in days.

    Returns
    -------
    Rolling volatility series (first window-1 values are NaN).
    """
    vol = price_changes.rolling(window=window).std()
    vol.name = "rolling_volatility"
    return vol


# ---------------------------------------------------------------------------
# P6: Residual Load
# ---------------------------------------------------------------------------


def compute_residual_load(load: pd.Series, renewable_generation: pd.Series) -> pd.Series:
    """Compute residual load (load minus renewable generation).

    Parameters
    ----------
    load:
        Total load series (MW).
    renewable_generation:
        Total renewable generation series (MW).

    Returns
    -------
    Residual load series (MW).
    """
    residual = load - renewable_generation
    residual.name = "residual_load_mw"
    return residual


# ---------------------------------------------------------------------------
# Re-exports from eda_analysis_advanced (backward compatibility)
# ---------------------------------------------------------------------------
