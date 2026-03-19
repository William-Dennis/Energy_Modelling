"""Pure EDA computation functions (no Streamlit dependency).

These functions provide trading-relevant analyses that power the
deepened EDA dashboard sections added in Phase 2. Each function is
independently testable and operates on pandas/numpy data.

Functions are grouped by the priority ranking from Phase 1 audit:
  P0 - Data pre-processing / cleaning
  P1 - Price change distribution
  P2 - Autocorrelation & direction persistence
  P3 - Forecast error analysis
  P4 - Lagged feature -> direction correlation
  P5 - Volatility & regime detection
  P6 - Residual load
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# P0: Data Pre-processing
# ---------------------------------------------------------------------------

# Interconnector columns that did not exist before a certain date and should
# be zero-filled rather than dropped or forward-filled.
_INTERCONNECTORS_FILL_ZERO = (
    "ntc_dk_2_export_mw",
    "ntc_dk_2_import_mw",
    "ntc_nl_export_mw",
    "ntc_nl_import_mw",
)

# Commodity columns that are missing on weekends / holidays and should be
# linearly interpolated then back-filled.
_COMMODITY_COLS = ("carbon_price_usd", "gas_price_usd")


def clean_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-specific cleaning to the raw hourly DE-LU dataset.

    The cleaning steps replicate the notebook
    ``notebooks/baseline-day-ahead-price-prediction-for-deu.ipynb`` and
    are designed to fix **all** NaN values without dropping any rows:

    1. Drop the first row (all-NaN artefact of the data collection join).
    2. Forward-fill any column that has exactly 1 missing value.
    3. Fill interconnector NTC columns with 0 (capacity did not exist).
    4. Linearly interpolate + back-fill commodity prices (weekend gaps).
    5. Fill ``load_forecast_mw`` gaps with the value from 24 h prior.

    Parameters
    ----------
    df:
        Raw hourly DataFrame with DatetimeIndex (as read from parquet).

    Returns
    -------
    Cleaned DataFrame with the same shape (minus the first row) and
    zero remaining NaN values.
    """
    df = df.copy()

    # Step 1: drop the first row (artefact)
    df = df.iloc[1:]

    # Step 2: forward-fill single-NaN columns
    for col in df.columns:
        if df[col].isna().sum() == 1:
            df[col] = df[col].ffill()

    # Step 3: zero-fill interconnector NTCs
    for col in _INTERCONNECTORS_FILL_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Step 4: interpolate commodity prices
    for col in _COMMODITY_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")
            df[col] = df[col].bfill()

    # Step 5: fill load_forecast_mw with 24-hour-prior value
    if "load_forecast_mw" in df.columns:
        mask = df["load_forecast_mw"].isna()
        if mask.any():
            for idx in df.index[mask]:
                prior = idx - pd.Timedelta("1d")
                if prior in df.index:
                    df.loc[idx, "load_forecast_mw"] = df.loc[prior, "load_forecast_mw"]
            # Safety: if any remain (edge case at start of dataset), ffill
            df["load_forecast_mw"] = df["load_forecast_mw"].ffill()

    return df


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
# Grouping helper
# ---------------------------------------------------------------------------


def direction_by_group(price_changes: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Compute direction win rates broken down by a grouping variable.

    Parameters
    ----------
    price_changes:
        Daily price change series.
    groups:
        Categorical grouping variable (same index as price_changes).

    Returns
    -------
    DataFrame indexed by group with columns: n_total, n_up, n_down, pct_up, pct_down.
    """
    df = pd.DataFrame({"change": price_changes, "group": groups})
    df["is_up"] = df["change"] > 0
    df["is_down"] = df["change"] < 0

    result = df.groupby("group").agg(
        n_total=("change", "count"),
        n_up=("is_up", "sum"),
        n_down=("is_down", "sum"),
    )
    result["pct_up"] = result["n_up"] / result["n_total"] * 100
    result["pct_down"] = result["n_down"] / result["n_total"] * 100
    return result


# ---------------------------------------------------------------------------
# Phase 6: Feedback-loop analyses driven by strategy performance insights
# ---------------------------------------------------------------------------


def day_of_week_edge_by_year(
    price_changes: pd.Series,
    dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute the directional edge per day-of-week per year.

    The "edge" for a given (year, day) is the day's up-rate minus the
    overall up-rate for that year.  This isolates the day-of-week effect
    from year-level trend shifts.

    Parameters
    ----------
    price_changes:
        Daily price change series.
    dates:
        Corresponding delivery dates (same length as price_changes).

    Returns
    -------
    DataFrame with columns: year, dow (0=Mon..6=Sun), up_rate, overall_up_rate, edge.
    """
    dates = pd.to_datetime(dates)
    df = pd.DataFrame(
        {
            "change": price_changes.values,
            "year": dates.year,
            "dow": dates.dayofweek,
        }
    )
    df["is_up"] = df["change"] > 0

    yearly_up = df.groupby("year")["is_up"].mean().rename("overall_up_rate")
    by_year_dow = df.groupby(["year", "dow"])["is_up"].mean().rename("up_rate").reset_index()
    by_year_dow = by_year_dow.merge(yearly_up, on="year")
    by_year_dow["edge"] = by_year_dow["up_rate"] - by_year_dow["overall_up_rate"]
    return by_year_dow


def feature_drift(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
) -> pd.DataFrame:
    """Measure distribution shift between training and validation feature sets.

    For each numeric column present in both DataFrames, compute:
    - train_mean, val_mean, shift_pct (relative change)
    - train_std, val_std, std_ratio

    Parameters
    ----------
    train_features:
        Training-period DataFrame.
    val_features:
        Validation-period DataFrame.

    Returns
    -------
    DataFrame indexed by feature name with drift statistics.
    """
    common = sorted(
        set(train_features.select_dtypes("number").columns)
        & set(val_features.select_dtypes("number").columns)
    )
    rows = []
    for col in common:
        tmean = float(train_features[col].mean())
        vmean = float(val_features[col].mean())
        tstd = float(train_features[col].std())
        vstd = float(val_features[col].std())
        shift_pct = (vmean - tmean) / tmean * 100 if tmean != 0 else 0.0
        std_ratio = vstd / tstd if tstd != 0 else float("inf")
        rows.append(
            {
                "feature": col,
                "train_mean": tmean,
                "val_mean": vmean,
                "shift_pct": shift_pct,
                "train_std": tstd,
                "val_std": vstd,
                "std_ratio": std_ratio,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def quarterly_direction_rates(
    price_changes: pd.Series,
    dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute up-rate and mean |price_change| by year and quarter.

    Parameters
    ----------
    price_changes:
        Daily price change series.
    dates:
        Corresponding delivery dates.

    Returns
    -------
    DataFrame with columns: year, quarter, up_rate, mean_abs_change, n.
    """
    dates = pd.to_datetime(dates)
    df = pd.DataFrame(
        {
            "change": price_changes.values,
            "year": dates.year,
            "quarter": dates.quarter,
        }
    )
    df["is_up"] = df["change"] > 0
    df["abs_change"] = df["change"].abs()

    result = (
        df.groupby(["year", "quarter"])
        .agg(
            up_rate=("is_up", "mean"),
            mean_abs_change=("abs_change", "mean"),
            n=("change", "count"),
        )
        .reset_index()
    )
    return result


def volatility_regime_performance(
    price_changes: pd.Series,
    window: int = 30,
    n_regimes: int = 3,
) -> pd.DataFrame:
    """Classify each day into a volatility regime and compute direction stats.

    Parameters
    ----------
    price_changes:
        Daily price change series.
    window:
        Rolling window for volatility calculation.
    n_regimes:
        Number of regimes (quantile bins). Default 3 = low/mid/high.

    Returns
    -------
    DataFrame with columns: regime, n, up_rate, mean_abs_change, mean_change.
    """
    vol = price_changes.rolling(window=window).std()
    valid = pd.DataFrame(
        {
            "change": price_changes,
            "volatility": vol,
        }
    ).dropna()

    labels = ["low", "mid", "high"] if n_regimes == 3 else [f"q{i + 1}" for i in range(n_regimes)]
    valid["regime"] = pd.qcut(valid["volatility"], n_regimes, labels=labels, duplicates="drop")

    result = (
        valid.groupby("regime", observed=True)
        .agg(
            n=("change", "count"),
            up_rate=("change", lambda x: (x > 0).mean()),
            mean_abs_change=("change", lambda x: x.abs().mean()),
            mean_change=("change", "mean"),
        )
        .reset_index()
    )
    return result


def wind_quintile_analysis(
    combined_wind: pd.Series,
    direction: pd.Series,
    price_changes: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Analyze direction rates by wind power quintile.

    Parameters
    ----------
    combined_wind:
        Combined wind forecast/generation series (offshore + onshore).
    direction:
        Target direction (+1/-1).
    price_changes:
        Daily price change series.
    n_bins:
        Number of quantile bins.

    Returns
    -------
    DataFrame with columns: wind_bin, n, up_rate, mean_price_change.
    """
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    df = pd.DataFrame(
        {
            "wind": combined_wind.values,
            "direction": direction.values,
            "change": price_changes.values,
        }
    ).dropna()
    df["wind_bin"] = pd.qcut(df["wind"], n_bins, labels=labels, duplicates="drop")

    result = (
        df.groupby("wind_bin", observed=True)
        .agg(
            n=("change", "count"),
            up_rate=("direction", lambda x: (x > 0).mean()),
            mean_price_change=("change", "mean"),
        )
        .reset_index()
    )
    return result
