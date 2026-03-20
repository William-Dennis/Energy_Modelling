"""Advanced EDA analyses (feedback-loop and grouping functions).

Split from eda_analysis.py to keep module sizes manageable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

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
# Feedback-loop analyses driven by strategy performance insights
# ---------------------------------------------------------------------------


def day_of_week_edge_by_year(
    price_changes: pd.Series,
    dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute the directional edge per day-of-week per year.

    The "edge" for a given (year, day) is the day's up-rate minus the
    overall up-rate for that year.

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
    dates = pd.Series(pd.to_datetime(dates))
    df = pd.DataFrame(
        {
            "change": price_changes.values,
            "year": dates.dt.year.values,
            "dow": dates.dt.dayofweek.values,
        }
    )
    df["is_up"] = df["change"] > 0

    yearly_up = df.groupby("year")["is_up"].mean().rename("overall_up_rate")
    by_year_dow = df.groupby(["year", "dow"])["is_up"].mean().rename("up_rate").reset_index()
    by_year_dow = by_year_dow.merge(yearly_up, on="year")
    by_year_dow["edge"] = by_year_dow["up_rate"] - by_year_dow["overall_up_rate"]
    return by_year_dow


def _compute_single_feature_drift(
    train_col: pd.Series,
    val_col: pd.Series,
) -> dict[str, Any]:
    """Compute drift statistics for a single feature column."""
    tmean = float(train_col.mean())
    vmean = float(val_col.mean())
    tstd = float(train_col.std())
    vstd = float(val_col.std())
    shift_pct = (vmean - tmean) / tmean * 100 if tmean != 0 else 0.0
    std_ratio = vstd / tstd if tstd != 0 else float("inf")
    return {
        "train_mean": tmean,
        "val_mean": vmean,
        "shift_pct": shift_pct,
        "train_std": tstd,
        "val_std": vstd,
        "std_ratio": std_ratio,
    }


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
        stats = _compute_single_feature_drift(train_features[col], val_features[col])
        stats["feature"] = col
        rows.append(stats)
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
            "year": pd.DatetimeIndex(dates).year,
            "quarter": pd.DatetimeIndex(dates).quarter,
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


def _classify_volatility_regimes(
    price_changes: pd.Series,
    window: int,
    n_regimes: int,
) -> pd.DataFrame:
    """Return a DataFrame with change, volatility, and regime labels."""
    vol = price_changes.rolling(window=window).std()
    valid = pd.DataFrame(
        {
            "change": price_changes,
            "volatility": vol,
        }
    ).dropna()

    labels = ["low", "mid", "high"] if n_regimes == 3 else [f"q{i + 1}" for i in range(n_regimes)]
    valid["regime"] = pd.qcut(valid["volatility"], n_regimes, labels=labels, duplicates="drop")
    return valid


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
    valid = _classify_volatility_regimes(price_changes, window, n_regimes)

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


def _bin_and_aggregate_wind(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Bin wind values into quantiles and aggregate direction stats."""
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    df["wind_bin"] = pd.qcut(df["wind"], n_bins, labels=labels, duplicates="drop")

    return (
        df.groupby("wind_bin", observed=True)
        .agg(
            n=("change", "count"),
            up_rate=("direction", lambda x: (x > 0).mean()),
            mean_price_change=("change", "mean"),
        )
        .reset_index()
    )


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
    df = pd.DataFrame(
        {
            "wind": combined_wind.values,
            "direction": direction.values,
            "change": price_changes.values,
        }
    ).dropna()
    return _bin_and_aggregate_wind(df, n_bins)
