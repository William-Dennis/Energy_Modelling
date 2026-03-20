"""Derived feature engineering for the daily backtest frame.

Pure functions that compute derived features from the raw daily feature
columns. Each function takes a ``pd.DataFrame`` with the raw feature columns
already present and returns a new DataFrame with additional columns appended.

All derived features are look-ahead safe:
- Features derived from same-day forecast columns remain same-day aligned.
- Features derived from lagged realised columns are already 1-day lagged.
- Rolling statistics operate on the lagged ``price_change_eur_mwh`` series.

Groups
------
1. Supply/demand balance  — net_demand_mw, renewable_penetration_pct
2. Price spreads          — de_fr_spread, de_nl_spread, de_avg_neighbour_spread
3. Price mean-reversion   — price_zscore_20d, price_range
4. Commodity trends       — gas_trend_3d, carbon_trend_3d, fuel_cost_index
5. Surprise signals       — wind_forecast_error, load_surprise
6. Volatility regime      — rolling_vol_7d, rolling_vol_14d
7. Aggregated generation  — total_fossil_mw, net_flow_mw
8. Calendar encodings     — dow_int, is_weekend
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Group 1: Supply / demand balance
# ---------------------------------------------------------------------------

_LOAD_FCAST = "load_forecast_mw_mean"
_WIND_ON_FCAST = "forecast_wind_onshore_mw_mean"
_WIND_OFF_FCAST = "forecast_wind_offshore_mw_mean"
_SOLAR_FCAST = "forecast_solar_mw_mean"


def add_net_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``net_demand_mw``: load forecast minus all renewable forecasts.

    This is the demand that must be met by dispatchable (price-setting)
    generation. Higher net demand → higher marginal cost called → higher price.
    Correlation with direction: approximately +0.30 (strongest single signal).

    Returns 0.0 for the column if any required source column is absent.
    """
    required = [_LOAD_FCAST, _WIND_ON_FCAST, _WIND_OFF_FCAST, _SOLAR_FCAST]
    if not all(c in df.columns for c in required):
        return df.assign(net_demand_mw=0.0)
    net = df[_LOAD_FCAST] - df[_WIND_ON_FCAST] - df[_WIND_OFF_FCAST] - df[_SOLAR_FCAST]
    return df.assign(net_demand_mw=net)


def add_renewable_penetration(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``renewable_penetration_pct``: renewable share of total load forecast.

    High penetration suppresses prices via the merit-order effect.
    Guarded against zero load (returns 0.0 where load_forecast == 0).
    Returns 0.0 for the column if any required source column is absent.
    """
    required = [_LOAD_FCAST, _WIND_ON_FCAST, _WIND_OFF_FCAST, _SOLAR_FCAST]
    if not all(c in df.columns for c in required):
        return df.assign(renewable_penetration_pct=0.0)
    total_re = df[_WIND_ON_FCAST] + df[_WIND_OFF_FCAST] + df[_SOLAR_FCAST]
    load = df[_LOAD_FCAST].replace(0.0, np.nan)
    pct = total_re / load
    return df.assign(renewable_penetration_pct=pct.fillna(0.0))


# ---------------------------------------------------------------------------
# Group 2: Price spreads
# ---------------------------------------------------------------------------

_PRICE_MEAN = "price_mean"
_PRICE_FR = "price_fr_eur_mwh_mean"
_PRICE_NL = "price_nl_eur_mwh_mean"
_PRICE_AT = "price_at_eur_mwh_mean"
_PRICE_CZ = "price_cz_eur_mwh_mean"
_PRICE_PL = "price_pl_eur_mwh_mean"
_PRICE_DK1 = "price_dk_1_eur_mwh_mean"

_NEIGHBOUR_COLS = [_PRICE_FR, _PRICE_NL, _PRICE_AT, _PRICE_CZ, _PRICE_PL, _PRICE_DK1]


def add_price_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Add three DE-LU vs neighbour price spread columns.

    * ``de_fr_spread``: DE price − FR price (corr ≈ −0.132).
    * ``de_nl_spread``: DE price − NL price (corr ≈ −0.10).
    * ``de_avg_neighbour_spread``: DE price − mean of all 6 neighbours.

    Positive spread = DE more expensive than neighbour; negative = DE cheaper.
    When DE is cheaper, price tends to rise (convergence); when expensive, fall.

    Columns that are absent from *df* are silently skipped; their spread is
    set to 0.0 so downstream strategies always receive a finite value.
    """
    out = df.copy()
    price = df[_PRICE_MEAN] if _PRICE_MEAN in df.columns else pd.Series(0.0, index=df.index)

    if _PRICE_FR in df.columns:
        out["de_fr_spread"] = price - df[_PRICE_FR]
    else:
        out["de_fr_spread"] = 0.0

    if _PRICE_NL in df.columns:
        out["de_nl_spread"] = price - df[_PRICE_NL]
    else:
        out["de_nl_spread"] = 0.0

    present_neighbours = [c for c in _NEIGHBOUR_COLS if c in df.columns]
    if present_neighbours:
        neighbour_mean = df[present_neighbours].mean(axis=1)
        out["de_avg_neighbour_spread"] = price - neighbour_mean
    else:
        out["de_avg_neighbour_spread"] = 0.0

    return out


# ---------------------------------------------------------------------------
# Group 3: Price mean-reversion
# ---------------------------------------------------------------------------

_PRICE_MAX = "price_max"
_PRICE_MIN = "price_min"

_MA_WINDOW = 20
_MA_MIN_PERIODS = 5


def add_price_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``price_zscore_20d``: price deviation from 20-day rolling mean.

    Formula: (price_mean − MA20) / std20.
    Correlation with direction: approximately −0.173.
    Positive z-score (DE price elevated) → price likely to fall.
    Uses min_periods=5 to produce values early in the series.
    Returns 0.0 for the column if ``price_mean`` is absent.
    """
    if _PRICE_MEAN not in df.columns:
        return df.assign(price_zscore_20d=0.0)
    price = df[_PRICE_MEAN]
    ma = price.rolling(_MA_WINDOW, min_periods=_MA_MIN_PERIODS).mean()
    std = price.rolling(_MA_WINDOW, min_periods=_MA_MIN_PERIODS).std()
    std = std.replace(0.0, np.nan)
    zscore = (price - ma) / std
    return df.assign(price_zscore_20d=zscore.fillna(0.0))


def add_price_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``price_range``: intraday price spread (max − min, lagged).

    Returns 0.0 for the column if ``price_max`` or ``price_min`` is absent.
    """
    if _PRICE_MAX not in df.columns or _PRICE_MIN not in df.columns:
        return df.assign(price_range=0.0)
    return df.assign(price_range=df[_PRICE_MAX] - df[_PRICE_MIN])


# ---------------------------------------------------------------------------
# Group 4: Commodity trends
# ---------------------------------------------------------------------------

_GAS = "gas_price_usd_mean"
_CARBON = "carbon_price_usd_mean"

_TREND_WINDOW = 3
# CCGT heat rate (MWh_th / MWh_e) and emission factor (tCO2 / MWh_e)
_GAS_HEAT_RATE = 7.5
_EMISSION_FACTOR = 0.37


def add_commodity_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Add gas and carbon price trend signals plus a combined fuel cost index.

    * ``gas_trend_3d``: 3-day change in gas price (momentum, corr ≈ +0.07).
    * ``carbon_trend_3d``: 3-day change in carbon price (corr ≈ +0.05).
    * ``fuel_cost_index``: heat-rate-weighted CCGT fuel cost (level, not trend).
      Formula: gas_price × 7.5 + carbon_price × 0.37.
      Note: level correlation is weak; prefer trend-based signals.

    NaN values in trend columns (first 3 rows from .diff(3)) are filled with
    0.0 so downstream code and tests always receive finite values.
    Returns 0.0-filled columns if required source columns are absent.
    """
    out = df.copy()
    if _GAS in df.columns:
        out["gas_trend_3d"] = df[_GAS].diff(_TREND_WINDOW).fillna(0.0)
    else:
        out["gas_trend_3d"] = 0.0

    if _CARBON in df.columns:
        out["carbon_trend_3d"] = df[_CARBON].diff(_TREND_WINDOW).fillna(0.0)
    else:
        out["carbon_trend_3d"] = 0.0

    gas_vals = df[_GAS] if _GAS in df.columns else 0.0
    carbon_vals = df[_CARBON] if _CARBON in df.columns else 0.0
    out["fuel_cost_index"] = gas_vals * _GAS_HEAT_RATE + carbon_vals * _EMISSION_FACTOR
    return out


# ---------------------------------------------------------------------------
# Group 5: Surprise / forecast-error signals
# ---------------------------------------------------------------------------

_GEN_WIND_ON = "gen_wind_onshore_mw_mean"
_GEN_WIND_OFF = "gen_wind_offshore_mw_mean"
_LOAD_ACTUAL = "load_actual_mw_mean"


def add_surprise_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind forecast error and load surprise columns.

    * ``wind_forecast_error``: today's wind forecast minus yesterday's
      realised wind generation. Positive = forecast higher than yesterday's
      actual → greater supply expected → bearish signal.
    * ``load_surprise``: today's load forecast minus yesterday's actual load.
      Positive = expecting more demand than yesterday → bullish signal.

    Both columns use already-lagged realised columns (gen_wind, load_actual)
    so there is no look-ahead bias. The first row will be NaN; filled with 0.
    Returns 0.0-filled columns if required source columns are absent.
    """
    fcast_wind_on = df[_WIND_ON_FCAST] if _WIND_ON_FCAST in df.columns else 0.0
    fcast_wind_off = df[_WIND_OFF_FCAST] if _WIND_OFF_FCAST in df.columns else 0.0
    actual_wind_on = df[_GEN_WIND_ON] if _GEN_WIND_ON in df.columns else 0.0
    actual_wind_off = df[_GEN_WIND_OFF] if _GEN_WIND_OFF in df.columns else 0.0
    load_fcast = df[_LOAD_FCAST] if _LOAD_FCAST in df.columns else 0.0
    load_actual = df[_LOAD_ACTUAL] if _LOAD_ACTUAL in df.columns else 0.0

    fcast_wind = fcast_wind_on + fcast_wind_off
    actual_wind = actual_wind_on + actual_wind_off
    wind_err = fcast_wind - actual_wind
    load_surp = load_fcast - load_actual

    if hasattr(wind_err, "fillna"):
        wind_err = wind_err.fillna(0.0)
    if hasattr(load_surp, "fillna"):
        load_surp = load_surp.fillna(0.0)

    return df.assign(wind_forecast_error=wind_err, load_surprise=load_surp)


# ---------------------------------------------------------------------------
# Group 6: Volatility regime
# ---------------------------------------------------------------------------

_PRICE_CHANGE = "price_change_eur_mwh"

_VOL_7_WINDOW = 7
_VOL_14_WINDOW = 14
_VOL_MIN_PERIODS = 3


def add_rolling_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling price-change standard deviation over 7 and 14 days.

    * ``rolling_vol_7d``: short-term volatility regime indicator.
    * ``rolling_vol_14d``: medium-term volatility regime indicator.

    These are computed on ``price_change_eur_mwh`` which is already realised
    (lagged). They serve as regime indicators for conditional strategies.
    Uses min_periods=3 so regime is available near the start of the series.
    Returns 0.0-filled columns if ``price_change_eur_mwh`` is absent.
    """
    if _PRICE_CHANGE not in df.columns:
        return df.assign(rolling_vol_7d=0.0, rolling_vol_14d=0.0)
    changes = df[_PRICE_CHANGE]
    vol7 = changes.rolling(_VOL_7_WINDOW, min_periods=_VOL_MIN_PERIODS).std()
    vol14 = changes.rolling(_VOL_14_WINDOW, min_periods=_VOL_MIN_PERIODS).std()
    return df.assign(
        rolling_vol_7d=vol7.fillna(0.0),
        rolling_vol_14d=vol14.fillna(0.0),
    )


# ---------------------------------------------------------------------------
# Group 7: Aggregated generation / flow
# ---------------------------------------------------------------------------

_GEN_GAS = "gen_fossil_gas_mw_mean"
_GEN_COAL = "gen_fossil_hard_coal_mw_mean"
_GEN_LIGNITE = "gen_fossil_brown_coal_lignite_mw_mean"
_FLOW_FR = "flow_fr_net_import_mw_mean"
_FLOW_NL = "flow_nl_net_import_mw_mean"


def add_aggregated_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Add total fossil dispatch and net cross-border flow columns.

    * ``total_fossil_mw``: sum of gas + hard coal + lignite generation
      (yesterday, lagged). Combined fossil dispatch signal (corr ≈ −0.20).
    * ``net_flow_mw``: sum of FR and NL net import flows. Negative values
      mean DE is a net exporter; positive means DE is importing.

    Returns 0.0 for missing constituent columns (treats absent generation
    as zero contribution to the sum).
    """
    gas = df[_GEN_GAS] if _GEN_GAS in df.columns else 0.0
    coal = df[_GEN_COAL] if _GEN_COAL in df.columns else 0.0
    lignite = df[_GEN_LIGNITE] if _GEN_LIGNITE in df.columns else 0.0
    flow_fr = df[_FLOW_FR] if _FLOW_FR in df.columns else 0.0
    flow_nl = df[_FLOW_NL] if _FLOW_NL in df.columns else 0.0
    return df.assign(
        total_fossil_mw=gas + coal + lignite,
        net_flow_mw=flow_fr + flow_nl,
    )


# ---------------------------------------------------------------------------
# Group 8: Calendar encodings
# ---------------------------------------------------------------------------

_DELIVERY_DATE = "delivery_date"


def add_calendar_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer day-of-week and weekend flag columns.

    * ``dow_int``: weekday as integer 1 (Monday) to 7 (Sunday).
    * ``is_weekend``: True for Saturday (6) and Sunday (7).

    These encode the strong day-of-week effect as numeric features for
    ML models that cannot natively handle dates.
    """
    dates = pd.to_datetime(df[_DELIVERY_DATE])
    dow = dates.dt.weekday + 1  # 1=Mon, 7=Sun
    return df.assign(
        dow_int=dow.values,
        is_weekend=(dow >= 6).values,
    )


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all derived feature groups to a daily backtest frame.

    This is the single entry point called from ``build_daily_backtest_frame()``.
    The input DataFrame must already contain the raw feature columns produced
    by ``build_daily_features()`` plus ``price_change_eur_mwh`` and
    ``delivery_date``.

    Returns the same DataFrame with all derived columns appended.
    """
    df = add_net_demand(df)
    df = add_renewable_penetration(df)
    df = add_price_spreads(df)
    df = add_price_zscore(df)
    df = add_price_range(df)
    df = add_commodity_trends(df)
    df = add_surprise_signals(df)
    df = add_rolling_volatility(df)
    df = add_aggregated_generation(df)
    df = add_calendar_encodings(df)
    return df
