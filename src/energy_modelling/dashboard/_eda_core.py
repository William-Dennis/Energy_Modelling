"""Core EDA tab — data loading and main render dispatcher."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from energy_modelling.dashboard._eda_constants import (
    BACKTEST_DATA_PATH,
    DATA_PATH,
    FOSSIL_COLS,
    PRICE_COL,
    RENEWABLE_COLS,
)
from energy_modelling.dashboard._eda_sections_basic import (
    _section_generation,
    _section_load,
    _section_neighbours,
    _section_overview,
    _section_price_ts,
)
from energy_modelling.dashboard._eda_sections_market import (
    _section_commodities,
    _section_correlations,
    _section_heatmap,
    _section_scatter,
    _section_weather,
)
from energy_modelling.dashboard._eda_sections_distributions import (
    _section_distributions,
    _section_negative,
)
from energy_modelling.dashboard._eda_sections_feedback import (
    _section_dow_edge_stability,
    _section_feature_drift,
    _section_quarterly_patterns,
    _section_volatility_regime_performance,
)
from energy_modelling.dashboard._eda_sections_forecasts import (
    _section_feature_importance,
    _section_forecast_errors,
)
from energy_modelling.dashboard._eda_sections_signals import (
    _section_strategy_correlation_insights,
    _section_wind_quintile,
)
from energy_modelling.dashboard._eda_sections_trading import (
    _section_autocorrelation,
    _section_price_changes,
)
from energy_modelling.dashboard._eda_sections_volatility import (
    _section_residual_load,
    _section_volatility_regimes,
)
from energy_modelling.dashboard.eda_analysis import clean_hourly_data


@st.cache_data
def _load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = clean_hourly_data(df)
    df["hour"] = df.index.hour  # type: ignore[union-attr]
    df["dayofweek"] = df.index.dayofweek  # type: ignore[union-attr]
    df["month"] = df.index.month  # type: ignore[union-attr]
    df["year"] = df.index.year  # type: ignore[union-attr]
    df["date"] = df.index.date  # type: ignore[union-attr]

    gen_cols = [c for c in df.columns if c.startswith("gen_") and c.endswith("_mw")]
    df["total_generation_mw"] = df[gen_cols].sum(axis=1)
    df["renewable_generation_mw"] = df[[c for c in RENEWABLE_COLS if c in df.columns]].sum(axis=1)
    df["fossil_generation_mw"] = df[[c for c in FOSSIL_COLS if c in df.columns]].sum(axis=1)
    df["renewable_share_pct"] = (
        df["renewable_generation_mw"] / df["total_generation_mw"] * 100
    ).clip(0, 100)
    return df


@st.cache_data
def _load_backtest_data() -> pd.DataFrame | None:
    """Load the daily backtest CSV for Phase 6 feedback sections."""
    if not BACKTEST_DATA_PATH.exists():
        return None
    df = pd.read_csv(BACKTEST_DATA_PATH, parse_dates=["delivery_date"])
    df["year"] = df["delivery_date"].dt.year
    return df


def render() -> None:
    """Render the EDA tab contents."""
    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Run `uv run collect-data --step all --kaggle --years 2019 ... --years 2025` first."
        )
        return

    df = _load_data()

    years = sorted(df["year"].unique())
    year_range = st.slider(
        "Year range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years))),
        key="eda_year_range",
    )
    mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    dff = df[mask]

    _render_basic_sections(dff)
    _render_trading_sections(dff)
    _render_feedback_sections(dff, year_range)


def _render_basic_sections(dff: pd.DataFrame) -> None:
    _section_overview(dff)
    _section_price_ts(dff)
    _section_generation(dff)
    _section_load(dff)
    _section_neighbours(dff)
    _section_commodities(dff)
    _section_weather(dff)
    _section_correlations(dff)
    _section_distributions(dff)
    _section_negative(dff)
    _section_heatmap(dff)
    _section_scatter(dff)


def _render_trading_sections(dff: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Trading Signal Analysis")
    st.caption(
        "The sections below analyse daily price *changes* (settlement - last settlement) "
        "— the actual quantity that trading strategies must predict."
    )
    _section_price_changes(dff)
    _section_autocorrelation(dff)
    _section_forecast_errors(dff)
    _section_feature_importance(dff)
    _section_volatility_regimes(dff)
    _section_residual_load(dff)


def _render_feedback_sections(
    dff: pd.DataFrame, year_range: tuple,
) -> None:
    backtest_df = _load_backtest_data()
    if backtest_df is not None:
        cdf = backtest_df[
            (backtest_df["year"] >= year_range[0]) & (backtest_df["year"] <= year_range[1])
        ]
        if not cdf.empty:
            st.divider()
            st.subheader("Strategy Feedback Analysis (Phase 6)")
            st.caption(
                "The sections below feed Phase 5 strategy performance insights "
                "back into deeper EDA — using the daily backtest dataset."
            )
            _section_dow_edge_stability(cdf)
            _section_feature_drift(cdf)
            _section_quarterly_patterns(cdf)
            _section_volatility_regime_performance(cdf)
            _section_wind_quintile(cdf)
            _section_strategy_correlation_insights(cdf)
