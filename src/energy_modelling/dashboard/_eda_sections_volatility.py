"""Volatility regime and residual load EDA sections."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard._eda_constants import PRICE_COL, RENEWABLE_COLS
from energy_modelling.dashboard.eda_analysis import (
    compute_daily_settlement,
    compute_price_changes,
    compute_residual_load,
    direction_base_rates,
    rolling_volatility,
)


def _section_volatility_regimes(dff: pd.DataFrame) -> None:
    """P5: Volatility clustering and regime detection."""
    st.header("17. Volatility & Regime Analysis")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)

    window = st.slider(
        "Rolling window (days)",
        min_value=7,
        max_value=90,
        value=30,
        key="eda_vol_window",
    )
    vol = rolling_volatility(changes, window=window)

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_volatility_chart(changes, vol, window)
    with col_r:
        _display_regime_comparison(changes, vol)
    _plot_yearly_volatility(changes, dff)


def _plot_volatility_chart(
    changes: pd.Series, vol: pd.Series, window: int,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=changes.index,
            y=changes.values,
            mode="lines",
            name="Daily Change",
            line={"color": "steelblue", "width": 0.8},
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=vol.index,
            y=vol.values,
            mode="lines",
            name=f"{window}d Volatility",
            line={"color": "red", "width": 2},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"Price Changes & {window}-Day Rolling Volatility",
        yaxis={"title": "Price Change (EUR/MWh)"},
        yaxis2={
            "title": "Volatility (EUR/MWh)",
            "overlaying": "y",
            "side": "right",
        },
        height=450,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_regime_comparison(
    changes: pd.Series, vol: pd.Series,
) -> None:
    vol_median = vol.dropna().median()
    vol_clean = vol.dropna()
    is_high_vol = vol_clean > vol_median

    changes_aligned = changes.loc[vol_clean.index]
    high_vol_changes = changes_aligned[is_high_vol]
    low_vol_changes = changes_aligned[~is_high_vol]

    if len(high_vol_changes) > 0 and len(low_vol_changes) > 0:
        high_rates = direction_base_rates(high_vol_changes)
        low_rates = direction_base_rates(low_vol_changes)
        regime_df = _build_regime_df(high_rates, low_rates, high_vol_changes, low_vol_changes)
        st.markdown("**Regime Comparison** (split at median volatility)")
        st.dataframe(regime_df, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data to split into regimes.")


def _build_regime_df(
    high_rates: dict, low_rates: dict,
    high_vol_changes: pd.Series, low_vol_changes: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [
                "% Up days", "Mean up move (EUR)", "Mean down move (EUR)",
                "Std dev (EUR)", "N days",
            ],
            "High Volatility": [
                f"{high_rates['pct_up']:.1f}%",
                f"{high_rates['mean_up_move']:.2f}",
                f"{high_rates['mean_down_move']:.2f}",
                f"{high_vol_changes.std():.2f}",
                str(high_rates["n_total"]),
            ],
            "Low Volatility": [
                f"{low_rates['pct_up']:.1f}%",
                f"{low_rates['mean_up_move']:.2f}",
                f"{low_rates['mean_down_move']:.2f}",
                f"{low_vol_changes.std():.2f}",
                str(low_rates["n_total"]),
            ],
        }
    )


def _plot_yearly_volatility(changes: pd.Series, dff: pd.DataFrame) -> None:
    if "year" not in dff.columns:
        return
    yearly_vol = changes.groupby(changes.index.year).std().reset_index()
    yearly_vol.columns = ["year", "annual_std"]
    fig = px.bar(
        yearly_vol,
        x="year",
        y="annual_std",
        title="Price Change Volatility by Year",
        labels={"year": "Year", "annual_std": "Std Dev (EUR/MWh)"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_residual_load(dff: pd.DataFrame) -> None:
    """P6: Residual load (load - renewables) analysis."""
    st.header("18. Residual Load Analysis")

    if "load_actual_mw" not in dff.columns:
        st.info("Load data not available for residual load analysis.")
        return

    ren_cols = [c for c in RENEWABLE_COLS if c in dff.columns]
    if not ren_cols:
        st.info("Renewable generation data not available.")
        return

    load = dff["load_actual_mw"]
    renewable = dff[ren_cols].sum(axis=1)
    residual = compute_residual_load(load, renewable)

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_residual_vs_price(residual, dff)
    with col_r:
        _plot_residual_change_vs_price(residual, dff)
    _plot_residual_timeseries(residual, load, renewable)


def _plot_residual_vs_price(
    residual: pd.Series, dff: pd.DataFrame,
) -> None:
    daily_residual = residual.groupby(residual.index.date).mean()
    daily_residual.index = pd.DatetimeIndex(daily_residual.index)
    daily_price = dff[PRICE_COL].groupby(dff.index.date).mean()
    daily_price.index = pd.DatetimeIndex(daily_price.index)

    scatter_df = pd.DataFrame(
        {"Residual Load (MW)": daily_residual, "Price (EUR/MWh)": daily_price}
    ).dropna()

    if len(scatter_df) > 5_000:
        scatter_df = scatter_df.sample(5_000, random_state=42)

    fig = px.scatter(
        scatter_df,
        x="Residual Load (MW)",
        y="Price (EUR/MWh)",
        opacity=0.4,
        title="Daily Residual Load vs Price",
        trendline="ols",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def _plot_residual_change_vs_price(
    residual: pd.Series, dff: pd.DataFrame,
) -> None:
    daily_residual = residual.groupby(residual.index.date).mean()
    daily_residual.index = pd.DatetimeIndex(daily_residual.index)
    daily_residual_change = daily_residual.diff().dropna()
    settlements = compute_daily_settlement(dff[PRICE_COL])
    price_change = compute_price_changes(settlements)

    common = daily_residual_change.index.intersection(price_change.index)
    if len(common) > 10:
        delta_df = pd.DataFrame(
            {
                "Residual Load Change (MW)": daily_residual_change.loc[common],
                "Price Change (EUR/MWh)": price_change.loc[common],
            }
        ).dropna()

        fig = px.scatter(
            delta_df,
            x="Residual Load Change (MW)",
            y="Price Change (EUR/MWh)",
            opacity=0.4,
            title="Change in Residual Load vs Change in Price",
            trendline="ols",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for residual load change analysis.")


def _plot_residual_timeseries(
    residual: pd.Series,
    load: pd.Series,
    renewable: pd.Series,
) -> None:
    ts = (
        pd.DataFrame(
            {"Residual Load": residual, "Load": load, "Renewables": renewable}
        )
        .resample("1W")
        .mean()
    )
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=["Residual Load", "Load", "Renewables"],
        labels={"timestamp_utc": "", "value": "MW", "variable": ""},
        title="Weekly Average: Load, Renewables, and Residual Load",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
