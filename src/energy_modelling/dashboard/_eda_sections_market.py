"""Market EDA sections: commodities, weather, correlations, heatmap, scatter."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from energy_modelling.dashboard._eda_constants import (
    GEN_COLS_DISPLAY,
    PRICE_COL,
    WEATHER_COLS_DISPLAY,
    _AGG_FREQ,
)


def _section_commodities(dff: pd.DataFrame) -> None:
    st.header("Carbon & Gas Prices")
    cols = [c for c in ["carbon_price_usd", "gas_price_usd"] if c in dff.columns]
    if not cols:
        st.info("No carbon or gas price data available.")
        return
    agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_comm_agg",
        index=2,
    )
    freq = _AGG_FREQ[agg]
    ts = dff[cols].resample(freq).mean()
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=cols,
        labels={"timestamp_utc": "", "value": "USD", "variable": "Commodity"},
        title=f"Carbon & Gas Prices ({agg})",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _section_weather(dff: pd.DataFrame) -> None:
    st.header("Weather Overview")
    avail = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
    if not avail:
        st.info("No weather data available.")
        return
    var = st.selectbox(
        "Weather variable",
        avail,
        format_func=lambda x: WEATHER_COLS_DISPLAY[x],
        key="eda_weather_var",
    )
    weather_agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_weather_agg",
        index=2,
    )
    weather_freq = _AGG_FREQ[weather_agg]
    ts = dff[[var]].resample(weather_freq).mean()
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=var,
        labels={"timestamp_utc": "", var: WEATHER_COLS_DISPLAY[var]},
        title=f"{WEATHER_COLS_DISPLAY[var]} ({weather_agg} Mean)",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_correlations(dff: pd.DataFrame) -> None:
    st.header("Key Correlations with Price")
    gen_cols = [c for c in GEN_COLS_DISPLAY if c in dff.columns]
    weather_cols = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
    corr_cols = (
        [
            "renewable_share_pct",
            "total_generation_mw",
            "fossil_generation_mw",
            "renewable_generation_mw",
        ]
        + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
        + weather_cols
        + gen_cols
    )
    corr_cols = [c for c in corr_cols if c in dff.columns]
    corr = dff[corr_cols + [PRICE_COL]].corr()[PRICE_COL].drop(PRICE_COL).sort_values()

    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        labels={"x": "Correlation with Price", "y": ""},
        title="Feature Correlations with Day-Ahead Price",
        color=corr.values,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
    )
    fig.update_layout(height=max(600, len(corr_cols) * 22), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _section_heatmap(dff: pd.DataFrame) -> None:
    st.header("Correlation Heatmap")
    gen_cols = [c for c in GEN_COLS_DISPLAY if c in dff.columns]
    weather_cols = [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
    cols = [PRICE_COL] + gen_cols + weather_cols
    for extra in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"]:
        if extra in dff.columns:
            cols.append(extra)
    labels = (
        ["Price"]
        + [GEN_COLS_DISPLAY.get(c, c) for c in gen_cols]
        + [WEATHER_COLS_DISPLAY.get(c, c) for c in weather_cols]
        + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
    )
    corr = dff[cols].corr()
    fig = px.imshow(
        corr.values,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Full Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(height=800, width=900)
    st.plotly_chart(fig, use_container_width=True)


def _section_scatter(dff: pd.DataFrame) -> None:
    st.header("Renewable Share vs Price")
    sdf = dff[[PRICE_COL, "renewable_share_pct", "year"]].dropna()
    if len(sdf) > 10_000:
        sdf = sdf.sample(10_000, random_state=42)
    fig = px.scatter(
        sdf,
        x="renewable_share_pct",
        y=PRICE_COL,
        color="year",
        opacity=0.4,
        labels={"renewable_share_pct": "Renewable Share [%]", PRICE_COL: "EUR/MWh"},
        title="Day-Ahead Price vs Renewable Share (10k sample)",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
