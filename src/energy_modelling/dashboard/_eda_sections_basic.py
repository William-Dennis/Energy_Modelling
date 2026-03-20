"""Basic EDA sections: overview, price time-series, generation, load, neighbours."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard._eda_constants import (
    GEN_COLORS,
    GEN_COLS_DISPLAY,
    PRICE_COL,
    _AGG_FREQ,
    _DISPLAY_ORDER,
)


def _section_overview(dff: pd.DataFrame) -> None:
    st.header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dff):,}")
    c2.metric("Columns", f"{len(dff.columns)}")
    c3.metric(
        "Date range",
        f"{dff.index.min().strftime('%Y-%m-%d')} to {dff.index.max().strftime('%Y-%m-%d')}",
    )
    c4.metric("Hourly frequency", "1h")

    with st.expander("Descriptive statistics"):
        show_cols = [PRICE_COL] + list(GEN_COLS_DISPLAY) + list(
            {
                "weather_temperature_2m_degc": "Temperature (2m) [C]",
                "weather_relative_humidity_2m_pct": "Humidity (2m) [%]",
                "weather_wind_speed_10m_kmh": "Wind Speed (10m) [km/h]",
                "weather_wind_speed_100m_kmh": "Wind Speed (100m) [km/h]",
                "weather_shortwave_radiation_wm2": "GHI [W/m2]",
                "weather_direct_normal_irradiance_wm2": "DNI [W/m2]",
                "weather_precipitation_mm": "Precipitation [mm]",
            }
        )
        show_cols = [c for c in show_cols if c in dff.columns]
        st.dataframe(
            dff[show_cols].describe().round(2).T.style.format("{:.2f}"),
            use_container_width=True,
        )

    with st.expander("Missing data per column"):
        missing = dff.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values in the filtered data.")
        else:
            st.dataframe(
                pd.DataFrame(
                    {"Missing": missing, "Pct": (missing / len(dff) * 100).round(2)}
                ).sort_values("Pct", ascending=False)
            )


def _section_price_ts(dff: pd.DataFrame) -> None:
    st.header("Day-Ahead Price Time Series")
    agg = st.radio(
        "Aggregation",
        ["Hourly (raw)", "Daily mean", "Weekly mean", "Monthly mean"],
        horizontal=True,
        key="eda_price_agg",
        index=3,
    )
    freq_map = {
        "Hourly (raw)": None,
        "Daily mean": "1D",
        "Weekly mean": "1W",
        "Monthly mean": "1ME",
    }
    freq = freq_map[agg]
    ts = dff[[PRICE_COL]] if freq is None else dff[[PRICE_COL]].resample(freq).mean()
    fig = px.line(
        ts.reset_index(),
        x="timestamp_utc",
        y=PRICE_COL,
        labels={"timestamp_utc": "", PRICE_COL: "EUR/MWh"},
        title=f"Day-Ahead Price ({agg})",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean", f"{dff[PRICE_COL].mean():.1f} EUR/MWh")
    c2.metric("Median", f"{dff[PRICE_COL].median():.1f} EUR/MWh")
    c3.metric("Std Dev", f"{dff[PRICE_COL].std():.1f}")
    c4.metric("Min", f"{dff[PRICE_COL].min():.1f} EUR/MWh")
    c5.metric("Max", f"{dff[PRICE_COL].max():.1f} EUR/MWh")


def _section_generation(dff: pd.DataFrame) -> None:
    st.header("Generation Mix")
    gen_agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_gen_agg",
        index=2,
    )
    gen_freq = _AGG_FREQ[gen_agg]
    gen_cols = [c for c in GEN_COLS_DISPLAY if c in dff.columns]
    gen_rs = dff[gen_cols].resample(gen_freq).mean().rename(columns=GEN_COLS_DISPLAY)
    _plot_generation_mix(gen_rs, gen_agg)
    _plot_renewable_share(dff, gen_freq)


def _plot_generation_mix(gen_rs: pd.DataFrame, gen_agg: str) -> None:
    fig = go.Figure()
    for fuel in _DISPLAY_ORDER:
        if fuel in gen_rs.columns:
            fig.add_trace(
                go.Scatter(
                    x=gen_rs.index,
                    y=gen_rs[fuel],
                    name=fuel,
                    mode="lines",
                    stackgroup="one",
                    line={"width": 0},
                    fillcolor=GEN_COLORS.get(fuel, "#888888"),
                )
            )
    fig.update_layout(
        title=f"Generation Mix ({gen_agg} Average, MW)",
        yaxis_title="MW",
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_renewable_share(dff: pd.DataFrame, gen_freq: str) -> None:
    st.subheader("Renewable Share Over Time")
    ren = dff[["renewable_share_pct"]].resample(gen_freq).mean()
    fig = px.line(
        ren.reset_index(),
        x="timestamp_utc",
        y="renewable_share_pct",
        labels={"timestamp_utc": "", "renewable_share_pct": "Renewable %"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def _section_load(dff: pd.DataFrame) -> None:
    st.header("Load & Forecasts")
    load_cols = [c for c in ["load_actual_mw", "load_forecast_mw"] if c in dff.columns]
    fc_cols = [c for c in dff.columns if c.startswith("forecast_") and c.endswith("_mw")]
    if not load_cols and not fc_cols:
        st.info("No load or forecast data available.")
        return

    agg = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        horizontal=True,
        key="eda_load_agg",
        index=2,
    )
    freq = _AGG_FREQ[agg]

    if load_cols:
        ts = dff[load_cols].resample(freq).mean()
        fig = px.line(
            ts.reset_index(),
            x="timestamp_utc",
            y=load_cols,
            labels={"timestamp_utc": "", "value": "MW", "variable": "Series"},
            title=f"Total Load: Actual vs Forecast ({agg})",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if fc_cols:
        ts = dff[fc_cols].resample(freq).mean()
        fig = px.line(
            ts.reset_index(),
            x="timestamp_utc",
            y=fc_cols,
            labels={"timestamp_utc": "", "value": "MW", "variable": "Source"},
            title=f"Wind & Solar DA Forecast ({agg})",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def _section_neighbours(dff: pd.DataFrame) -> None:
    st.header("Neighbour Prices & Cross-Border Flows")
    from energy_modelling.dashboard._eda_constants import NEIGHBOUR_PRICE_COLS

    nb_avail = {k: v for k, v in NEIGHBOUR_PRICE_COLS.items() if k in dff.columns}
    flow_cols = [c for c in dff.columns if c.startswith("flow_") and c.endswith("_net_import_mw")]

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_neighbour_prices(dff, nb_avail)
    with col_r:
        _plot_cross_border_flows(dff, flow_cols)


def _plot_neighbour_prices(dff: pd.DataFrame, nb_avail: dict) -> None:
    if nb_avail:
        agg = st.radio(
            "Aggregation",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            key="eda_nb_agg",
            index=2,
        )
        freq = _AGG_FREQ[agg]
        plot_cols = [PRICE_COL] + list(nb_avail)
        labels = {"price_eur_mwh": "DE-LU", **nb_avail}
        ts = dff[plot_cols].resample(freq).mean().rename(columns=labels)
        fig = px.line(
            ts.reset_index(),
            x="timestamp_utc",
            y=list(labels.values()),
            labels={"timestamp_utc": "", "value": "EUR/MWh", "variable": "Zone"},
            title=f"Day-Ahead Prices: DE-LU vs Neighbours ({agg})",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No neighbour price data available.")


def _plot_cross_border_flows(dff: pd.DataFrame, flow_cols: list) -> None:
    if flow_cols:
        agg = st.radio(
            "Aggregation",
            ["Daily", "Weekly", "Monthly"],
            horizontal=True,
            key="eda_flow_agg",
            index=2,
        )
        freq = _AGG_FREQ[agg]
        ts = dff[flow_cols].resample(freq).mean()
        display = {
            c: c.replace("flow_", "").replace("_net_import_mw", "").upper() for c in flow_cols
        }
        ts = ts.rename(columns=display)
        fig = px.line(
            ts.reset_index(),
            x="timestamp_utc",
            y=list(display.values()),
            labels={"timestamp_utc": "", "value": "MW (+ = import)", "variable": "Border"},
            title=f"Net Cross-Border Flows ({agg})",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cross-border flow data available.")
