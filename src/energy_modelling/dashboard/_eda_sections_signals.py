"""Phase 6 signal sections: wind quintile and strategy correlation insights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard.eda_analysis import wind_quintile_analysis


def _section_wind_quintile(cdf: pd.DataFrame) -> None:
    """Section 23: Wind quintile direction analysis."""
    st.header("23. Wind Quintile Analysis")
    st.caption(
        "The wind signal is the second-most reliable after Monday. "
        "Low wind -> up (62%), High wind -> down (68%). Is this stable?"
    )

    wind_cols = _resolve_wind_columns(cdf)
    if not wind_cols or "target_direction" not in cdf.columns:
        st.info("Wind or direction columns not available.")
        return

    combined_wind = cdf[wind_cols].sum(axis=1)
    wqa = wind_quintile_analysis(
        combined_wind=combined_wind,
        direction=cdf["target_direction"],
        price_changes=cdf["price_change_eur_mwh"],
    )

    _plot_wind_quintile_charts(wqa)
    _plot_wind_yearly_stability(cdf, wind_cols)

    with st.expander("Wind quintile data"):
        st.dataframe(wqa, use_container_width=True)


def _resolve_wind_columns(cdf: pd.DataFrame) -> list[str]:
    wind_cols: list[str] = []
    for col in ("forecast_wind_offshore_mw_mean", "forecast_wind_onshore_mw_mean"):
        if col in cdf.columns:
            wind_cols.append(col)
    if not wind_cols:
        for col in ("gen_wind_offshore_mw_mean", "gen_wind_onshore_mw_mean"):
            if col in cdf.columns:
                wind_cols.append(col)
    return wind_cols


def _plot_wind_quintile_charts(wqa: pd.DataFrame) -> None:
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            wqa,
            x="wind_bin",
            y="up_rate",
            color="up_rate",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0.5,
            title="Up Rate by Wind Power Quintile",
            labels={"up_rate": "Up Rate", "wind_bin": "Wind Quintile (Q1=Low)"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.bar(
            wqa,
            x="wind_bin",
            y="mean_price_change",
            color="mean_price_change",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Mean Price Change by Wind Quintile",
            labels={"mean_price_change": "EUR/MWh", "wind_bin": "Wind Quintile (Q1=Low)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def _plot_wind_yearly_stability(
    cdf: pd.DataFrame, wind_cols: list[str],
) -> None:
    st.subheader("Wind Quintile Stability by Year")
    year_data = _compute_yearly_wind_quintiles(cdf, wind_cols)

    if not year_data:
        return

    all_years = pd.concat(year_data)
    pivot = all_years.pivot_table(index="wind_bin", columns="year", values="up_rate")
    _render_wind_yearly_heatmap(pivot)


def _compute_yearly_wind_quintiles(
    cdf: pd.DataFrame, wind_cols: list[str],
) -> list[pd.DataFrame]:
    years = sorted(cdf["year"].unique())
    year_data: list[pd.DataFrame] = []
    for yr in years:
        yr_df = cdf[cdf["year"] == yr]
        if len(yr_df) < 30:
            continue
        yr_wind = yr_df[wind_cols].sum(axis=1)
        try:
            yr_wqa = wind_quintile_analysis(
                combined_wind=yr_wind,
                direction=yr_df["target_direction"],
                price_changes=yr_df["price_change_eur_mwh"],
            )
            yr_wqa["year"] = yr
            year_data.append(yr_wqa)
        except Exception:
            continue
    return year_data


def _render_wind_yearly_heatmap(pivot: pd.DataFrame) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=list(pivot.index),
            colorscale="RdYlGn",
            zmid=0.5,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}%",
            colorbar=dict(title="Up Rate"),
        )
    )
    fig.update_layout(
        title="Wind Quintile Up Rate by Year",
        xaxis_title="Year",
        yaxis_title="Wind Quintile",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _section_strategy_correlation_insights(cdf: pd.DataFrame) -> None:
    """Section 24: Composite signal decomposition and strategy insights."""
    st.header("24. Strategy Correlation Insights")
    st.caption(
        "Which features drive direction on ambiguous days (Wed/Thu)? "
        "And how do key signals correlate with each other?"
    )

    feature_cols = _get_signal_feature_cols(cdf)
    if "target_direction" not in cdf.columns or not feature_cols:
        st.info("Required columns not available for correlation analysis.")
        return

    corr_compare = _compute_correlation_comparison(cdf, feature_cols)

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_correlation_comparison(corr_compare)
    with col_r:
        _plot_signal_inter_correlations(cdf)

    with st.expander("Full correlation comparison"):
        st.dataframe(corr_compare.style.format("{:.3f}"), use_container_width=True)


def _get_signal_feature_cols(cdf: pd.DataFrame) -> list[str]:
    exclude = {
        "delivery_date", "split", "year", "settlement_price",
        "price_change_eur_mwh", "target_direction",
        "pnl_long_eur", "pnl_short_eur", "last_settlement_price",
    }
    return [
        c for c in cdf.columns
        if c not in exclude and cdf[c].dtype in ("float64", "int64")
    ]


def _compute_correlation_comparison(
    cdf: pd.DataFrame, feature_cols: list[str],
) -> pd.DataFrame:
    dow = cdf["delivery_date"].dt.dayofweek
    wed_thu = cdf[dow.isin([2, 3])]
    all_corr = cdf[feature_cols].corrwith(cdf["target_direction"]).rename("all_days")
    wt_corr = wed_thu[feature_cols].corrwith(wed_thu["target_direction"]).rename("wed_thu")
    corr_compare = pd.concat([all_corr, wt_corr], axis=1).dropna()
    corr_compare["diff"] = corr_compare["wed_thu"] - corr_compare["all_days"]
    return corr_compare.sort_values("wed_thu", key=abs, ascending=False)


def _plot_correlation_comparison(corr_compare: pd.DataFrame) -> None:
    top_wt = corr_compare.head(10).reset_index()
    top_wt = top_wt.rename(columns={"index": "feature"})
    fig = px.bar(
        top_wt,
        x="feature",
        y=["all_days", "wed_thu"],
        barmode="group",
        title="Top Feature-Direction Correlations: All Days vs Wed/Thu",
        labels={"value": "Correlation", "feature": ""},
    )
    fig.update_layout(height=400, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def _plot_signal_inter_correlations(cdf: pd.DataFrame) -> None:
    key_signals = [
        "load_forecast_mw_mean",
        "forecast_wind_offshore_mw_mean",
        "forecast_wind_onshore_mw_mean",
        "gen_fossil_gas_mw_mean",
        "gen_wind_onshore_mw_mean",
        "forecast_solar_mw_mean",
    ]
    available = [c for c in key_signals if c in cdf.columns]
    if len(available) < 2:
        return
    sig_corr = cdf[available].corr()
    short_names = {c: c.replace("_mw_mean", "").replace("_mean", "") for c in available}
    sig_corr = sig_corr.rename(index=short_names, columns=short_names)

    fig = go.Figure(
        data=go.Heatmap(
            z=sig_corr.values,
            x=list(sig_corr.columns),
            y=list(sig_corr.index),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(sig_corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(title="Key Signal Inter-Correlations", height=400)
    st.plotly_chart(fig, use_container_width=True)
