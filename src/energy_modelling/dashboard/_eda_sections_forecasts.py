"""Forecast error and feature importance EDA sections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard._eda_constants import (
    GEN_COLS_DISPLAY,
    PRICE_COL,
    WEATHER_COLS_DISPLAY,
)
from energy_modelling.dashboard.eda_analysis import (
    compute_daily_settlement,
    compute_forecast_errors,
    compute_price_changes,
    lagged_direction_correlation,
)

_FORECAST_PAIRS = [
    ("load_actual_mw", "load_forecast_mw", "Load"),
    ("gen_solar_mw", "forecast_solar_mw", "Solar"),
    ("gen_wind_onshore_mw", "forecast_wind_onshore_mw", "Wind Onshore"),
    ("gen_wind_offshore_mw", "forecast_wind_offshore_mw", "Wind Offshore"),
]


def _section_forecast_errors(dff: pd.DataFrame) -> None:
    """P3: Forecast error analysis for load, wind, solar."""
    st.header("15. Forecast Error Analysis")

    available_pairs = [
        (act, fc, name)
        for act, fc, name in _FORECAST_PAIRS
        if act in dff.columns and fc in dff.columns
    ]
    if not available_pairs:
        st.info("No forecast/actual column pairs found in the data.")
        return

    selected = st.selectbox(
        "Select forecast type",
        [name for _, _, name in available_pairs],
        key="eda_forecast_error_select",
    )
    act_col, fc_col, _ = next((a, f, n) for a, f, n in available_pairs if n == selected)
    errors = compute_forecast_errors(dff[act_col], dff[fc_col])

    _display_forecast_metrics(errors)
    _plot_forecast_error_charts(errors, selected)
    _plot_actual_vs_forecast(dff, act_col, fc_col, selected)


def _display_forecast_metrics(errors: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Error", f"{errors['error'].mean():.1f} MW")
    c2.metric("MAE", f"{errors['abs_error'].mean():.1f} MW")
    c3.metric("RMSE", f"{np.sqrt((errors['error'] ** 2).mean()):.1f} MW")
    c4.metric("Mean Bias", "Over" if errors["error"].mean() > 0 else "Under")


def _plot_forecast_error_charts(
    errors: pd.DataFrame, selected: str,
) -> None:
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            errors,
            x="error",
            nbins=80,
            title=f"{selected}: Forecast Error Distribution (forecast - actual)",
            labels={"error": "MW"},
            marginal="box",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        _plot_forecast_error_by_month(errors, selected)


def _plot_forecast_error_by_month(
    errors: pd.DataFrame, selected: str,
) -> None:
    monthly_error = errors.copy()
    monthly_error["month"] = monthly_error.index.month
    monthly_stats = monthly_error.groupby("month")["error"].agg(["mean", "std"]).reset_index()
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    monthly_stats["month_name"] = monthly_stats["month"].map(lambda m: month_names[m - 1])
    fig = px.bar(
        monthly_stats,
        x="month_name",
        y="mean",
        error_y="std",
        title=f"{selected}: Mean Forecast Error by Month",
        labels={"mean": "Mean Error (MW)", "month_name": ""},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _plot_actual_vs_forecast(
    dff: pd.DataFrame, act_col: str, fc_col: str, selected: str,
) -> None:
    sample = dff[[act_col, fc_col]].dropna()
    if len(sample) > 10_000:
        sample = sample.sample(10_000, random_state=42)
    fig = px.scatter(
        sample,
        x=act_col,
        y=fc_col,
        opacity=0.3,
        title=f"{selected}: Actual vs Forecast",
        labels={act_col: "Actual (MW)", fc_col: "Forecast (MW)"},
    )
    min_val = min(sample[act_col].min(), sample[fc_col].min())
    max_val = max(sample[act_col].max(), sample[fc_col].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Perfect",
        )
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def _section_feature_importance(dff: pd.DataFrame) -> None:
    """P4: Lagged feature importance for price direction."""
    st.header("16. Feature Importance for Price Direction")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)
    direction = np.sign(changes)
    direction.name = "direction"

    feature_cols = _build_feature_col_list(dff)
    if not feature_cols:
        st.info("No feature columns available for importance analysis.")
        return

    result = _compute_feature_importance(dff, feature_cols, direction)
    if result is None:
        return
    corr = result
    _plot_feature_importance(corr)


def _build_feature_col_list(dff: pd.DataFrame) -> list[str]:
    feature_cols = (
        [c for c in GEN_COLS_DISPLAY if c in dff.columns]
        + [c for c in WEATHER_COLS_DISPLAY if c in dff.columns]
        + [c for c in ["load_actual_mw", "carbon_price_usd", "gas_price_usd"] if c in dff.columns]
        + [
            c
            for c in dff.columns
            if c.startswith("price_") and c.endswith("_eur_mwh") and c != PRICE_COL
        ]
        + [c for c in dff.columns if c.startswith("flow_")]
    )
    return list(dict.fromkeys(feature_cols))


def _compute_feature_importance(
    dff: pd.DataFrame,
    feature_cols: list[str],
    direction: pd.Series,
) -> pd.Series | None:
    daily_features = dff[feature_cols].groupby(dff.index.date).mean()
    daily_features.index = pd.DatetimeIndex(daily_features.index)
    daily_features_lagged = daily_features.shift(1)

    rename_map = {**GEN_COLS_DISPLAY, **WEATHER_COLS_DISPLAY}
    rename_map.update(
        {"load_actual_mw": "Load Actual", "carbon_price_usd": "Carbon Price",
         "gas_price_usd": "Gas Price"}
    )
    for c in daily_features_lagged.columns:
        if c.startswith("price_") and c.endswith("_eur_mwh"):
            zone = c.replace("price_", "").replace("_eur_mwh", "").upper()
            rename_map[c] = f"Price {zone}"
        if c.startswith("flow_"):
            border = c.replace("flow_", "").replace("_net_import_mw", "").upper()
            rename_map[c] = f"Flow {border}"
    daily_features_lagged = daily_features_lagged.rename(columns=rename_map)

    common_idx = daily_features_lagged.index.intersection(direction.index)
    if len(common_idx) < 30:
        st.info("Not enough data points for feature importance analysis.")
        return None

    features_aligned = daily_features_lagged.loc[common_idx].dropna(axis=1, how="all")
    direction_aligned = direction.loc[common_idx]
    valid = features_aligned.dropna()
    direction_valid = direction_aligned.loc[valid.index]
    return lagged_direction_correlation(valid, direction_valid)


def _plot_feature_importance(corr: pd.Series) -> None:
    fig = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        labels={"x": "Correlation with Next-Day Direction", "y": ""},
        title="D-1 Feature Correlation with D Price Direction (lagged)",
        color=corr.values,
        color_continuous_scale="RdBu_r",
        range_color=[-0.3, 0.3],
    )
    fig.update_layout(height=max(500, len(corr) * 22), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Top predictive features"):
        top_n = min(10, len(corr))
        top = corr.head(top_n)
        st.dataframe(
            pd.DataFrame({"Feature": top.index, "Correlation": top.values}).style.format(
                {"Correlation": "{:.4f}"}
            ),
            use_container_width=True,
        )
