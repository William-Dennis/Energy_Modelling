"""Phase 6 strategy feedback EDA sections: day-of-week, drift, quarterly, regime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard._eda_constants import _DOW_NAMES
from energy_modelling.dashboard.eda_analysis import (
    day_of_week_edge_by_year,
    feature_drift,
    quarterly_direction_rates,
    volatility_regime_performance,
)


def _section_dow_edge_stability(cdf: pd.DataFrame) -> None:
    """Section 19: Day-of-week edge stability across years."""
    st.header("19. Day-of-Week Edge Stability")
    st.caption(
        "Is the Monday-long/Saturday-short edge stable, or is it decaying? "
        "Each cell shows the directional edge (up_rate - baseline) for a day-year pair."
    )

    result = day_of_week_edge_by_year(
        price_changes=cdf["price_change_eur_mwh"],
        dates=cdf["delivery_date"],
    )
    result["dow_name"] = result["dow"].map(_DOW_NAMES)
    _plot_dow_heatmap(result)

    with st.expander("Raw edge data"):
        display = result[["year", "dow_name", "up_rate", "overall_up_rate", "edge"]].copy()
        display["edge_pct"] = (display["edge"] * 100).round(1)
        display["up_rate_pct"] = (display["up_rate"] * 100).round(1)
        st.dataframe(display, use_container_width=True)


def _plot_dow_heatmap(result: pd.DataFrame) -> None:
    pivot = result.pivot_table(index="dow_name", columns="year", values="edge", aggfunc="first")
    row_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex([d for d in row_order if d in pivot.index])

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=list(pivot.index),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}%",
            colorbar=dict(title="Edge"),
        )
    )
    fig.update_layout(
        title="Day-of-Week Directional Edge by Year",
        xaxis_title="Year",
        yaxis_title="Day of Week",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _section_feature_drift(cdf: pd.DataFrame) -> None:
    """Section 20: Feature distribution drift between train and validation."""
    st.header("20. Feature Drift (Train vs Validation)")
    st.caption(
        "How much do feature distributions shift between training (2019-2023) and "
        "validation (2024)? Large shifts can break threshold-based strategies."
    )

    train = cdf[cdf["split"] == "train"] if "split" in cdf.columns else cdf[cdf["year"] <= 2023]
    val = cdf[cdf["split"] == "validation"] if "split" in cdf.columns else cdf[cdf["year"] == 2024]

    if train.empty or val.empty:
        st.info("Need both train and validation data for drift analysis.")
        return

    exclude = {
        "delivery_date", "split", "year", "settlement_price",
        "price_change_eur_mwh", "target_direction",
        "pnl_long_eur", "pnl_short_eur", "last_settlement_price",
    }
    feat_cols = [
        c for c in train.columns if c not in exclude and train[c].dtype in ("float64", "int64")
    ]
    drift_df = feature_drift(train[feat_cols], val[feat_cols])
    drift_sorted = drift_df.sort_values("shift_pct", key=abs, ascending=False)
    _plot_drift_chart(drift_sorted)

    with st.expander("Full drift table"):
        styled = drift_sorted.style.format(
            {
                "train_mean": "{:.2f}",
                "val_mean": "{:.2f}",
                "shift_pct": "{:+.1f}%",
                "train_std": "{:.2f}",
                "val_std": "{:.2f}",
                "std_ratio": "{:.2f}",
            }
        )
        st.dataframe(styled, use_container_width=True)


def _plot_drift_chart(drift_sorted: pd.DataFrame) -> None:
    top_n = min(15, len(drift_sorted))
    top = drift_sorted.head(top_n).reset_index()
    fig = px.bar(
        top,
        x="feature",
        y="shift_pct",
        color="shift_pct",
        color_continuous_scale="RdYlBu_r",
        color_continuous_midpoint=0,
        title=f"Top {top_n} Feature Shifts (Train -> Validation, % change in mean)",
        labels={"shift_pct": "Shift %", "feature": ""},
    )
    fig.update_layout(height=450, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def _section_quarterly_patterns(cdf: pd.DataFrame) -> None:
    """Section 21: Quarterly direction rates and volatility."""
    st.header("21. Quarterly Direction Rates")
    st.caption(
        "Is Q4 consistently best, or is the 2024 Q4 outperformance a volatility artefact? "
        "Shows up-rate and mean |price change| by year-quarter."
    )

    qdr = quarterly_direction_rates(
        price_changes=cdf["price_change_eur_mwh"],
        dates=cdf["delivery_date"],
    )

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_quarterly_up_rate(qdr)
    with col_r:
        _plot_quarterly_volatility(qdr)

    with st.expander("Quarterly data table"):
        display = qdr.copy()
        display["up_rate_pct"] = (display["up_rate"] * 100).round(1)
        display["mean_abs_change"] = display["mean_abs_change"].round(2)
        st.dataframe(display, use_container_width=True)


def _plot_quarterly_up_rate(qdr: pd.DataFrame) -> None:
    pivot_up = qdr.pivot_table(index="year", columns="quarter", values="up_rate")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_up.values,
            x=[f"Q{c}" for c in pivot_up.columns],
            y=[str(r) for r in pivot_up.index],
            colorscale="RdYlGn",
            zmid=0.5,
            text=np.round(pivot_up.values * 100, 1),
            texttemplate="%{text}%",
            colorbar=dict(title="Up Rate"),
        )
    )
    fig.update_layout(title="Up Rate by Year-Quarter", height=350)
    st.plotly_chart(fig, use_container_width=True)


def _plot_quarterly_volatility(qdr: pd.DataFrame) -> None:
    pivot_vol = qdr.pivot_table(index="year", columns="quarter", values="mean_abs_change")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_vol.values,
            x=[f"Q{c}" for c in pivot_vol.columns],
            y=[str(r) for r in pivot_vol.index],
            colorscale="YlOrRd",
            text=np.round(pivot_vol.values, 1),
            texttemplate="%{text}",
            colorbar=dict(title="EUR/MWh"),
        )
    )
    fig.update_layout(title="Mean |Price Change| by Year-Quarter", height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_volatility_regime_performance(cdf: pd.DataFrame) -> None:
    """Section 22: Direction stats conditioned on volatility regime."""
    st.header("22. Volatility Regime Performance")
    st.caption(
        "Do high-volatility regimes favour longs or shorts? "
        "Shows direction stats within low/mid/high 30-day rolling-volatility regimes."
    )

    vrp = volatility_regime_performance(
        price_changes=cdf["price_change_eur_mwh"],
        window=30,
        n_regimes=3,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_regime_up_rate(vrp)
    with col_r:
        _plot_regime_abs_change(vrp)

    st.dataframe(
        vrp.style.format(
            {"up_rate": "{:.1%}", "mean_abs_change": "{:.2f}", "mean_change": "{:+.2f}"}
        ),
        use_container_width=True,
    )


def _plot_regime_up_rate(vrp: pd.DataFrame) -> None:
    fig = px.bar(
        vrp,
        x="regime",
        y="up_rate",
        color="regime",
        color_discrete_map={"low": "#2ecc71", "mid": "#f1c40f", "high": "#e74c3c"},
        title="Up Rate by Volatility Regime",
        labels={"up_rate": "Up Rate", "regime": "Regime"},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _plot_regime_abs_change(vrp: pd.DataFrame) -> None:
    fig = px.bar(
        vrp,
        x="regime",
        y="mean_abs_change",
        color="regime",
        color_discrete_map={"low": "#2ecc71", "mid": "#f1c40f", "high": "#e74c3c"},
        title="Mean |Price Change| by Regime",
        labels={"mean_abs_change": "EUR/MWh", "regime": "Regime"},
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
