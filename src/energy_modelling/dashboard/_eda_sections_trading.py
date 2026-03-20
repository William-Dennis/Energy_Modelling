"""Trading-focused EDA sections: price changes, autocorrelation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard._eda_constants import PRICE_COL
from energy_modelling.dashboard.eda_analysis import (
    autocorrelation,
    compute_daily_settlement,
    compute_direction_streaks,
    compute_price_changes,
    direction_base_rates,
    direction_by_group,
)


def _section_price_changes(dff: pd.DataFrame) -> None:
    """P1: Daily price change distribution — the actual trading signal."""
    st.header("13. Price Change Distribution")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)
    rates = direction_base_rates(changes)

    _display_price_change_metrics(rates)
    _plot_price_change_charts(changes)
    _plot_direction_by_dow(changes)


def _display_price_change_metrics(rates: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Up days", f"{rates['n_up']} ({rates['pct_up']:.1f}%)")
    c2.metric("Down days", f"{rates['n_down']} ({rates['pct_down']:.1f}%)")
    c3.metric("Mean up move", f"{rates['mean_up_move']:.2f} EUR")
    c4.metric("Mean down move", f"{rates['mean_down_move']:.2f} EUR")
    c5.metric("Median change", f"{rates['median_change']:.2f} EUR")

    c6, c7, c8 = st.columns(3)
    c6.metric("Skewness", f"{rates['skewness']:.3f}")
    c7.metric("Kurtosis", f"{rates['kurtosis']:.3f}")
    c8.metric("Total days", f"{rates['n_total']}")


def _plot_price_change_charts(changes: pd.Series) -> None:
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            changes,
            nbins=80,
            title="Daily Price Change Distribution",
            labels={"value": "EUR/MWh", "count": "Days"},
            marginal="box",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        _plot_direction_by_month(changes)


def _plot_direction_by_month(changes: pd.Series) -> None:
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    months = pd.Series(
        changes.index.month.map(lambda m: month_names[m - 1]),
        index=changes.index,
        name="month",
    )
    monthly_dir = direction_by_group(changes, months)
    present = [m for m in month_names if m in monthly_dir.index]
    monthly_dir = monthly_dir.reindex(present)
    fig = px.bar(
        monthly_dir.reset_index(),
        x="group" if "group" in monthly_dir.reset_index().columns else monthly_dir.index.name,
        y="pct_up",
        title="% Up Days by Month",
        labels={"pct_up": "% Up", "month": ""},
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _plot_direction_by_dow(changes: pd.Series) -> None:
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = pd.Series(
        changes.index.dayofweek.map(lambda d: dow_names[d]),
        index=changes.index,
        name="day",
    )
    dow_dir = direction_by_group(changes, dow)
    present_dow = [d for d in dow_names if d in dow_dir.index]
    dow_dir = dow_dir.reindex(present_dow)
    fig = px.bar(
        dow_dir.reset_index(),
        x="group" if "group" in dow_dir.reset_index().columns else dow_dir.index.name,
        y="pct_up",
        title="% Up Days by Day of Week",
        labels={"pct_up": "% Up", "day": ""},
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_autocorrelation(dff: pd.DataFrame) -> None:
    """P2: Autocorrelation and direction persistence of price changes."""
    st.header("14. Autocorrelation & Direction Persistence")

    settlements = compute_daily_settlement(dff[PRICE_COL])
    changes = compute_price_changes(settlements)

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_acf(changes)
    with col_r:
        _display_streaks_and_transitions(changes)


def _plot_acf(changes: pd.Series) -> None:
    max_lag = min(40, len(changes) // 3)
    if max_lag < 1:
        st.info("Not enough data for autocorrelation analysis.")
        return
    acf = autocorrelation(changes, max_lag=max_lag)
    n = len(changes)
    sig = 1.96 / np.sqrt(n)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=acf.index, y=acf.values, name="ACF"))
    fig.add_hline(y=sig, line_dash="dash", line_color="red", annotation_text="95% CI")
    fig.add_hline(y=-sig, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    fig.update_layout(
        title="Autocorrelation of Daily Price Changes",
        xaxis_title="Lag (days)",
        yaxis_title="ACF",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_streaks_and_transitions(changes: pd.Series) -> None:
    streaks = compute_direction_streaks(changes)
    st.markdown("**Direction Streaks**")
    s1, s2 = st.columns(2)
    s1.metric("Max up streak", f"{streaks['max_up_streak']} days")
    s2.metric("Max down streak", f"{streaks['max_down_streak']} days")
    s3, s4 = st.columns(2)
    s3.metric("Mean up streak", f"{streaks['mean_up_streak']:.1f} days")
    s4.metric("Mean down streak", f"{streaks['mean_down_streak']:.1f} days")

    directions = np.sign(changes.values)
    transitions = pd.DataFrame(
        index=["After Up", "After Down"],
        columns=["Next Up", "Next Down"],
        dtype=float,
    )
    for prev, label in [(1, "After Up"), (-1, "After Down")]:
        mask = directions[:-1] == prev
        next_dirs = directions[1:][mask]
        if len(next_dirs) > 0:
            transitions.loc[label, "Next Up"] = (next_dirs > 0).mean() * 100
            transitions.loc[label, "Next Down"] = (next_dirs < 0).mean() * 100
        else:
            transitions.loc[label] = [np.nan, np.nan]
    st.markdown("**Transition Probabilities (%)**")
    st.dataframe(transitions.style.format("{:.1f}"), use_container_width=True)
