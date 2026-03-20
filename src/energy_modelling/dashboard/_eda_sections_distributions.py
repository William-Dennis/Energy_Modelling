"""Distribution and negative-price EDA sections."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard._eda_constants import PRICE_COL


def _section_distributions(dff: pd.DataFrame) -> None:
    st.header("Price Distribution & Temporal Patterns")
    _plot_price_histograms(dff)
    _plot_hourly_and_monthly_profiles(dff)
    _plot_dow_profile(dff)


def _plot_price_histograms(dff: pd.DataFrame) -> None:
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.histogram(
            dff,
            x=PRICE_COL,
            nbins=100,
            title="Price Distribution",
            labels={PRICE_COL: "EUR/MWh"},
            marginal="box",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.box(
            dff.reset_index(),
            x="year",
            y=PRICE_COL,
            title="Price by Year",
            labels={"year": "Year", PRICE_COL: "EUR/MWh"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def _plot_hourly_and_monthly_profiles(dff: pd.DataFrame) -> None:
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        _plot_hourly_profile(dff)
    with col_r2:
        _plot_monthly_profile(dff)


def _plot_hourly_profile(dff: pd.DataFrame) -> None:
    hp = dff.groupby("hour")[PRICE_COL].agg(["mean", "median", "std"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hp["hour"], y=hp["mean"], name="Mean", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=hp["hour"], y=hp["median"], name="Median", mode="lines+markers"))
    fig.add_trace(
        go.Scatter(
            x=hp["hour"],
            y=hp["mean"] + hp["std"],
            fill=None,
            mode="lines",
            line={"dash": "dash", "color": "rgba(100,100,100,0.3)"},
            name="+1 Std",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hp["hour"],
            y=hp["mean"] - hp["std"],
            fill="tonexty",
            mode="lines",
            line={"dash": "dash", "color": "rgba(100,100,100,0.3)"},
            name="-1 Std",
        )
    )
    fig.update_layout(
        title="Hourly Price Profile",
        xaxis_title="Hour (UTC)",
        yaxis_title="EUR/MWh",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_monthly_profile(dff: pd.DataFrame) -> None:
    mp = dff.groupby("month")[PRICE_COL].agg(["mean", "median"]).reset_index()
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    mp["month_name"] = mp["month"].map(lambda m: month_names[m - 1])
    fig = px.bar(
        mp,
        x="month_name",
        y=["mean", "median"],
        barmode="group",
        title="Monthly Price Profile",
        labels={"value": "EUR/MWh", "month_name": "Month"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def _plot_dow_profile(dff: pd.DataFrame) -> None:
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dp = dff.groupby("dayofweek")[PRICE_COL].agg(["mean", "median"]).reset_index()
    dp["day_name"] = dp["dayofweek"].map(lambda d: dow_names[d])
    fig = px.bar(
        dp,
        x="day_name",
        y=["mean", "median"],
        barmode="group",
        title="Day-of-Week Price Profile",
        labels={"value": "EUR/MWh", "day_name": ""},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _section_negative(dff: pd.DataFrame) -> None:
    st.header("Negative Price Analysis")
    neg = dff[dff[PRICE_COL] < 0]
    total_hours, neg_hours = len(dff), len(neg)

    c1, c2, c3 = st.columns(3)
    c1.metric("Negative price hours", f"{neg_hours:,}")
    c2.metric("Pct of total", f"{neg_hours / total_hours * 100:.2f}%")
    c3.metric("Most negative", f"{neg[PRICE_COL].min():.1f} EUR/MWh" if neg_hours else "N/A")

    if neg_hours == 0:
        return

    _plot_negative_by_year(neg)
    _plot_negative_details(dff, neg)


def _plot_negative_by_year(neg: pd.DataFrame) -> None:
    npy = neg.groupby("year").size().reset_index(name="count")
    fig = px.bar(
        npy,
        x="year",
        y="count",
        title="Negative Price Hours per Year",
        labels={"year": "Year", "count": "Hours"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def _plot_negative_details(dff: pd.DataFrame, neg: pd.DataFrame) -> None:
    cl, cr = st.columns(2)
    with cl:
        nh = neg.groupby("hour").size().reset_index(name="count")
        fig = px.bar(
            nh,
            x="hour",
            y="count",
            title="Negative Prices by Hour (UTC)",
            labels={"hour": "Hour", "count": "Occurrences"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        st.markdown("**Renewable share comparison:**")
        ren_neg = neg["renewable_share_pct"].mean()
        ren_pos = dff[dff[PRICE_COL] >= 0]["renewable_share_pct"].mean()
        comp = pd.DataFrame(
            {"Condition": ["Negative", "Non-negative"], "Avg Renewable %": [ren_neg, ren_pos]}
        )
        fig = px.bar(
            comp,
            x="Condition",
            y="Avg Renewable %",
            title="Avg Renewable Share: Negative vs Non-Negative",
            color="Condition",
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
