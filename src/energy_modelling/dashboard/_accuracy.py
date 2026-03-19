"""Tab 5 -- Futures Market Simulation Accuracy.

Compares the converged synthetic-futures market price against the real
day-ahead settlement price to assess whether this pool of strategies
collectively produces realistic price forecasts.

Expects that the Futures Market tab has already been run (results stored in
``st.session_state``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.backtest.futures_market_runner import FuturesMarketResult


def _build_comparison(
    market_result: FuturesMarketResult,
    daily_data: pd.DataFrame,
    eval_start: str,
    eval_end: str,
) -> pd.DataFrame:
    """Align market prices, settlement prices, and yesterday-settlement
    into a single DataFrame for comparison."""

    data = daily_data.copy()
    data["delivery_date"] = pd.to_datetime(data["delivery_date"]).dt.date
    data = data.set_index("delivery_date", drop=False)
    mask = (data.index >= pd.Timestamp(eval_start).date()) & (
        data.index <= pd.Timestamp(eval_end).date()
    )
    data = data.loc[mask]

    market_prices = market_result.equilibrium.final_market_prices
    settlement = data["settlement_price"].astype(float)
    yesterday = data["last_settlement_price"].astype(float)

    df = pd.DataFrame(
        {
            "Date": settlement.index,
            "Settlement (Real)": settlement.values,
            "Yesterday Settlement": yesterday.values,
        }
    ).set_index("Date")

    mp = market_prices.rename("Market Price")
    df = df.join(mp, how="left")
    df = df.dropna()
    df = df.reset_index()

    # Error columns
    df["Market Error"] = df["Market Price"] - df["Settlement (Real)"]
    df["Yesterday Error"] = df["Yesterday Settlement"] - df["Settlement (Real)"]
    return df


def _error_stats(errors: pd.Series, label: str) -> dict[str, float]:
    return {
        "Model": label,
        "MAE (EUR/MWh)": float(errors.abs().mean()),
        "RMSE (EUR/MWh)": float(np.sqrt((errors**2).mean())),
        "Mean Error": float(errors.mean()),
        "Std Error": float(errors.std()),
        "Max Abs Error": float(errors.abs().max()),
    }


def _render_period(
    period: str,
    mr: FuturesMarketResult | None,
    daily: pd.DataFrame,
    eval_start: str,
    eval_end: str,
) -> None:
    if mr is None:
        st.info(f"No market results for {period}. Run the Market tab first.")
        return

    comp = _build_comparison(mr, daily, eval_start, eval_end)
    if comp.empty:
        st.warning(f"No overlapping data for {period}.")
        return

    # --- Error statistics ---------------------------------------------------
    st.subheader(f"Forecast Accuracy Summary - {period}")
    stats = pd.DataFrame(
        [
            _error_stats(comp["Market Error"], "Converged Market Price"),
            _error_stats(comp["Yesterday Error"], "Yesterday Settlement"),
        ]
    )
    st.dataframe(stats.set_index("Model"), use_container_width=True)

    # Headline metrics
    mkt_mae = comp["Market Error"].abs().mean()
    yst_mae = comp["Yesterday Error"].abs().mean()
    improvement = (yst_mae - mkt_mae) / yst_mae * 100 if yst_mae > 0 else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Market MAE", f"{mkt_mae:.2f} EUR/MWh")
    c2.metric("Yesterday MAE", f"{yst_mae:.2f} EUR/MWh")
    c3.metric(
        "MAE Improvement",
        f"{improvement:+.1f}%",
        delta=f"{improvement:+.1f}%",
        delta_color="normal" if improvement > 0 else "inverse",
    )

    # --- Price overlay chart ------------------------------------------------
    st.subheader(f"Price Comparison - {period}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comp["Date"],
            y=comp["Settlement (Real)"],
            name="Settlement (Real)",
            mode="lines",
            line={"color": "black", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comp["Date"],
            y=comp["Market Price"],
            name="Converged Market Price",
            mode="lines",
            line={"color": "#1E90FF", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comp["Date"],
            y=comp["Yesterday Settlement"],
            name="Yesterday Settlement",
            mode="lines",
            line={"color": "#FF6347", "width": 1, "dash": "dot"},
        )
    )
    fig.update_layout(
        title=f"Settlement vs Market vs Yesterday - {period}",
        yaxis_title="EUR/MWh",
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Error time series --------------------------------------------------
    st.subheader(f"Forecast Error Over Time - {period}")
    err_df = pd.DataFrame(
        {
            "Date": comp["Date"],
            "Market Error": comp["Market Error"],
            "Yesterday Error": comp["Yesterday Error"],
        }
    )
    fig = px.line(
        err_df.melt(id_vars="Date", var_name="Model", value_name="Error"),
        x="Date",
        y="Error",
        color="Model",
        title="Daily Forecast Error (predicted - actual)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Error distributions ------------------------------------------------
    st.subheader(f"Error Distribution - {period}")
    cl, cr = st.columns(2)
    with cl:
        fig = px.histogram(
            comp,
            x="Market Error",
            nbins=50,
            title="Market Price Error Distribution",
            labels={"Market Error": "Error (EUR/MWh)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig = px.histogram(
            comp,
            x="Yesterday Error",
            nbins=50,
            title="Yesterday-Settlement Error Distribution",
            labels={"Yesterday Error": "Error (EUR/MWh)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # --- Scatter: predicted vs actual ---------------------------------------
    st.subheader(f"Predicted vs Actual - {period}")
    cl2, cr2 = st.columns(2)
    price_min = comp["Settlement (Real)"].min() - 10
    price_max = comp["Settlement (Real)"].max() + 10
    with cl2:
        fig = px.scatter(
            comp,
            x="Settlement (Real)",
            y="Market Price",
            title="Market Price vs Actual Settlement",
            labels={"Settlement (Real)": "Actual (EUR/MWh)", "Market Price": "Market (EUR/MWh)"},
            opacity=0.5,
        )
        fig.add_shape(
            type="line",
            x0=price_min,
            y0=price_min,
            x1=price_max,
            y1=price_max,
            line={"dash": "dash", "color": "gray"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with cr2:
        fig = px.scatter(
            comp,
            x="Settlement (Real)",
            y="Yesterday Settlement",
            title="Yesterday Settlement vs Actual",
            labels={
                "Settlement (Real)": "Actual (EUR/MWh)",
                "Yesterday Settlement": "Yesterday (EUR/MWh)",
            },
            opacity=0.5,
        )
        fig.add_shape(
            type="line",
            x0=price_min,
            y0=price_min,
            x1=price_max,
            y1=price_max,
            line={"dash": "dash", "color": "gray"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Rolling MAE --------------------------------------------------------
    st.subheader(f"Rolling 30-Day MAE - {period}")
    rolling_mkt = comp["Market Error"].abs().rolling(30).mean()
    rolling_yst = comp["Yesterday Error"].abs().rolling(30).mean()
    roll_df = pd.DataFrame(
        {
            "Date": comp["Date"],
            "Market (30d MAE)": rolling_mkt,
            "Yesterday (30d MAE)": rolling_yst,
        }
    ).dropna()
    if not roll_df.empty:
        fig = px.line(
            roll_df.melt(id_vars="Date", var_name="Model", value_name="MAE"),
            x="Date",
            y="MAE",
            color="Model",
            title="30-Day Rolling MAE",
        )
        fig.update_layout(height=400, yaxis_title="MAE (EUR/MWh)")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the futures-market-simulation-accuracy tab."""

    m24 = st.session_state.get("market_2024")
    m25 = st.session_state.get("market_2025")
    public_daily = st.session_state.get("backtest_public_daily")
    hidden_daily = st.session_state.get("backtest_hidden_daily")

    if m24 is None and m25 is None:
        st.info(
            "Run the **Backtest** tab then the **Futures Market** tab first "
            "so converged market prices are available for comparison."
        )
        return

    st.markdown(
        "Compares the **converged synthetic-futures price** (strategy-consensus) "
        "against the **real day-ahead settlement** to assess whether this pool of "
        "strategies produces realistic price forecasts. The **yesterday-settlement** "
        "baseline (naive forecast) is shown for reference."
    )

    tabs = st.tabs(["2024", "2025"])
    with tabs[0]:
        _render_period("2024", m24, public_daily, "2024-01-01", "2024-12-31")
    with tabs[1]:
        if hidden_daily is None:
            st.info("Hidden 2025 data not available.")
        else:
            from energy_modelling.dashboard._backtest import combine_public_hidden

            combined = combine_public_hidden(public_daily, hidden_daily)
            _render_period("2025", m25, combined, "2025-01-01", "2025-12-31")
