"""Tab 4 -- Synthetic Futures Market.

Runs the synthetic-market model over all backtest strategies and shows
market-adjusted leaderboards, rank changes, convergence diagnostics,
weight evolution, and cumulative market-adjusted PnL.

Expects that the Backtest tab has already been run (results stored in
``st.session_state``).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.backtest.futures_market_runner import (
    FuturesMarketResult,
    run_futures_market_evaluation,
)
from energy_modelling.backtest.runner import BacktestResult
from energy_modelling.backtest.scoring import market_leaderboard_score
from energy_modelling.dashboard._backtest import (
    STRATEGY_FACTORIES,
    combine_public_hidden,
)


# ---------------------------------------------------------------------------
# Market helpers
# ---------------------------------------------------------------------------


def _run_market(
    selected: list[str],
    daily: pd.DataFrame,
    training_end: date,
    eval_start: date,
    eval_end: date,
) -> FuturesMarketResult | None:
    factories = {n: STRATEGY_FACTORIES[n] for n in selected}
    try:
        return run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=daily,
            training_end=training_end,
            evaluation_start=eval_start,
            evaluation_end=eval_end,
        )
    except Exception:  # noqa: BLE001
        return None


def _market_lb(market_result: FuturesMarketResult) -> pd.DataFrame:
    rows = []
    for name, r in market_result.market_results.items():
        m = r.metrics
        orig = market_result.original_results[name].metrics
        rows.append(
            {
                "Strategy": name,
                "Market PnL": m["total_pnl"],
                "Original PnL": orig["total_pnl"],
                "Sharpe": m["sharpe_ratio"],
                "Max Drawdown": m["max_drawdown"],
                "Trades": m["trade_count"],
                "Win Rate": m["win_rate"],
                "Score": market_leaderboard_score(m),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["Market PnL", "Sharpe", "Max Drawdown"], ascending=[False, False, True]
    )


def _fmt_market_lb(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.drop(columns=["Score"])
        .rename(
            columns={
                "Market PnL": "Market PnL (EUR)",
                "Original PnL": "Original PnL (EUR)",
                "Max Drawdown": "Max Drawdown (EUR)",
            }
        )
        .assign(
            **{
                "Sharpe": lambda df: df["Sharpe"].map(lambda v: f"{v:.2f}"),
                "Trades": lambda df: df["Trades"].map(lambda v: f"{v:.0f}"),
                "Win Rate": lambda df: df["Win Rate"].map(lambda v: f"{v:.1%}"),
            }
        )
    )


# ---------------------------------------------------------------------------
# Market section for one period
# ---------------------------------------------------------------------------

# Outer band (p10–p90): subtle fill, no border lines
_OUTER_COLOR = "rgba(99,110,250,0.12)"
_OUTER_BORDER = "rgba(99,110,250,0.35)"
# Inner band (p25–p75): stronger fill
_INNER_COLOR = "rgba(99,110,250,0.25)"
_INNER_BORDER = "rgba(99,110,250,0.55)"
# Median line
_MEDIAN_COLOR = "#636EFA"
# Real-price reference lines
_REAL_COLOR = "rgba(255,255,255,0.45)"
_REAL_LABEL_COLOR = "rgba(255,255,255,0.6)"


def _render_price_quantile_evolution(
    eq_iterations: list,
    daily_data: pd.DataFrame,
    eval_start: date,
    eval_end: date,
) -> None:
    """Chart: synthetic price quantile fan across iterations vs real market."""
    if len(eq_iterations) < 2:
        return

    # Per-iteration quantile arrays
    iters = [it.iteration for it in eq_iterations]
    qs = {q: [] for q in [0.10, 0.25, 0.50, 0.75, 0.90]}
    for it in eq_iterations:
        vals = it.market_prices.values
        for q in qs:
            qs[q].append(float(np.quantile(vals, q)))

    p10, p25, p50, p75, p90 = qs[0.10], qs[0.25], qs[0.50], qs[0.75], qs[0.90]

    # Real settlement quantiles
    data = daily_data.copy()
    data["delivery_date"] = pd.to_datetime(data["delivery_date"]).dt.date
    data = data.set_index("delivery_date")
    mask = (data.index >= eval_start) & (data.index <= eval_end)
    real_vals = data.loc[mask, "settlement_price"].astype(float).values
    rq = {q: float(np.quantile(real_vals, q)) for q in [0.10, 0.25, 0.50, 0.75, 0.90]}

    st.subheader("Synthetic Price Distribution vs Iterations")

    fig = go.Figure()

    # --- Outer band: p10–p90 (fill between) ---
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=p90,
            mode="lines",
            line={"width": 0, "color": _OUTER_BORDER},
            name="p90",
            showlegend=False,
            hovertemplate="Iter %{x} — p90: %{y:.1f} EUR/MWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=p10,
            mode="lines",
            line={"width": 0, "color": _OUTER_BORDER},
            fill="tonexty",
            fillcolor=_OUTER_COLOR,
            name="p10–p90 range",
            showlegend=True,
            hovertemplate="Iter %{x} — p10: %{y:.1f} EUR/MWh<extra></extra>",
        )
    )

    # --- Inner band: p25–p75 ---
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=p75,
            mode="lines",
            line={"width": 0, "color": _INNER_BORDER},
            name="p75",
            showlegend=False,
            hovertemplate="Iter %{x} — p75: %{y:.1f} EUR/MWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=p25,
            mode="lines",
            line={"width": 0, "color": _INNER_BORDER},
            fill="tonexty",
            fillcolor=_INNER_COLOR,
            name="p25–p75 range",
            showlegend=True,
            hovertemplate="Iter %{x} — p25: %{y:.1f} EUR/MWh<extra></extra>",
        )
    )

    # --- Median line ---
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=p50,
            mode="lines+markers",
            line={"color": _MEDIAN_COLOR, "width": 2.5},
            marker={"size": 6, "color": _MEDIAN_COLOR},
            name="p50 (median)",
            hovertemplate="Iter %{x} — median: %{y:.1f} EUR/MWh<extra></extra>",
        )
    )

    # --- Real settlement reference lines ---
    x_min, x_max = iters[0], iters[-1]
    ref_labels = {0.10: "p10", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"}
    for i, (q, label) in enumerate(ref_labels.items()):
        show = i == 0  # single legend entry for all real lines
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[rq[q], rq[q]],
                mode="lines",
                line={"color": _REAL_COLOR, "dash": "dot", "width": 1},
                name="Real settlement" if show else None,
                showlegend=show,
                hovertemplate=f"Real {label}: {rq[q]:.1f} EUR/MWh<extra></extra>",
            )
        )
        fig.add_annotation(
            x=x_max,
            y=rq[q],
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font={"size": 10, "color": _REAL_LABEL_COLOR},
        )

    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Price (EUR/MWh)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        xaxis={"dtick": 1, "gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.08)"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"r": 60},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_market_period(
    period: str, mr: FuturesMarketResult | None, daily_data: pd.DataFrame | None = None
) -> None:
    if mr is None:
        st.info(f"Market simulation not available for {period}.")
        return

    eq = mr.equilibrium

    # Convergence summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Converged", "Yes" if eq.converged else "No")
    c2.metric("Iterations", str(len(eq.iterations)))
    c3.metric("Final Delta", f"EUR {eq.convergence_delta:.4f}")

    # Market leaderboard
    st.subheader(f"Market-Adjusted Leaderboard - {period}")
    lb = _market_lb(mr)
    if not lb.empty:
        st.dataframe(_fmt_market_lb(lb), use_container_width=True)

    # Rank changes
    st.subheader("Rank Change: Original vs Market")
    orig_order = lb.sort_values("Original PnL", ascending=False)["Strategy"].reset_index(drop=True)
    mkt_order = lb["Strategy"].reset_index(drop=True)
    rank_rows = []
    for s in lb["Strategy"]:
        o = int(orig_order[orig_order == s].index[0]) + 1
        m = int(mkt_order[mkt_order == s].index[0]) + 1
        rank_rows.append({"Strategy": s, "Original Rank": o, "Market Rank": m, "Change": o - m})
    st.dataframe(pd.DataFrame(rank_rows), use_container_width=True)

    # Convergence plot
    if len(eq.iterations) > 1:
        st.subheader("Convergence")
        deltas = [{"Iteration": 0, "Max Delta": 0.0}]
        for i in range(1, len(eq.iterations)):
            d = float(
                (eq.iterations[i].market_prices - eq.iterations[i - 1].market_prices).abs().max()
            )
            deltas.append({"Iteration": i, "Max Delta": d})
        fig = px.line(
            pd.DataFrame(deltas),
            x="Iteration",
            y="Max Delta",
            title="Price Convergence (max |delta| per iteration)",
        )
        fig.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)

    # Price quantile evolution vs real settlement
    if daily_data is not None and len(eq.iterations) > 1:
        eval_start = eq.iterations[0].market_prices.index.min()
        eval_end = eq.iterations[0].market_prices.index.max()
        _render_price_quantile_evolution(eq.iterations, daily_data, eval_start, eval_end)

    # Weight evolution
    st.subheader("Strategy Weights Across Iterations")
    wrows = []
    for it in eq.iterations:
        for name, w in it.strategy_weights.items():
            wrows.append({"Iteration": it.iteration, "Strategy": name, "Weight": w})
    if wrows:
        fig = px.area(
            pd.DataFrame(wrows),
            x="Iteration",
            y="Weight",
            color="Strategy",
            title="Strategy Weight Evolution",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Market price line
    st.subheader("Converged Market Price")
    mp = eq.final_market_prices.reset_index()
    mp.columns = ["Date", "Market Price"]
    fig = px.line(mp, x="Date", y="Market Price", title="Converged Market Price")
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative PnL
    st.subheader("Cumulative PnL (Market-Adjusted)")
    cum_rows = []
    for name, r in mr.market_results.items():
        for d, v in r.cumulative_pnl.items():
            cum_rows.append({"Date": d, "Strategy": name, "Cumulative PnL": v})
    if cum_rows:
        fig = px.line(
            pd.DataFrame(cum_rows),
            x="Date",
            y="Cumulative PnL",
            color="Strategy",
            title=f"Cumulative Market-Adjusted PnL - {period}",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the synthetic futures market tab."""

    val_results = st.session_state.get("backtest_val_results")
    if val_results is None:
        st.info(
            "Run the **Backtest** tab first so that strategy predictions "
            "are available for the market simulation."
        )
        return

    selected = st.session_state.get("backtest_selected", [])
    public_daily = st.session_state.get("backtest_public_daily")
    hidden_daily = st.session_state.get("backtest_hidden_daily")

    if len(selected) < 2:
        st.warning("The synthetic market needs at least 2 strategies.")
        return

    run_btn = st.button("Run Synthetic Market", type="primary", key="mkt_run")
    if not run_btn and "market_2024" not in st.session_state:
        from energy_modelling.backtest.io import RESULTS_DIR, load_market_results

        m24 = load_market_results(RESULTS_DIR / "market_2024.pkl")
        m25 = load_market_results(RESULTS_DIR / "market_2025.pkl")
        if m24 is not None:
            st.session_state["market_2024"] = m24
        if m25 is not None:
            st.session_state["market_2025"] = m25

    if not run_btn and "market_2024" not in st.session_state:
        st.info("Click **Run Synthetic Market** to compute equilibrium prices.")
        return

    if run_btn:
        combined = combine_public_hidden(public_daily, hidden_daily)
        with st.spinner("Running synthetic futures market..."):
            m24 = _run_market(
                selected, public_daily, date(2023, 12, 31), date(2024, 1, 1), date(2024, 12, 31)
            )
            m25 = None
            if hidden_daily is not None:
                m25 = _run_market(
                    selected, combined, date(2024, 12, 31), date(2025, 1, 1), date(2025, 12, 31)
                )
        st.session_state["market_2024"] = m24
        st.session_state["market_2025"] = m25

        from energy_modelling.backtest.io import RESULTS_DIR, save_market_results

        if m24 is not None:
            save_market_results(m24, RESULTS_DIR / "market_2024.pkl")
        if m25 is not None:
            save_market_results(m25, RESULTS_DIR / "market_2025.pkl")

    m24 = st.session_state.get("market_2024")
    m25 = st.session_state.get("market_2025")

    tabs = st.tabs(["Market: 2024", "Market: 2025"])
    with tabs[0]:
        _render_market_period("2024", m24, daily_data=public_daily)
    with tabs[1]:
        if hidden_daily is None:
            st.info("Hidden 2025 data not available.")
        else:
            _render_market_period("2025", m25, daily_data=hidden_daily)
