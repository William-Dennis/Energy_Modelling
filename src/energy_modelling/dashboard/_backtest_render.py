"""Rendering helpers for the backtest-leaderboard tab.

Extracted from ``_backtest.py`` to keep each module under 300 lines.
All functions are private UI helpers called by :func:`_backtest.render`.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from energy_modelling.backtest.runner import BacktestResult
from energy_modelling.backtest.scoring import leaderboard_score
from energy_modelling.dashboard import monthly_pnl_heatmap


def _fmt_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    """Format a leaderboard frame for display."""
    return frame.drop(columns=["Score"]).assign(
        **{
            "Total PnL": lambda df: df["Total PnL"].map(lambda v: f"EUR {v:,.2f}"),
            "Sharpe": lambda df: df["Sharpe"].map(lambda v: f"{v:.2f}"),
            "Max Drawdown": lambda df: df["Max Drawdown"].map(lambda v: f"EUR {v:,.2f}"),
            "Trades": lambda df: df["Trades"].map(lambda v: f"{v:.0f}"),
            "Win Rate": lambda df: df["Win Rate"].map(lambda v: f"{v:.1%}"),
        }
    )


def _comparison_frame(
    results_by_period: dict[str, dict[str, BacktestResult]],
    attr: str,
) -> pd.DataFrame:
    """Build a long-form comparison DataFrame across periods."""
    rows = []
    for period, strats in results_by_period.items():
        for name, r in strats.items():
            for d, v in getattr(r, attr).items():
                rows.append({"Date": d, "Strategy": name, "Period": period, "Value": v})
    return pd.DataFrame(rows)


def _render_period_summary(
    val: dict[str, BacktestResult],
    hid: dict[str, BacktestResult],
) -> None:
    """Render the period summary table."""
    rows = []
    for name in sorted(set(val) | set(hid)):
        v, h = val.get(name), hid.get(name)
        rows.append(
            {
                "Strategy": name,
                "2024 PnL": v.metrics["total_pnl"] if v else float("nan"),
                "2024 Sharpe": v.metrics["sharpe_ratio"] if v else float("nan"),
                "2024 Max DD": v.metrics["max_drawdown"] if v else float("nan"),
                "2025 PnL": h.metrics["total_pnl"] if h else float("nan"),
                "2025 Sharpe": h.metrics["sharpe_ratio"] if h else float("nan"),
                "2025 Max DD": h.metrics["max_drawdown"] if h else float("nan"),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["2024 PnL", "2025 PnL"], ascending=[False, False])

    def _safe(v: float, fmt: str) -> str:
        return "-" if pd.isna(v) else fmt.format(v)

    st.dataframe(
        frame.assign(
            **{
                "2024 PnL": lambda df: df["2024 PnL"].map(lambda v: f"EUR {v:,.2f}"),
                "2024 Sharpe": lambda df: df["2024 Sharpe"].map(lambda v: f"{v:.2f}"),
                "2024 Max DD": lambda df: df["2024 Max DD"].map(lambda v: f"EUR {v:,.2f}"),
                "2025 PnL": lambda df: df["2025 PnL"].map(lambda v: _safe(v, "EUR {0:,.2f}")),
                "2025 Sharpe": lambda df: df["2025 Sharpe"].map(lambda v: _safe(v, "{0:.2f}")),
                "2025 Max DD": lambda df: df["2025 Max DD"].map(lambda v: _safe(v, "EUR {0:,.2f}")),
            }
        ),
        use_container_width=True,
    )


def _render_feature_timing(glossary: pd.DataFrame) -> None:
    """Render the feature timing section."""
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(
            glossary.groupby("timing_group").size().reset_index(name="columns"),
            use_container_width=True,
        )
    with c2:
        t1, t2 = st.tabs(["Lagged Realised", "Same-Day Forecast"])
        with t1:
            st.dataframe(
                glossary[glossary["timing_group"] == "lagged_realised"][["column", "description"]],
                use_container_width=True,
            )
        with t2:
            st.dataframe(
                glossary[glossary["timing_group"] == "same_day_forecast"][
                    ["column", "description"]
                ],
                use_container_width=True,
            )


def _render_leaderboards(lb_2024: pd.DataFrame, lb_2025: pd.DataFrame) -> None:
    """Render the 2024/2025 leaderboard tabs."""
    tabs = st.tabs(["2024 Validation", "2025 Hidden Test"])
    with tabs[0]:
        st.subheader("2024 Leaderboard (Yesterday-Settlement Price)")
        st.dataframe(_fmt_leaderboard(lb_2024), use_container_width=True)
    with tabs[1]:
        if lb_2025.empty:
            st.info("Hidden 2025 data not available.")
        else:
            st.subheader("2025 Leaderboard (Yesterday-Settlement Price)")
            st.dataframe(_fmt_leaderboard(lb_2025), use_container_width=True)


def _render_cumulative_pnl(
    val_results: dict[str, BacktestResult],
    hid_results: dict[str, BacktestResult],
) -> None:
    """Render the cumulative PnL comparison chart."""
    st.header("Cumulative PnL Comparison")
    cum_df = _comparison_frame({"2024": val_results, "2025": hid_results}, "cumulative_pnl")
    if not cum_df.empty:
        fig = px.line(
            cum_df,
            x="Date",
            y="Value",
            color="Strategy",
            line_dash="Period",
            title="Cumulative PnL",
            labels={"Value": "Cumulative PnL"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


def _render_drawdown(
    val_results: dict[str, BacktestResult],
    hid_results: dict[str, BacktestResult],
) -> None:
    """Render the drawdown comparison chart."""
    st.header("Drawdown Comparison")
    dd_payload: dict[str, dict[str, BacktestResult]] = {"2024": {}, "2025": {}}
    for period, results in [("2024", val_results), ("2025", hid_results)]:
        for name, r in results.items():
            dd_payload[period][name] = BacktestResult(
                predictions=r.predictions,
                daily_pnl=r.daily_pnl,
                cumulative_pnl=r.cumulative_pnl.cummax() - r.cumulative_pnl,
                trade_count=r.trade_count,
                days_evaluated=r.days_evaluated,
                metrics=r.metrics,
            )
    dd_df = _comparison_frame(dd_payload, "cumulative_pnl")
    if not dd_df.empty:
        fig = px.line(
            dd_df,
            x="Date",
            y="Value",
            color="Strategy",
            line_dash="Period",
            title="Drawdown",
            labels={"Value": "Drawdown"},
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_strategy_detail(
    val_results: dict[str, BacktestResult],
    hid_results: dict[str, BacktestResult],
) -> None:
    """Render the strategy detail section with selectable strategy."""
    st.header("Strategy Detail")
    focus = st.selectbox("Inspect strategy", options=list(val_results.keys()), key="ch_focus")
    dtabs = st.tabs(["2024 Detail", "2025 Detail"])
    with dtabs[0]:
        _render_detail(focus, "2024", val_results.get(focus))
    with dtabs[1]:
        _render_detail(focus, "2025", hid_results.get(focus))


def _render_detail_charts(
    strategy_name: str, period: str, result: BacktestResult,
) -> None:
    """Render PnL histogram, heatmap, and predictions table."""
    cl, cr = st.columns(2)
    with cl:
        fig = px.histogram(
            result.daily_pnl, nbins=40,
            title=f"Daily PnL - {strategy_name} ({period})",
            labels={"value": "Daily PnL", "count": "Frequency"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        st.plotly_chart(
            monthly_pnl_heatmap(
                result.daily_pnl, title=f"Monthly PnL - {strategy_name} ({period})"
            ),
            use_container_width=True,
        )
    st.subheader(f"Latest Predictions - {period}")
    st.dataframe(
        pd.DataFrame({
            "prediction": result.predictions,
            "daily_pnl": result.daily_pnl,
            "cumulative_pnl": result.cumulative_pnl,
        }).tail(20),
        use_container_width=True,
    )


def _render_detail(
    strategy_name: str,
    period: str,
    result: BacktestResult | None,
) -> None:
    """Render detailed metrics, charts, and predictions for one strategy."""
    if result is None:
        st.info(f"No {period} result available.")
        return

    m = result.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total PnL", f"EUR {m['total_pnl']:,.0f}")
    c2.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
    c3.metric("Max Drawdown", f"EUR {m['max_drawdown']:,.0f}")
    c4.metric("Trades", f"{m['trade_count']:.0f}")
    c5.metric("Win Rate", f"{m['win_rate']:.1%}")
    _render_detail_charts(strategy_name, period, result)
