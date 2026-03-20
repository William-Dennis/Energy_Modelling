"""Dashboard sub-package for energy market EDA and strategy evaluation.

Shared utilities used across all dashboard tabs live here to avoid
duplication.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Display-name helpers
# ---------------------------------------------------------------------------


def class_display_name(cls: type) -> str:
    """Convert a CamelCase strategy class name to a human-readable display name.

    E.g. ``AlwaysLongStrategy`` -> ``Always Long``,
         ``AlwaysShortStrategy`` -> ``Always Short``.
    """
    name = cls.__name__
    name = re.sub(r"Strategy$", "", name)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name.strip()


# ---------------------------------------------------------------------------
# Reusable chart builders
# ---------------------------------------------------------------------------

_MONTH_LABELS = [
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


def monthly_pnl_heatmap(
    daily_pnl: pd.Series,
    title: str = "Monthly PnL",
) -> go.Figure:
    """Build an RdYlGn monthly PnL heatmap from a daily PnL series.

    Works with ``BacktestResult.daily_pnl`` or any series indexed
    by date.
    """
    pnl = daily_pnl.copy()
    idx = pd.DatetimeIndex(pnl.index)
    pnl.index = idx
    monthly = pnl.groupby([idx.year, idx.month]).sum().reset_index()
    monthly.columns = ["year", "month", "pnl"]

    if monthly.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    pivot = monthly.pivot(index="year", columns="month", values="pnl").fillna(0.0)
    pivot.columns = [_MONTH_LABELS[m - 1] for m in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="Month: %{x}<br>Year: %{y}<br>PnL (EUR): %{text}<extra></extra>",
        )
    )
    fig.update_layout(title=f"{title} (EUR)", yaxis_autorange="reversed")
    return fig


def render_metric_cards(
    metrics: dict[str, float],
    *,
    prefix: str = "EUR",
) -> None:
    """Render two rows of metric cards (6 primary + 4 secondary).

    Expects the union of keys from ``compute_metrics`` and
    ``compute_backtest_metrics``.  Missing keys are silently skipped.
    """
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total PnL", f"{prefix} {metrics.get('total_pnl', 0):,.0f}")
    c2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    c3.metric(
        "Win Rate",
        f"{metrics.get('win_rate', 0):.1%}",
    )
    c4.metric("Max Drawdown", f"{prefix} {metrics.get('max_drawdown', 0):,.0f}")

    if "profit_factor" in metrics:
        c5.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    elif "trade_count" in metrics:
        c5.metric("Trades", f"{metrics['trade_count']:.0f}")
    else:
        c5.metric("--", "--")

    if "num_trading_days" in metrics:
        c6.metric("Trading Days", f"{metrics['num_trading_days']:.0f}")
    elif "days_evaluated" in metrics:
        c6.metric("Days Evaluated", f"{metrics['days_evaluated']:.0f}")
    else:
        c6.metric("--", "--")

    # Secondary row
    c7, c8, c9, c10 = st.columns(4)
    c7.metric("Avg Win", f"{prefix} {metrics.get('avg_win', 0):,.1f}")
    c8.metric("Avg Loss", f"{prefix} {metrics.get('avg_loss', 0):,.1f}")
    c9.metric("Best Day", f"{prefix} {metrics.get('best_day', 0):,.1f}")
    c10.metric("Worst Day", f"{prefix} {metrics.get('worst_day', 0):,.1f}")
