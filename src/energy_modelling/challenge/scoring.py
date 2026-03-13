"""Scoring helpers for the student strategy challenge."""

from __future__ import annotations

import math

import pandas as pd

_ANNUALISATION_FACTOR = math.sqrt(252)


def compute_challenge_metrics(daily_pnl: pd.Series, trade_count: int) -> dict[str, float]:
    """Compute simple leaderboard metrics from daily PnL."""

    pnl = daily_pnl.astype(float)
    active = pnl[pnl != 0.0]
    n = len(pnl)

    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative

    daily_mean = float(pnl.mean()) if n > 0 else 0.0
    daily_std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    sharpe_ratio = daily_mean / daily_std * _ANNUALISATION_FACTOR if daily_std > 0 else 0.0

    wins = active[active > 0]
    losses = active[active < 0]

    return {
        "total_pnl": float(pnl.sum()),
        "days_evaluated": float(n),
        "trade_count": float(trade_count),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown.max()) if len(drawdown) > 0 else 0.0,
        "win_rate": float(len(wins) / trade_count) if trade_count > 0 else 0.0,
        "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
        "best_day": float(pnl.max()) if n > 0 else 0.0,
        "worst_day": float(pnl.min()) if n > 0 else 0.0,
    }


def leaderboard_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Return sortable leaderboard fields.

    Higher is better for the first two fields, lower is better for drawdown, so
    the final component is negated.
    """

    return (
        float(metrics["total_pnl"]),
        float(metrics["sharpe_ratio"]),
        -float(metrics["max_drawdown"]),
    )
