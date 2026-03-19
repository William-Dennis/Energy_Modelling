"""Scoring helpers for the student strategy challenge."""

from __future__ import annotations

import math

import pandas as pd

_ANNUALISATION_FACTOR = math.sqrt(252)


def compute_challenge_metrics(daily_pnl: pd.Series, trade_count: int) -> dict[str, float]:
    """Compute leaderboard metrics from daily PnL.

    Returns a dict with keys: ``total_pnl``, ``days_evaluated``,
    ``trade_count``, ``sharpe_ratio``, ``max_drawdown``, ``win_rate``,
    ``avg_win``, ``avg_loss``, ``best_day``, ``worst_day``,
    ``profit_factor``, ``annualized_pnl_eur``.
    """

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

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = float(pnl.sum())
    years = n / 252.0 if n > 0 else 1.0
    annualized_pnl_eur = total_pnl / years if years > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "days_evaluated": float(n),
        "trade_count": float(trade_count),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown.max()) if len(drawdown) > 0 else 0.0,
        "win_rate": float(len(wins) / trade_count) if trade_count > 0 else 0.0,
        "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
        "best_day": float(pnl.max()) if n > 0 else 0.0,
        "worst_day": float(pnl.min()) if n > 0 else 0.0,
        "profit_factor": profit_factor,
        "annualized_pnl_eur": annualized_pnl_eur,
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


# ---------------------------------------------------------------------------
# Market-adjusted scoring
# ---------------------------------------------------------------------------


def compute_market_adjusted_metrics(
    market_daily_pnl: pd.Series,
    original_daily_pnl: pd.Series,
    trade_count: int,
) -> dict[str, float]:
    """Compute leaderboard metrics that include market-relative fields.

    The base metrics are computed from *market_daily_pnl* (PnL measured
    against the converged market price).  Two additional fields compare the
    strategy to its original (pre-market) performance:

    * ``alpha_pnl``: excess PnL over the original entry-price PnL.
    * ``original_total_pnl``: the strategy's PnL under the vanilla scoring.

    Parameters
    ----------
    market_daily_pnl:
        Daily PnL computed as ``direction * (settlement - market_price) * 24``.
    original_daily_pnl:
        Daily PnL computed as ``direction * (settlement - last_settlement) * 24``.
    trade_count:
        Number of days the strategy actively traded.
    """

    base = compute_challenge_metrics(market_daily_pnl, trade_count)
    original_total = float(original_daily_pnl.astype(float).sum())
    base["alpha_pnl"] = base["total_pnl"] - original_total
    base["original_total_pnl"] = original_total
    return base


def market_leaderboard_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Sortable score under market pricing.

    Same structure as :func:`leaderboard_score` but applied to
    market-adjusted metrics.
    """

    return (
        float(metrics["total_pnl"]),
        float(metrics["sharpe_ratio"]),
        -float(metrics["max_drawdown"]),
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def monthly_pnl(daily_pnl: pd.Series) -> pd.DataFrame:
    """Compute monthly PnL breakdown from a daily PnL series.

    Parameters
    ----------
    daily_pnl:
        Series indexed by date with daily PnL values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``year``, ``month``, ``pnl``.
    """
    pnl = daily_pnl.copy()
    idx = pd.DatetimeIndex(pnl.index)
    pnl.index = idx
    monthly = pnl.groupby([idx.year, idx.month]).sum()
    df = monthly.reset_index()
    df.columns = ["year", "month", "pnl"]
    return df


def rolling_sharpe(daily_pnl: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Parameters
    ----------
    daily_pnl:
        Series indexed by date with daily PnL values.
    window:
        Rolling window size in trading days.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio indexed by delivery date.
    """
    rolling_mean = daily_pnl.rolling(window=window).mean()
    rolling_std = daily_pnl.rolling(window=window).std(ddof=1)
    return rolling_mean / rolling_std * _ANNUALISATION_FACTOR
