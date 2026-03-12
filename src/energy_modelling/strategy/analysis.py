"""Performance analysis and metrics for backtested strategies.

Computes standard trading performance metrics from backtest results,
suitable for display in the Streamlit dashboard.
"""

from __future__ import annotations

import math

import pandas as pd

from energy_modelling.strategy.runner import BacktestResult

_ANNUALISATION_FACTOR = math.sqrt(252)


def compute_metrics(result: BacktestResult) -> dict[str, float]:
    """Compute key performance metrics from a backtest result.

    Parameters
    ----------
    result:
        A completed backtest result containing daily PnL.

    Returns
    -------
    dict[str, float]
        Dictionary with the following keys:

        - ``total_pnl``: Total profit/loss in EUR.
        - ``num_trading_days``: Number of days traded.
        - ``annualized_return_pct``: Annualized return percentage
          (assuming notional of avg daily abs PnL * 252).
        - ``sharpe_ratio``: Annualized Sharpe ratio (daily, sqrt(252)).
        - ``max_drawdown``: Maximum drawdown in EUR.
        - ``max_drawdown_pct``: Maximum drawdown as a percentage of
          peak cumulative PnL.
        - ``win_rate``: Fraction of profitable days.
        - ``avg_win``: Mean profit on winning days (EUR).
        - ``avg_loss``: Mean loss on losing days (EUR, negative).
        - ``profit_factor``: Gross profits / gross losses.
        - ``best_day``: Largest single-day gain (EUR).
        - ``worst_day``: Largest single-day loss (EUR).
    """
    pnl = result.daily_pnl
    n = len(pnl)

    total_pnl = float(pnl.sum())

    # Wins and losses
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float(len(wins) / n) if n > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualized from daily)
    daily_mean = float(pnl.mean()) if n > 0 else 0.0
    daily_std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    sharpe = (daily_mean / daily_std * _ANNUALISATION_FACTOR) if daily_std > 0 else float("inf")

    # Max drawdown
    cum = result.cumulative_pnl
    running_max = cum.cummax()
    drawdown = running_max - cum
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    # Max drawdown as percentage of peak
    peak_at_max_dd = float(running_max[drawdown.idxmax()]) if len(drawdown) > 0 else 0.0
    max_dd_pct = (max_dd / peak_at_max_dd * 100.0) if peak_at_max_dd > 0 else 0.0

    # Annualized return: total_pnl / years
    years = n / 252.0 if n > 0 else 1.0
    annualized_return_pct = (total_pnl / years) if years > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "num_trading_days": float(n),
        "annualized_return_pct": annualized_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "best_day": float(pnl.max()) if n > 0 else 0.0,
        "worst_day": float(pnl.min()) if n > 0 else 0.0,
    }


def monthly_pnl(result: BacktestResult) -> pd.DataFrame:
    """Compute monthly PnL breakdown.

    Parameters
    ----------
    result:
        A completed backtest result.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``year``, ``month``, ``pnl`` aggregated
        by calendar month.
    """
    pnl = result.daily_pnl.copy()
    idx = pd.DatetimeIndex(pnl.index)
    pnl.index = idx

    monthly = pnl.groupby([idx.year, idx.month]).sum()
    monthly_df = monthly.reset_index()
    monthly_df.columns = ["year", "month", "pnl"]
    return monthly_df


def rolling_sharpe(result: BacktestResult, window: int = 30) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Parameters
    ----------
    result:
        A completed backtest result.
    window:
        Rolling window size in trading days.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio indexed by delivery date.
    """
    pnl = result.daily_pnl
    rolling_mean = pnl.rolling(window=window).mean()
    rolling_std = pnl.rolling(window=window).std(ddof=1)
    return rolling_mean / rolling_std * _ANNUALISATION_FACTOR
