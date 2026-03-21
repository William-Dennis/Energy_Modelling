"""Walk-forward validation for strategy evaluation.

Implements expanding-window walk-forward validation where training data
grows each year while evaluation uses a single calendar year at a time.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from energy_modelling.backtest.runner import run_backtest
from energy_modelling.backtest.types import BacktestStrategy


def walk_forward_validate(
    strategy_factory: type[BacktestStrategy],
    daily_data: pd.DataFrame,
    eval_years: list[int] | None = None,
    first_train_year: int = 2019,
) -> pd.DataFrame:
    """Run expanding-window walk-forward validation.

    For each evaluation year *y* in *eval_years*, the strategy is trained
    on all data from *first_train_year* to *y - 1* and evaluated on *y*.

    Parameters
    ----------
    strategy_factory:
        A callable (typically a class) that returns a fresh
        ``BacktestStrategy`` instance.
    daily_data:
        Full daily dataset (must contain ``delivery_date`` column or
        a date index, plus all feature and label columns).
    eval_years:
        Calendar years to evaluate on.  Defaults to ``[2020, 2021, 2022, 2023, 2024]``.
    first_train_year:
        Earliest year included in training data.

    Returns
    -------
    pd.DataFrame
        One row per evaluation year with columns: ``eval_year``,
        ``train_end``, ``total_pnl``, ``sharpe_ratio``, ``win_rate``,
        ``trade_count``, ``days_evaluated``.
    """
    if eval_years is None:
        eval_years = [2020, 2021, 2022, 2023, 2024]

    # Normalise the date column
    if "delivery_date" in daily_data.columns:
        dates = pd.to_datetime(daily_data["delivery_date"]).dt.date
    else:
        dates = pd.Series(pd.to_datetime(daily_data.index).date)

    results: list[dict] = []
    for eval_year in eval_years:
        train_end = date(eval_year - 1, 12, 31)
        eval_start = date(eval_year, 1, 1)
        eval_end = date(eval_year, 12, 31)

        # Check that we have both training and evaluation data
        has_train = any(d.year >= first_train_year and d <= train_end for d in dates)
        has_eval = any(eval_start <= d <= eval_end for d in dates)

        if not has_train or not has_eval:
            continue

        strategy = strategy_factory()
        try:
            bt = run_backtest(
                strategy=strategy,
                daily_data=daily_data,
                training_end=train_end,
                evaluation_start=eval_start,
                evaluation_end=eval_end,
            )
        except ValueError:
            # No evaluation rows for this year
            continue

        results.append(
            {
                "eval_year": eval_year,
                "train_end": train_end,
                "total_pnl": bt.metrics["total_pnl"],
                "sharpe_ratio": bt.metrics["sharpe_ratio"],
                "win_rate": bt.metrics["win_rate"],
                "trade_count": bt.trade_count,
                "days_evaluated": bt.days_evaluated,
            }
        )

    return pd.DataFrame(results)
