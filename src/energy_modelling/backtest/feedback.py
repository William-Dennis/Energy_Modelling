"""Feedback loop infrastructure for strategy analysis.

Provides:
- ``StrategyReport`` dataclass for per-strategy performance summaries
- ``strategy_correlation_matrix()`` for pairwise prediction correlation
- ``feature_contribution_analysis()`` for feature-level usage and PnL stats
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StrategyReport:
    """Structured performance report for a single strategy.

    Parameters
    ----------
    name:
        Strategy class name.
    total_pnl:
        Cumulative PnL over the evaluation period (EUR).
    sharpe:
        Annualised Sharpe ratio.
    win_rate:
        Fraction of traded days that were profitable.
    daily_predictions:
        Series of +1 / -1 / NA per evaluation day.
    regime_performance:
        Optional PnL breakdown by volatility regime (low / mid / high).
    yearly_pnl:
        Optional PnL per calendar year.
    feature_usage:
        Optional list of feature column names this strategy depends on.
    """

    name: str
    total_pnl: float
    sharpe: float
    win_rate: float
    daily_predictions: pd.Series
    regime_performance: dict[str, float] | None = field(default=None)
    yearly_pnl: dict[int, float] | None = field(default=None)
    feature_usage: list[str] | None = field(default=None)


def strategy_correlation_matrix(
    predictions_map: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of daily predictions.

    Parameters
    ----------
    predictions_map:
        ``{strategy_name: predictions_series}`` where each series is
        indexed by date with values in {+1, -1, NA/None}.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix (strategy x strategy).
    """
    # Build a DataFrame with one column per strategy, coerce None/NA to NaN
    frames: dict[str, pd.Series] = {}
    for name, preds in predictions_map.items():
        s = preds.copy().astype("float64")
        frames[name] = s

    df = pd.DataFrame(frames)
    return df.corr(method="pearson")


def feature_contribution_analysis(
    feature_usage: dict[str, list[str]],
    daily_pnl_map: dict[str, pd.Series],
) -> pd.DataFrame:
    """Analyse per-feature contribution across strategies.

    For each unique feature mentioned in *feature_usage*:

    - **strategy_count**: number of strategies that list this feature.
    - **mean_pnl**: mean total PnL across strategies using this feature.

    Parameters
    ----------
    feature_usage:
        ``{strategy_name: [feature_col, ...]}`` mapping.
    daily_pnl_map:
        ``{strategy_name: daily_pnl_series}`` mapping.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns ``feature``, ``strategy_count``,
        ``mean_pnl``.
    """
    if not feature_usage:
        return pd.DataFrame(columns=["feature", "strategy_count", "mean_pnl"])

    # Invert: feature -> list of strategy names that use it
    feature_to_strategies: dict[str, list[str]] = {}
    for strat_name, features in feature_usage.items():
        for feat in features:
            feature_to_strategies.setdefault(feat, []).append(strat_name)

    rows: list[dict] = []
    for feat, strat_names in sorted(feature_to_strategies.items()):
        total_pnls: list[float] = []
        for sname in strat_names:
            if sname in daily_pnl_map:
                total_pnls.append(float(daily_pnl_map[sname].sum()))
        mean_pnl = float(np.mean(total_pnls)) if total_pnls else 0.0
        rows.append(
            {
                "feature": feat,
                "strategy_count": len(strat_names),
                "mean_pnl": mean_pnl,
            }
        )

    return pd.DataFrame(rows)
