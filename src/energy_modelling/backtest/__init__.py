"""Hackathon challenge helpers for daily DE-LU futures strategies."""

from energy_modelling.backtest.data import (
    HIDDEN_TEST_YEARS,
    PUBLIC_TRAIN_YEARS,
    PUBLIC_VALIDATION_YEARS,
    build_daily_backtest_frame,
    build_feature_glossary,
    build_public_daily_dataset,
    strip_hidden_labels,
    write_backtest_data,
)
from energy_modelling.backtest.futures_market_engine import (
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
    run_futures_market,
)
from energy_modelling.backtest.futures_market_runner import (
    FuturesMarketResult,
    run_futures_market_evaluation,
)
from energy_modelling.backtest.runner import BacktestResult, run_backtest
from energy_modelling.backtest.scoring import (
    compute_backtest_metrics,
    compute_market_adjusted_metrics,
    leaderboard_score,
    market_leaderboard_score,
    monthly_pnl,
    rolling_sharpe,
)
from energy_modelling.backtest.types import BacktestState, BacktestStrategy

__all__ = [
    "BacktestResult",
    "BacktestState",
    "BacktestStrategy",
    "HIDDEN_TEST_YEARS",
    "FuturesMarketEquilibrium",
    "FuturesMarketResult",
    "FuturesMarketIteration",
    "PUBLIC_TRAIN_YEARS",
    "PUBLIC_VALIDATION_YEARS",
    "build_daily_backtest_frame",
    "build_feature_glossary",
    "build_public_daily_dataset",
    "compute_backtest_metrics",
    "compute_market_adjusted_metrics",
    "leaderboard_score",
    "market_leaderboard_score",
    "monthly_pnl",
    "rolling_sharpe",
    "run_backtest",
    "run_futures_market_evaluation",
    "run_futures_market",
    "strip_hidden_labels",
    "write_backtest_data",
]
