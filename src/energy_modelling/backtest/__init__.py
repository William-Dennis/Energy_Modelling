"""Hackathon challenge helpers for daily DE-LU futures strategies."""

from energy_modelling.backtest.benchmarks import (
    ALL_BENCHMARKS,
    biased_settlement,
    get_benchmark,
    noisy_settlement,
    perfect_foresight_price,
    yesterday_settlement,
)
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
from energy_modelling.backtest.feedback import (
    StrategyReport,
    feature_contribution_analysis,
    strategy_correlation_matrix,
)
from energy_modelling.backtest.forecast_cache import (
    clear_cache,
    get_metadata,
    is_cached,
    load_all_forecasts,
    load_forecasts,
    store_forecasts,
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
from energy_modelling.backtest.io import (
    RESULTS_DIR,
    load_backtest_results,
    load_market_results,
    results_exist,
    save_backtest_results,
    save_market_results,
)
from energy_modelling.backtest.recompute import recompute_all
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
from energy_modelling.backtest.walk_forward import walk_forward_validate

__all__ = [
    "ALL_BENCHMARKS",
    "RESULTS_DIR",
    "BacktestResult",
    "BacktestState",
    "BacktestStrategy",
    "HIDDEN_TEST_YEARS",
    "FuturesMarketEquilibrium",
    "FuturesMarketResult",
    "FuturesMarketIteration",
    "PUBLIC_TRAIN_YEARS",
    "PUBLIC_VALIDATION_YEARS",
    "biased_settlement",
    "build_daily_backtest_frame",
    "build_feature_glossary",
    "build_public_daily_dataset",
    "load_all_forecasts",
    "load_backtest_results",
    "load_forecasts",
    "load_market_results",
    "clear_cache",
    "compute_backtest_metrics",
    "compute_market_adjusted_metrics",
    "get_metadata",
    "get_benchmark",
    "is_cached",
    "leaderboard_score",
    "market_leaderboard_score",
    "monthly_pnl",
    "noisy_settlement",
    "perfect_foresight_price",
    "results_exist",
    "rolling_sharpe",
    "recompute_all",
    "save_backtest_results",
    "save_market_results",
    "run_backtest",
    "run_futures_market_evaluation",
    "run_futures_market",
    "strip_hidden_labels",
    "StrategyReport",
    "feature_contribution_analysis",
    "store_forecasts",
    "strategy_correlation_matrix",
    "walk_forward_validate",
    "write_backtest_data",
    "yesterday_settlement",
]
