"""Persistence helpers for backtest and market results.

Provides save/load functions so the dashboard can display pre-computed
results immediately on startup without rerunning backtests.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from energy_modelling.backtest.futures_market_runner import FuturesMarketResult
from energy_modelling.backtest.runner import BacktestResult

RESULTS_DIR = Path("data/results")


def save_backtest_results(results: dict[str, BacktestResult], path: Path) -> None:
    """Serialize backtest results to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_backtest_results(path: Path) -> dict[str, BacktestResult] | None:
    """Load backtest results from disk. Returns None if not found."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def save_market_results(result: FuturesMarketResult, path: Path) -> None:
    """Serialize market evaluation result to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_market_results(path: Path) -> FuturesMarketResult | None:
    """Load market evaluation result from disk. Returns None if not found."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def results_exist() -> bool:
    """Check if pre-computed results are available on disk."""
    return (RESULTS_DIR / "backtest_val_2024.pkl").exists()
