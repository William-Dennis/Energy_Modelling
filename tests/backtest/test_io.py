"""Round-trip tests for backtest/market result persistence."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.futures_market_engine import (
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
)
from energy_modelling.backtest.futures_market_runner import FuturesMarketResult
from energy_modelling.backtest.io import (
    RESULTS_DIR,
    load_backtest_results,
    load_market_results,
    results_exist,
    save_backtest_results,
    save_market_results,
)
from energy_modelling.backtest.runner import BacktestResult


def _make_backtest_result() -> BacktestResult:
    """Create a minimal BacktestResult for testing."""
    idx = pd.Series({date(2024, 1, 1): 50.0, date(2024, 1, 2): 51.0})
    return BacktestResult(
        predictions=idx,
        daily_pnl=pd.Series({date(2024, 1, 1): 1.0, date(2024, 1, 2): -0.5}),
        cumulative_pnl=pd.Series({date(2024, 1, 1): 1.0, date(2024, 1, 2): 0.5}),
        trade_count=2,
        days_evaluated=2,
        metrics={"total_pnl": 0.5, "sharpe_ratio": 0.3},
    )


def _make_market_result() -> FuturesMarketResult:
    """Create a minimal FuturesMarketResult for testing."""
    prices = pd.Series({date(2024, 1, 1): 50.0, date(2024, 1, 2): 51.0})
    iteration = FuturesMarketIteration(
        iteration=0,
        market_prices=prices,
        strategy_profits={"strat_a": 1.0},
        strategy_weights={"strat_a": 1.0},
        active_strategies=["strat_a"],
    )
    eq = FuturesMarketEquilibrium(
        iterations=[iteration],
        final_market_prices=prices,
        final_weights={"strat_a": 1.0},
        converged=True,
        convergence_delta=0.001,
    )
    br = _make_backtest_result()
    return FuturesMarketResult(
        equilibrium=eq,
        market_results={"strat_a": br},
        original_results={"strat_a": br},
    )


# -- Backtest round-trip ---------------------------------------------------


def test_backtest_round_trip(tmp_path):
    original = {"strat_a": _make_backtest_result()}
    path = tmp_path / "bt.pkl"

    save_backtest_results(original, path)
    loaded = load_backtest_results(path)

    assert loaded is not None
    assert set(loaded.keys()) == {"strat_a"}
    pd.testing.assert_series_equal(
        loaded["strat_a"].daily_pnl, original["strat_a"].daily_pnl
    )
    assert loaded["strat_a"].trade_count == 2
    assert loaded["strat_a"].metrics == original["strat_a"].metrics


# -- Market round-trip -----------------------------------------------------


def test_market_round_trip(tmp_path):
    original = _make_market_result()
    path = tmp_path / "mkt.pkl"

    save_market_results(original, path)
    loaded = load_market_results(path)

    assert loaded is not None
    assert loaded.equilibrium.converged is True
    assert loaded.equilibrium.convergence_delta == pytest.approx(0.001)
    pd.testing.assert_series_equal(
        loaded.equilibrium.final_market_prices,
        original.equilibrium.final_market_prices,
    )
    assert set(loaded.market_results.keys()) == {"strat_a"}


# -- Load returns None when missing ----------------------------------------


def test_load_backtest_missing(tmp_path):
    assert load_backtest_results(tmp_path / "nonexistent.pkl") is None


def test_load_market_missing(tmp_path):
    assert load_market_results(tmp_path / "nonexistent.pkl") is None


# -- results_exist reflects disk state -------------------------------------


def test_results_exist_false(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "energy_modelling.backtest.io.RESULTS_DIR", tmp_path / "empty"
    )
    assert results_exist() is False


def test_results_exist_true(tmp_path, monkeypatch):
    monkeypatch.setattr("energy_modelling.backtest.io.RESULTS_DIR", tmp_path)
    pkl = tmp_path / "backtest_val_2024.pkl"
    pkl.write_bytes(b"dummy")
    assert results_exist() is True
