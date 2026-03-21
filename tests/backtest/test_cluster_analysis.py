"""Tests for Phase 10d: Regime and Cluster Analysis.

Tests the core functions from ``scripts/phase10d_cluster_analysis.py``
using small synthetic datasets.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from phase10d_cluster_analysis import (  # noqa: E402
    build_forecast_matrix,
    build_profit_matrix,
    build_weight_matrix,
    cluster_by_correlation,
    compute_cluster_dominance,
    compute_regime_forecast_stats,
    identify_volatility_regimes,
)

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dates(n: int = 5) -> list[datetime.date]:
    return [datetime.date(2024, 1, d) for d in range(1, n + 1)]


def _make_iteration(
    iteration: int,
    profits: dict[str, float],
    weights: dict[str, float],
) -> FuturesMarketIteration:
    dates = _make_dates()
    return FuturesMarketIteration(
        iteration=iteration,
        market_prices=pd.Series([50.0] * len(dates), index=dates, name="market_price"),
        strategy_profits=profits,
        strategy_weights=weights,
        active_strategies=[n for n, w in weights.items() if w > 0],
    )


@pytest.fixture()
def simple_forecasts() -> dict[str, dict]:
    dates = _make_dates()
    return {
        "A": {d: 50.0 + i for i, d in enumerate(dates)},
        "B": {d: 50.0 + i + 0.1 for i, d in enumerate(dates)},
        "C": {d: 30.0 - i for i, d in enumerate(dates)},
    }


@pytest.fixture()
def simple_equilibrium() -> FuturesMarketEquilibrium:
    iters = [
        _make_iteration(0, {"A": 10, "B": 8, "C": -5}, {"A": 0.56, "B": 0.44, "C": 0.0}),
        _make_iteration(1, {"A": 12, "B": 6, "C": -8}, {"A": 0.67, "B": 0.33, "C": 0.0}),
        _make_iteration(2, {"A": 15, "B": 4, "C": -10}, {"A": 0.79, "B": 0.21, "C": 0.0}),
    ]
    return FuturesMarketEquilibrium(
        iterations=iters,
        final_market_prices=iters[-1].market_prices,
        final_weights=iters[-1].strategy_weights,
        converged=True,
        convergence_delta=0.001,
    )


# ---------------------------------------------------------------------------
# Tests: build_forecast_matrix
# ---------------------------------------------------------------------------


class TestBuildForecastMatrix:
    def test_shape(self, simple_forecasts: dict) -> None:
        fm = build_forecast_matrix(simple_forecasts)
        assert fm.shape == (3, 5)  # 3 strategies × 5 dates

    def test_strategy_names(self, simple_forecasts: dict) -> None:
        fm = build_forecast_matrix(simple_forecasts)
        assert set(fm.index) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Tests: cluster_by_correlation
# ---------------------------------------------------------------------------


class TestClusterByCorrelation:
    def test_correlated_strategies_cluster_together(self, simple_forecasts: dict) -> None:
        fm = build_forecast_matrix(simple_forecasts)
        labels = cluster_by_correlation(fm, n_clusters=2)
        # A and B are highly correlated, C is anti-correlated
        assert labels["A"] == labels["B"]
        assert labels["A"] != labels["C"]

    def test_single_strategy(self) -> None:
        dates = _make_dates()
        fm = build_forecast_matrix({"A": {d: 50.0 for d in dates}})
        labels = cluster_by_correlation(fm, n_clusters=2)
        assert len(labels) == 1
        assert labels["A"] == 1

    def test_n_clusters_respected(self, simple_forecasts: dict) -> None:
        fm = build_forecast_matrix(simple_forecasts)
        labels = cluster_by_correlation(fm, n_clusters=3)
        assert labels.nunique() <= 3


# ---------------------------------------------------------------------------
# Tests: build_profit_matrix and build_weight_matrix
# ---------------------------------------------------------------------------


class TestProfitAndWeightMatrices:
    def test_profit_matrix_shape(self, simple_equilibrium: FuturesMarketEquilibrium) -> None:
        pm = build_profit_matrix(simple_equilibrium, ["A", "B", "C"])
        assert pm.shape == (3, 3)  # 3 strategies × 3 iterations

    def test_weight_matrix_values(self, simple_equilibrium: FuturesMarketEquilibrium) -> None:
        wm = build_weight_matrix(simple_equilibrium, ["A", "B", "C"])
        assert wm.loc["A", 0] == pytest.approx(0.56)
        assert wm.loc["C", 2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: compute_cluster_dominance
# ---------------------------------------------------------------------------


class TestClusterDominance:
    def test_dominance_sums_to_one(self, simple_equilibrium: FuturesMarketEquilibrium) -> None:
        wm = build_weight_matrix(simple_equilibrium, ["A", "B", "C"])
        labels = pd.Series({"A": 1, "B": 1, "C": 2})
        dom = compute_cluster_dominance(wm, labels)
        # Per iteration, total weight should sum to ~1.0
        for col in dom.columns:
            assert dom[col].sum() == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: identify_volatility_regimes
# ---------------------------------------------------------------------------


class TestVolatilityRegimes:
    def test_returns_two_regimes(self) -> None:
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = pd.Series(range(100), index=dates, dtype=float)
        regimes = identify_volatility_regimes(prices, window=10)
        assert set(regimes.dropna().unique()) <= {"low_vol", "high_vol"}

    def test_length_matches_input(self) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = pd.Series(range(50), index=dates, dtype=float)
        regimes = identify_volatility_regimes(prices, window=10)
        assert len(regimes) == 50


# ---------------------------------------------------------------------------
# Tests: compute_regime_forecast_stats
# ---------------------------------------------------------------------------


class TestRegimeForecastStats:
    def test_returns_dataframe(self) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        real = pd.Series(range(50), index=dates, dtype=float)
        fm = pd.DataFrame(
            {"A": range(50), "B": range(50, 100)},
            index=dates,
        ).T
        regimes = pd.Series(
            ["low_vol"] * 25 + ["high_vol"] * 25,
            index=dates,
        )
        labels = pd.Series({"A": 1, "B": 1})
        stats = compute_regime_forecast_stats(fm, real, regimes, labels)
        assert isinstance(stats, pd.DataFrame)
        assert "mae" in stats.columns
        assert "regime" in stats.columns
