"""Tests for Phase 10f: Strategy Robustness Analysis.

Tests the core functions from ``scripts/phase10f_strategy_robustness.py``
using small synthetic datasets -- no real data or pickles needed.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from phase10f_strategy_robustness import (  # noqa: E402
    classify_strategy,
    compute_forecast_redundancy,
    compute_standalone_pnl,
    compute_weight_stability,
    run_market_fast,
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


@pytest.fixture()
def dates() -> list[datetime.date]:
    return _make_dates()


@pytest.fixture()
def simple_forecasts(dates: list[datetime.date]) -> dict[str, dict]:
    return {
        "StratA": {d: 50.0 + i for i, d in enumerate(dates)},
        "StratB": {d: 45.0 - i for i, d in enumerate(dates)},
        "StratC": {d: 48.0 for d in dates},
    }


@pytest.fixture()
def correlated_forecasts(dates: list[datetime.date]) -> dict[str, dict]:
    """Two nearly identical strategies plus one different one."""
    return {
        "CorrA": {d: 50.0 + i * 2.0 for i, d in enumerate(dates)},
        "CorrB": {d: 50.1 + i * 2.0 for i, d in enumerate(dates)},
        "Different": {d: 30.0 - i for i, d in enumerate(dates)},
    }


@pytest.fixture()
def real_prices(dates: list[datetime.date]) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    return pd.Series([50.0, 52.0, 48.0, 55.0, 47.0], index=idx, name="real_price")


@pytest.fixture()
def initial_prices(dates: list[datetime.date]) -> pd.Series:
    idx = pd.DatetimeIndex(dates)
    return pd.Series([49.0, 51.0, 49.0, 53.0, 48.0], index=idx, name="initial_price")


def _make_iteration(
    iteration: int,
    profits: dict[str, float],
    weights: dict[str, float],
    n_dates: int = 5,
) -> FuturesMarketIteration:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    active = [n for n, w in weights.items() if w > 0.0]
    return FuturesMarketIteration(
        iteration=iteration,
        market_prices=pd.Series([50.0] * n_dates, index=dates),
        strategy_profits=profits,
        strategy_weights=weights,
        active_strategies=active,
    )


@pytest.fixture()
def simple_equilibrium() -> FuturesMarketEquilibrium:
    iters = []
    for k in range(5):
        w_a = max(0.0, 0.6 - k * 0.1)
        w_b = max(0.0, 0.3 - k * 0.05)
        w_c = 1.0 - w_a - w_b if (w_a + w_b) < 1.0 else 0.0
        iters.append(
            _make_iteration(
                k,
                {"StratA": 100.0, "StratB": 50.0, "StratC": -10.0},
                {"StratA": w_a, "StratB": w_b, "StratC": w_c},
            )
        )
    return FuturesMarketEquilibrium(
        iterations=iters,
        final_market_prices=iters[-1].market_prices,
        final_weights=iters[-1].strategy_weights,
        converged=False,
        convergence_delta=1.0,
    )


# ---------------------------------------------------------------------------
# Test: compute_standalone_pnl
# ---------------------------------------------------------------------------


class TestComputeStandalonePnl:
    def test_returns_all_strategies(self, simple_forecasts, real_prices, initial_prices):
        pnl = compute_standalone_pnl(simple_forecasts, real_prices, initial_prices)
        assert set(pnl.keys()) == {"StratA", "StratB", "StratC"}

    def test_sign_rule_logic(self, dates):
        """If forecast > initial and real > initial, PnL should be positive."""
        idx = pd.DatetimeIndex([dates[0]])
        real = pd.Series([60.0], index=idx)
        init = pd.Series([50.0], index=idx)
        # Use Timestamp key to match .get() on the pd.Series index
        ts = idx[0]
        forecasts = {"Bull": {ts: 55.0}}  # forecast > init -> long
        pnl = compute_standalone_pnl(forecasts, real, init)
        assert pnl["Bull"] > 0  # real > init, so long is profitable

    def test_short_position(self, dates):
        """If forecast < initial and real < initial, PnL should be positive."""
        idx = pd.DatetimeIndex([dates[0]])
        real = pd.Series([40.0], index=idx)
        init = pd.Series([50.0], index=idx)
        ts = idx[0]
        forecasts = {"Bear": {ts: 45.0}}  # forecast < init -> short
        pnl = compute_standalone_pnl(forecasts, real, init)
        assert pnl["Bear"] > 0  # real < init, so short is profitable


# ---------------------------------------------------------------------------
# Test: compute_weight_stability
# ---------------------------------------------------------------------------


class TestComputeWeightStability:
    def test_returns_all_strategies(self, simple_equilibrium):
        stability = compute_weight_stability(simple_equilibrium, ["StratA", "StratB", "StratC"])
        assert set(stability.keys()) == {"StratA", "StratB", "StratC"}

    def test_constant_weight_zero_variance(self):
        """A strategy with constant weight should have zero variance."""
        iters = [_make_iteration(k, {"A": 10.0}, {"A": 0.5}) for k in range(5)]
        eq = FuturesMarketEquilibrium(
            iterations=iters,
            final_market_prices=iters[-1].market_prices,
            final_weights={"A": 0.5},
            converged=True,
            convergence_delta=0.0,
        )
        stability = compute_weight_stability(eq, ["A"])
        assert stability["A"] == 0.0

    def test_varying_weight_nonzero_variance(self, simple_equilibrium):
        stability = compute_weight_stability(simple_equilibrium, ["StratA"])
        assert stability["StratA"] > 0.0


# ---------------------------------------------------------------------------
# Test: compute_forecast_redundancy
# ---------------------------------------------------------------------------


class TestComputeForecastRedundancy:
    def test_correlated_strategies_high_redundancy(self, correlated_forecasts):
        red = compute_forecast_redundancy(correlated_forecasts)
        # CorrA and CorrB are nearly identical
        assert red["CorrA"] > 0.99
        assert red["CorrB"] > 0.99

    def test_uncorrelated_lower_redundancy(self, simple_forecasts):
        red = compute_forecast_redundancy(simple_forecasts)
        # Correlation can slightly exceed 1.0 due to floating-point; clamp check
        assert all(-0.01 <= v <= 1.01 for v in red.values())

    def test_single_strategy(self, dates):
        forecasts = {"Only": {d: float(i) for i, d in enumerate(dates)}}
        red = compute_forecast_redundancy(forecasts)
        assert red["Only"] == 0.0


# ---------------------------------------------------------------------------
# Test: classify_strategy
# ---------------------------------------------------------------------------


class TestClassifyStrategy:
    def test_robust(self):
        result = classify_strategy(
            standalone_pnl=100.0,
            market_adjusted_pnl=80.0,
            loo_mae_delta=0.5,  # positive = helps
            redundancy=0.5,
            standalone_median=50.0,
            market_median=40.0,
        )
        assert result == "robust"

    def test_destabilising(self):
        result = classify_strategy(
            standalone_pnl=100.0,
            market_adjusted_pnl=80.0,
            loo_mae_delta=-0.5,  # negative = removing helps
            redundancy=0.5,
            standalone_median=50.0,
            market_median=40.0,
        )
        assert result == "destabilising"

    def test_redundant(self):
        result = classify_strategy(
            standalone_pnl=100.0,
            market_adjusted_pnl=80.0,
            loo_mae_delta=0.05,
            redundancy=0.96,  # very correlated
            standalone_median=50.0,
            market_median=40.0,
        )
        assert result == "redundant"

    def test_standalone_only(self):
        result = classify_strategy(
            standalone_pnl=100.0,
            market_adjusted_pnl=10.0,  # below median
            loo_mae_delta=0.05,
            redundancy=0.5,
            standalone_median=50.0,
            market_median=40.0,
        )
        assert result == "standalone_only"

    def test_weak(self):
        result = classify_strategy(
            standalone_pnl=10.0,  # below median
            market_adjusted_pnl=10.0,  # below median
            loo_mae_delta=0.05,
            redundancy=0.5,
            standalone_median=50.0,
            market_median=40.0,
        )
        assert result == "weak"

    def test_none_loo_skips_destabilising_check(self):
        """When LOO is None, destabilising classification should not trigger."""
        result = classify_strategy(
            standalone_pnl=100.0,
            market_adjusted_pnl=80.0,
            loo_mae_delta=None,
            redundancy=0.5,
            standalone_median=50.0,
            market_median=40.0,
        )
        assert result == "robust"


# ---------------------------------------------------------------------------
# Test: run_market_fast
# ---------------------------------------------------------------------------


class TestRunMarketFast:
    def test_returns_expected_keys(self, simple_forecasts, real_prices, initial_prices):
        result = run_market_fast(
            simple_forecasts,
            real_prices,
            initial_prices,
            max_iterations=10,
            ema_alpha=0.5,
        )
        assert "converged" in result
        assert "mae" in result
        assert "rmse" in result
        assert "final_profits" in result
        assert "final_weights" in result

    def test_small_market_runs(self, simple_forecasts, real_prices, initial_prices):
        result = run_market_fast(
            simple_forecasts,
            real_prices,
            initial_prices,
            max_iterations=5,
            ema_alpha=0.1,
        )
        assert result["n_strategies"] == 3
        assert result["n_iterations"] <= 5
        assert result["mae"] >= 0.0

    def test_ema_alpha_1_undampened(self, simple_forecasts, real_prices, initial_prices):
        """ema_alpha=1.0 should give undampened spec behaviour."""
        result = run_market_fast(
            simple_forecasts,
            real_prices,
            initial_prices,
            max_iterations=10,
            ema_alpha=1.0,
        )
        assert result["n_iterations"] <= 10
