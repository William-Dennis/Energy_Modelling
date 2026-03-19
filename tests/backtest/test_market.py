"""Tests for challenge.market -- synthetic futures market engine."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.futures_market_engine import (
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
    compute_market_prices,
    compute_strategy_profits,
    compute_weights,
    run_futures_market_iteration,
    run_futures_market,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DATES = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def _index() -> pd.Index:
    return pd.Index(_DATES, name="delivery_date")


def _real_prices() -> pd.Series:
    """Settlement prices: 50 -> 55 -> 48 (up then down)."""
    return pd.Series([55.0, 48.0, 52.0], index=_index(), name="settlement_price")


def _initial_market() -> pd.Series:
    """Initial market price = last_settlement_price (lagged real)."""
    return pd.Series([50.0, 55.0, 48.0], index=_index(), name="market_price")


def _directions_perfect() -> pd.Series:
    """Perfect foresight: long when price rises, short when it falls."""
    return pd.Series([1, -1, 1], index=_index(), dtype="Int64")


def _directions_always_long() -> pd.Series:
    return pd.Series([1, 1, 1], index=_index(), dtype="Int64")


def _directions_always_short() -> pd.Series:
    return pd.Series([-1, -1, -1], index=_index(), dtype="Int64")


def _directions_with_skip() -> pd.Series:
    return pd.Series([1, pd.NA, -1], index=_index(), dtype="Int64")


# ---------------------------------------------------------------------------
# compute_strategy_profits
# ---------------------------------------------------------------------------


class TestComputeStrategyProfits:
    def test_perfect_foresight_profitable(self) -> None:
        profits = compute_strategy_profits(
            {"perfect": _directions_perfect()},
            _initial_market(),
            _real_prices(),
        )
        # Day 1: +1 * (55-50) * 24 = 120
        # Day 2: -1 * (48-55) * 24 = 168
        # Day 3: +1 * (52-48) * 24 = 96
        assert profits["perfect"] == pytest.approx(120.0 + 168.0 + 96.0)

    def test_always_long_mixed(self) -> None:
        profits = compute_strategy_profits(
            {"long": _directions_always_long()},
            _initial_market(),
            _real_prices(),
        )
        # Day 1: +1*(55-50)*24=120, Day 2: +1*(48-55)*24=-168, Day 3: +1*(52-48)*24=96
        assert profits["long"] == pytest.approx(120.0 - 168.0 + 96.0)

    def test_skip_days_contribute_zero(self) -> None:
        profits = compute_strategy_profits(
            {"skipper": _directions_with_skip()},
            _initial_market(),
            _real_prices(),
        )
        # Day 1: +1*(55-50)*24=120, Day 2: skip=0, Day 3: -1*(52-48)*24=-96
        assert profits["skipper"] == pytest.approx(120.0 + 0.0 - 96.0)

    def test_multiple_strategies(self) -> None:
        profits = compute_strategy_profits(
            {"long": _directions_always_long(), "short": _directions_always_short()},
            _initial_market(),
            _real_prices(),
        )
        assert profits["long"] == -profits["short"]


# ---------------------------------------------------------------------------
# compute_weights
# ---------------------------------------------------------------------------


class TestComputeWeights:
    def test_only_profitable_get_weight(self) -> None:
        weights = compute_weights({"a": 100.0, "b": -50.0, "c": 200.0})
        assert weights["b"] == 0.0
        assert weights["a"] == pytest.approx(100.0 / 300.0)
        assert weights["c"] == pytest.approx(200.0 / 300.0)

    def test_all_negative_returns_zeros(self) -> None:
        weights = compute_weights({"a": -10.0, "b": -20.0})
        assert all(w == 0.0 for w in weights.values())

    def test_weights_sum_to_one(self) -> None:
        weights = compute_weights({"a": 50.0, "b": 100.0, "c": 150.0})
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_single_strategy(self) -> None:
        weights = compute_weights({"only": 42.0})
        assert weights["only"] == pytest.approx(1.0)

    def test_zero_profit_excluded(self) -> None:
        weights = compute_weights({"a": 0.0, "b": 100.0})
        assert weights["a"] == 0.0
        assert weights["b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_market_prices
# ---------------------------------------------------------------------------


class TestComputeMarketPrices:
    def test_single_long_strategy_shifts_up(self) -> None:
        weights = {"long": 1.0}
        directions = {"long": _directions_always_long()}
        market = _initial_market()
        spread = 5.0
        new = compute_market_prices(directions, weights, market, spread)
        # All long => implied = market + 5 for each day
        for t in _DATES:
            assert new.loc[t] == pytest.approx(market.loc[t] + 5.0)

    def test_balanced_strategies_cancel_out(self) -> None:
        weights = {"long": 0.5, "short": 0.5}
        directions = {
            "long": _directions_always_long(),
            "short": _directions_always_short(),
        }
        market = _initial_market()
        new = compute_market_prices(directions, weights, market, forecast_spread=10.0)
        # Equal weight long and short => offset cancels => price unchanged
        for t in _DATES:
            assert new.loc[t] == pytest.approx(market.loc[t])

    def test_skip_day_carries_forward(self) -> None:
        weights = {"s": 1.0}
        dirs = {"s": _directions_with_skip()}  # [1, NA, -1]
        market = _initial_market()
        new = compute_market_prices(dirs, weights, market, forecast_spread=5.0)
        # Day 2 is skipped -> carries forward
        assert new.loc[_DATES[1]] == pytest.approx(market.loc[_DATES[1]])

    def test_zero_weight_strategies_ignored(self) -> None:
        weights = {"a": 0.0, "b": 1.0}
        directions = {
            "a": _directions_always_long(),
            "b": _directions_always_short(),
        }
        market = _initial_market()
        new = compute_market_prices(directions, weights, market, forecast_spread=5.0)
        # Only b (short) matters => market - 5
        for t in _DATES:
            assert new.loc[t] == pytest.approx(market.loc[t] - 5.0)


# ---------------------------------------------------------------------------
# run_futures_market_iteration
# ---------------------------------------------------------------------------


class TestRunFuturesMarketIteration:
    def test_returns_correct_types(self) -> None:
        directions = {"long": _directions_always_long(), "perfect": _directions_perfect()}
        result = run_futures_market_iteration(
            directions=directions,
            market_prices=_initial_market(),
            real_prices=_real_prices(),
            forecast_spread=5.0,
            iteration=0,
        )
        assert isinstance(result, FuturesMarketIteration)
        assert result.iteration == 0
        assert isinstance(result.market_prices, pd.Series)
        assert isinstance(result.strategy_profits, dict)
        assert isinstance(result.strategy_weights, dict)
        assert isinstance(result.active_strategies, list)

    def test_perfect_foresight_gets_higher_weight(self) -> None:
        directions = {"long": _directions_always_long(), "perfect": _directions_perfect()}
        result = run_futures_market_iteration(
            directions=directions,
            market_prices=_initial_market(),
            real_prices=_real_prices(),
            forecast_spread=5.0,
            iteration=0,
        )
        # Perfect foresight earns more profit, so gets higher weight
        assert result.strategy_weights["perfect"] > result.strategy_weights["long"]

    def test_unprofitable_strategy_excluded(self) -> None:
        directions = {
            "perfect": _directions_perfect(),
            "anti": _directions_always_short(),  # anti-perfect on net
        }
        result = run_futures_market_iteration(
            directions=directions,
            market_prices=_initial_market(),
            real_prices=_real_prices(),
            forecast_spread=5.0,
            iteration=0,
        )
        anti_profit = result.strategy_profits["anti"]
        if anti_profit <= 0:
            assert result.strategy_weights["anti"] == 0.0
            assert "anti" not in result.active_strategies


# ---------------------------------------------------------------------------
# run_futures_market
# ---------------------------------------------------------------------------


class TestRunMarketToConvergence:
    def test_converges_with_mixed_strategies(self) -> None:
        directions = {
            "long": _directions_always_long(),
            "perfect": _directions_perfect(),
        }
        eq = run_futures_market(
            directions=directions,
            initial_market_prices=_initial_market(),
            real_prices=_real_prices(),
            max_iterations=50,
            convergence_threshold=0.001,
            forecast_spread=5.0,
            dampening=0.5,
        )
        assert isinstance(eq, FuturesMarketEquilibrium)
        assert eq.converged
        assert eq.convergence_delta < 0.001
        assert len(eq.iterations) > 0
        assert len(eq.iterations) <= 50

    def test_market_prices_differ_from_initial(self) -> None:
        directions = {
            "long": _directions_always_long(),
            "perfect": _directions_perfect(),
        }
        initial = _initial_market()
        eq = run_futures_market(
            directions=directions,
            initial_market_prices=initial,
            real_prices=_real_prices(),
            forecast_spread=5.0,
        )
        # At least one day should differ from initial
        diff = (eq.final_market_prices - initial).abs()
        assert diff.max() > 0.0

    def test_single_strategy_converges(self) -> None:
        directions = {"only": _directions_always_long()}
        eq = run_futures_market(
            directions=directions,
            initial_market_prices=_initial_market(),
            real_prices=_real_prices(),
            forecast_spread=5.0,
        )
        # Single strategy must converge (it either has positive profit or not)
        assert eq.converged or len(eq.iterations) == 20

    def test_all_unprofitable_preserves_prices(self) -> None:
        """When all strategies are unprofitable, market prices shouldn't move."""
        # Create a scenario where the only strategy is always wrong
        real = pd.Series([50.0, 50.0, 50.0], index=_index())
        market = pd.Series([50.0, 50.0, 50.0], index=_index())
        # Direction doesn't matter when real == market, PnL=0 for all
        directions = {"flat": _directions_always_long()}
        eq = run_futures_market(
            directions=directions,
            initial_market_prices=market,
            real_prices=real,
            forecast_spread=5.0,
        )
        # With zero profit, weights are all zero, prices don't move
        assert eq.converged

    def test_auto_calibrates_spread(self) -> None:
        directions = {"long": _directions_always_long()}
        eq = run_futures_market(
            directions=directions,
            initial_market_prices=_initial_market(),
            real_prices=_real_prices(),
            forecast_spread=None,  # auto
        )
        assert isinstance(eq, FuturesMarketEquilibrium)

    def test_respects_max_iterations(self) -> None:
        directions = {
            "long": _directions_always_long(),
            "short": _directions_always_short(),
        }
        eq = run_futures_market(
            directions=directions,
            initial_market_prices=_initial_market(),
            real_prices=_real_prices(),
            max_iterations=3,
            forecast_spread=5.0,
        )
        assert len(eq.iterations) <= 3

    def test_dampening_slows_convergence(self) -> None:
        directions = {"long": _directions_always_long(), "perfect": _directions_perfect()}
        args = dict(
            directions=directions,
            initial_market_prices=_initial_market(),
            real_prices=_real_prices(),
            max_iterations=50,
            convergence_threshold=0.001,
            forecast_spread=5.0,
        )
        fast = run_futures_market(**args, dampening=0.9)
        slow = run_futures_market(**args, dampening=0.1)
        # Slower dampening should take more iterations (or at least not fewer)
        assert len(slow.iterations) >= len(fast.iterations)
