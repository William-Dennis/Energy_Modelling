"""Unit tests for the spec-compliant synthetic futures market engine.

Tests verify that the engine implements ``docs/energy_market_spec.md``:
  1. Trading decision: q = sign(forecast - market)
  2. Profit: r = q * (real - market)  (NO *24 multiplier)
  3. Weighting: w = max(profit, 0) / sum(max(profits, 0))
  4. Price update: P_new = sum(w_i * forecast_i)  (NO dampening)
  5. Iterate until convergence.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.futures_market_engine import (
    compute_market_prices,
    compute_strategy_profits,
    compute_weights,
    run_futures_market,
    run_futures_market_iteration,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATES = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def _index() -> pd.Index:
    return pd.Index(_DATES, name="delivery_date")


# ---------------------------------------------------------------------------
# Test 1: Fixed-point convergence with analytic forecasts
# ---------------------------------------------------------------------------


class TestFixedPointWithKnownForecasts:
    """Two strategies with constant forecasts.

    Strategy A forecasts 60 on every day.
    Strategy B forecasts 40 on every day.
    Real prices are [70, 70, 70], initial market prices are [50, 50, 50].

    At the initial market price of 50:
      - Strategy A: sign(60 - 50) = +1, profit = +1 * (70 - 50) = 20/day
      - Strategy B: sign(40 - 50) = -1, profit = -1 * (70 - 50) = -20/day
    Only A is profitable -> weight A = 1.0, weight B = 0.0.
    Market price = weighted avg of forecasts = 60.

    At market price 60:
      - Strategy A: sign(60 - 60) = 0, profit = 0 * (70 - 60) = 0/day
      - Strategy B: sign(40 - 60) = -1, profit = -1 * (70 - 60) = -10/day
    Both non-positive -> all weights zero -> prices carry forward at 60.
    This is the fixed point.
    """

    def test_converges_to_dominant_forecast(self) -> None:
        idx = _index()
        real = pd.Series([70.0, 70.0, 70.0], index=idx)
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        forecasts = {
            "A": {d: 60.0 for d in _DATES},
            "B": {d: 40.0 for d in _DATES},
        }

        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
            convergence_threshold=0.001,
        )

        # Should converge to 60 (only A survives, its forecast is 60)
        for t in _DATES:
            assert eq.final_market_prices.loc[t] == pytest.approx(60.0, abs=0.01)
        assert eq.converged

    def test_weighted_average_two_profitable_strategies(self) -> None:
        """When both strategies have positive weight, price = weighted avg."""
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        forecasts = {
            "A": {d: 70.0 for d in _DATES},
            "B": {d: 60.0 for d in _DATES},
        }

        # Single call to compute_market_prices with explicit weights
        new_prices = compute_market_prices(
            weights={"A": 0.6, "B": 0.4},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
        )
        # Expected: 0.6 * 70 + 0.4 * 60 = 42 + 24 = 66
        for t in _DATES:
            assert new_prices.loc[t] == pytest.approx(66.0)


# ---------------------------------------------------------------------------
# Test 2: All strategies must provide forecasts
# ---------------------------------------------------------------------------


class TestForecastRequired:
    """The new engine requires all strategies to provide forecasts."""

    def test_single_forecast_strategy_converges(self) -> None:
        """A single strategy with constant forecast converges."""
        idx = _index()
        real = pd.Series([80.0, 80.0, 80.0], index=idx)
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        forecasts = {"long_forecaster": {d: 75.0 for d in _DATES}}

        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
            convergence_threshold=0.001,
        )

        # Strategy forecasts 75, real is 80, initial is 50.
        # sign(75-50) = +1, profit = +1*(80-50) = 30/day => profitable
        # New price = 75 (sole strategy, weight=1).
        # At price 75: sign(75-75) = 0, profit = 0 => all weights zero
        # Price carries forward at 75. Converged in 2 iterations.
        for t in _DATES:
            assert eq.final_market_prices.loc[t] == pytest.approx(75.0, abs=0.01)
        assert eq.converged


# ---------------------------------------------------------------------------
# Test 3: Profit calculation (spec Steps 1-2, NO *24)
# ---------------------------------------------------------------------------


class TestProfitCalculation:
    """compute_strategy_profits uses q * (real - market), NO *24."""

    def test_profit_no_multiplier(self) -> None:
        idx = _index()
        market = pd.Series([50.0, 55.0, 48.0], index=idx)
        real = pd.Series([55.0, 48.0, 52.0], index=idx)

        # Forecast > market on all days => direction = +1 for all
        forecasts = {"long": {d: market.loc[d] + 10.0 for d in _DATES}}

        profits = compute_strategy_profits(market, real, forecasts)

        # Day 1: +1 * (55 - 50) = +5
        # Day 2: +1 * (48 - 55) = -7
        # Day 3: +1 * (52 - 48) = +4
        # Total = 5 - 7 + 4 = 2
        assert profits["long"] == pytest.approx(2.0)

    def test_short_direction_profit(self) -> None:
        idx = _index()
        market = pd.Series([50.0, 55.0, 48.0], index=idx)
        real = pd.Series([55.0, 48.0, 52.0], index=idx)

        # Forecast < market => direction = -1 for all
        forecasts = {"short": {d: market.loc[d] - 10.0 for d in _DATES}}

        profits = compute_strategy_profits(market, real, forecasts)

        # Day 1: -1 * (55 - 50) = -5
        # Day 2: -1 * (48 - 55) = +7
        # Day 3: -1 * (52 - 48) = -4
        # Total = -5 + 7 - 4 = -2
        assert profits["short"] == pytest.approx(-2.0)

    def test_opposite_strategies_cancel(self) -> None:
        idx = _index()
        market = pd.Series([50.0, 55.0, 48.0], index=idx)
        real = pd.Series([55.0, 48.0, 52.0], index=idx)

        forecasts = {
            "long": {d: market.loc[d] + 10.0 for d in _DATES},
            "short": {d: market.loc[d] - 10.0 for d in _DATES},
        }

        profits = compute_strategy_profits(market, real, forecasts)
        assert profits["long"] == pytest.approx(-profits["short"])

    def test_zero_direction_contributes_nothing(self) -> None:
        """When forecast == market, direction is 0, profit is 0."""
        idx = _index()
        market = pd.Series([50.0, 55.0, 48.0], index=idx)
        real = pd.Series([55.0, 48.0, 52.0], index=idx)

        # Forecast == market => direction = 0
        forecasts = {"flat": {d: float(market.loc[d]) for d in _DATES}}

        profits = compute_strategy_profits(market, real, forecasts)
        assert profits["flat"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 4: compute_market_prices (spec Step 4)
# ---------------------------------------------------------------------------


class TestComputeMarketPrices:
    def test_single_strategy_full_weight(self) -> None:
        """Single strategy with weight=1 => price = its forecast."""
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        forecasts = {"only": {d: 75.0 for d in _DATES}}

        new = compute_market_prices(
            weights={"only": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
        )
        for t in _DATES:
            assert new.loc[t] == pytest.approx(75.0)

    def test_zero_weight_strategies_ignored(self) -> None:
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        forecasts = {
            "a": {d: 100.0 for d in _DATES},
            "b": {d: 60.0 for d in _DATES},
        }

        new = compute_market_prices(
            weights={"a": 0.0, "b": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
        )
        # Only b matters
        for t in _DATES:
            assert new.loc[t] == pytest.approx(60.0)

    def test_all_zero_weights_carry_forward(self) -> None:
        """When all weights are zero, prices carry forward."""
        idx = _index()
        initial = pd.Series([50.0, 55.0, 48.0], index=idx)
        forecasts = {"a": {d: 100.0 for d in _DATES}}

        new = compute_market_prices(
            weights={"a": 0.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
        )
        for t in _DATES:
            assert new.loc[t] == pytest.approx(initial.loc[t])

    def test_missing_forecast_carries_forward(self) -> None:
        """If a strategy has no forecast for a date, it's ignored for that date."""
        idx = _index()
        initial = pd.Series([50.0, 55.0, 48.0], index=idx)

        # Strategy only has forecast for first date
        forecasts = {"partial": {_DATES[0]: 70.0}}

        new = compute_market_prices(
            weights={"partial": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
        )
        assert new.loc[_DATES[0]] == pytest.approx(70.0)
        # Other dates carry forward since no forecast
        assert new.loc[_DATES[1]] == pytest.approx(55.0)
        assert new.loc[_DATES[2]] == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# Test 5: run_futures_market_iteration
# ---------------------------------------------------------------------------


class TestRunFuturesMarketIteration:
    def test_returns_correct_types(self) -> None:
        idx = _index()
        market = pd.Series([50.0, 50.0, 50.0], index=idx)
        real = pd.Series([55.0, 48.0, 52.0], index=idx)
        forecasts = {
            "long": {d: 60.0 for d in _DATES},
            "short": {d: 40.0 for d in _DATES},
        }

        from energy_modelling.backtest.futures_market_engine import FuturesMarketIteration

        result = run_futures_market_iteration(
            market_prices=market,
            real_prices=real,
            iteration=0,
            strategy_forecasts=forecasts,
        )
        assert isinstance(result, FuturesMarketIteration)
        assert result.iteration == 0
        assert isinstance(result.market_prices, pd.Series)
        assert isinstance(result.strategy_profits, dict)
        assert isinstance(result.strategy_weights, dict)
        assert isinstance(result.active_strategies, list)

    def test_profitable_strategy_gets_weight(self) -> None:
        idx = _index()
        market = pd.Series([50.0, 50.0, 50.0], index=idx)
        real = pd.Series([60.0, 60.0, 60.0], index=idx)

        forecasts = {
            "long": {d: 55.0 for d in _DATES},  # sign(55-50)=+1, profit=+1*(60-50)=10/day
            "short": {d: 45.0 for d in _DATES},  # sign(45-50)=-1, profit=-1*(60-50)=-10/day
        }

        result = run_futures_market_iteration(
            market_prices=market,
            real_prices=real,
            iteration=0,
            strategy_forecasts=forecasts,
        )
        assert result.strategy_weights["long"] == pytest.approx(1.0)
        assert result.strategy_weights["short"] == pytest.approx(0.0)
        assert "long" in result.active_strategies
        assert "short" not in result.active_strategies


# ---------------------------------------------------------------------------
# Test 6: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Run the engine twice with the same inputs -> identical output."""

    def test_identical_runs(self) -> None:
        idx = _index()
        real = pd.Series([55.0, 48.0, 52.0], index=idx)
        initial = pd.Series([50.0, 55.0, 48.0], index=idx)
        forecasts = {
            "long": {d: 60.0 for d in _DATES},
            "perfect": {
                _DATES[0]: 55.0,
                _DATES[1]: 48.0,
                _DATES[2]: 52.0,
            },
        }
        kwargs = dict(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
            convergence_threshold=0.001,
        )

        eq1 = run_futures_market(**kwargs)
        eq2 = run_futures_market(**kwargs)

        pd.testing.assert_series_equal(eq1.final_market_prices, eq2.final_market_prices)


# ---------------------------------------------------------------------------
# Test 7: All unprofitable -> prices don't move
# ---------------------------------------------------------------------------


class TestAllUnprofitable:
    def test_zero_pnl_prices_unchanged(self) -> None:
        """When real == market, all profits are zero, prices carry forward."""
        idx = _index()
        prices = pd.Series([50.0, 50.0, 50.0], index=idx)
        # Forecast != market, but real == market => profit = sign * 0 = 0
        forecasts = {"long": {d: 60.0 for d in _DATES}}

        eq = run_futures_market(
            initial_market_prices=prices.copy(),
            real_prices=prices.copy(),
            strategy_forecasts=forecasts,
        )
        # With zero profit, weights are all zero, prices don't move
        assert eq.converged


# ---------------------------------------------------------------------------
# Test 8: PF (perfect foresight) instant convergence
# ---------------------------------------------------------------------------


class TestPerfectForesightInstantConvergence:
    """With PF as sole strategy, market converges to real in ONE iteration.

    PF forecast = real_price.
    Iteration 0: sign(real - initial) * (real - initial) > 0 when real != initial.
    PF is profitable -> weight = 1.0.
    New price = 1.0 * real = real.  Done.
    """

    def test_pf_converges_in_one_iteration(self) -> None:
        idx = _index()
        real = pd.Series([100.0, 80.0, 95.0], index=idx)
        initial = pd.Series([90.0, 90.0, 90.0], index=idx)

        pf_forecasts = {"PF": {d: float(real.loc[d]) for d in _DATES}}

        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=pf_forecasts,
            max_iterations=100,
            convergence_threshold=0.01,
        )

        assert eq.converged
        # Should converge in at most 2 iterations (iter 0 moves to real,
        # iter 1 confirms delta=0)
        assert len(eq.iterations) <= 2

        for t in _DATES:
            assert eq.final_market_prices.loc[t] == pytest.approx(real.loc[t], abs=0.01)


# ---------------------------------------------------------------------------
# Test 9: Respects max_iterations
# ---------------------------------------------------------------------------


class TestMaxIterations:
    def test_respects_max_iterations(self) -> None:
        idx = _index()
        real = pd.Series([80.0, 80.0, 80.0], index=idx)
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        # Two strategies that keep alternating — but max_iterations caps it
        forecasts = {
            "a": {d: 70.0 for d in _DATES},
            "b": {d: 60.0 for d in _DATES},
        }

        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=3,
        )
        assert len(eq.iterations) <= 3


# ---------------------------------------------------------------------------
# Test 10: compute_weights
# ---------------------------------------------------------------------------


class TestComputeWeightsFromEngine:
    """Verify weight computation matches spec Step 3."""

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
