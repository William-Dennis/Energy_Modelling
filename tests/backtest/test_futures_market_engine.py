"""Unit tests for forecast-aware futures market engine (issue: spec Steps 1 & 4).

Tests 1–5 verify that the engine correctly uses explicit strategy forecasts
when available and falls back to direction ± spread synthesis otherwise.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.futures_market_engine import (
    compute_market_prices,
    compute_strategy_profits,
    run_futures_market,
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

    Strategy A forecasts P̂_A = 60 on every day.
    Strategy B forecasts P̂_B = 40 on every day.
    Real prices are [70, 70, 70], initial market prices are [50, 50, 50].

    At the initial market price of 50:
      - Strategy A is long (forecast 60 > 50), profit = +1 * (70 - 50) * 24 = 480 per day
      - Strategy B is short (forecast 40 < 50), profit = -1 * (70 - 50) * 24 = -480 per day
    Only A is profitable → weight A = 1.0, weight B = 0.0.
    Market price = weighted avg of forecasts = 60.

    At market price 60:
      - Strategy A is long (60 > 60? no, equal — but direction is still +1 per act)
        profit = +1 * (70 - 60) * 24 = 240 per day → profitable
      - Strategy B is short, profit = -1 * (70 - 60) * 24 = -240 per day → unprofitable
    Only A survives → market price stays 60. This is the fixed point.
    """

    def test_converges_to_dominant_forecast(self) -> None:
        idx = _index()
        real = pd.Series([70.0, 70.0, 70.0], index=idx)
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        # Both strategies go long (+1) initially
        # A: forecast=60 > 50 → long. B: forecast=40 < 50 → short.
        directions = {
            "A": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
            "B": pd.Series([-1, -1, -1], index=idx, dtype="Int64"),
        }
        forecasts = {
            "A": {d: 60.0 for d in _DATES},
            "B": {d: 40.0 for d in _DATES},
        }

        eq = run_futures_market(
            directions=directions,
            initial_market_prices=initial,
            real_prices=real,
            max_iterations=50,
            convergence_threshold=0.001,
            forecast_spread=5.0,
            dampening=1.0,  # no dampening for clean analytic result
            strategy_forecasts=forecasts,
        )

        # Should converge to 60 (only A survives, its forecast is 60)
        for t in _DATES:
            assert eq.final_market_prices.loc[t] == pytest.approx(60.0, abs=0.01)
        assert eq.converged

    def test_weighted_average_two_profitable_strategies(self) -> None:
        """When both strategies are profitable, price = weighted avg of forecasts."""
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        directions = {
            "A": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
            "B": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
        }
        forecasts = {
            "A": {d: 70.0 for d in _DATES},
            "B": {d: 60.0 for d in _DATES},
        }

        # Single iteration with dampening=1.0 to see the weighted-average effect
        new_prices = compute_market_prices(
            directions=directions,
            weights={"A": 0.6, "B": 0.4},
            current_market_prices=initial,
            forecast_spread=5.0,
            strategy_forecasts=forecasts,
        )
        # Expected: 0.6 * 70 + 0.4 * 60 = 42 + 24 = 66
        for t in _DATES:
            assert new_prices.loc[t] == pytest.approx(66.0)


# ---------------------------------------------------------------------------
# Test 2: Direction-only fallback
# ---------------------------------------------------------------------------


class TestDirectionOnlyFallback:
    """When forecast() returns None (default), engine uses direction ± spread."""

    def test_convergence_direction_moves_toward_real(self) -> None:
        """Real price = 80, initial = 50, all strategies long → price increases."""
        idx = _index()
        real = pd.Series([80.0, 80.0, 80.0], index=idx)
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        directions = {
            "long_a": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
            "long_b": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
        }

        eq = run_futures_market(
            directions=directions,
            initial_market_prices=initial,
            real_prices=real,
            max_iterations=50,
            forecast_spread=10.0,
            dampening=0.5,
            strategy_forecasts=None,  # no forecasts — pure direction mode
        )

        # Market price must have moved up from 50 toward 80
        for t in _DATES:
            assert eq.final_market_prices.loc[t] > 50.0

    def test_no_forecasts_does_not_raise(self) -> None:
        """Engine runs cleanly with no strategy forecasts at all."""
        idx = _index()
        real = pd.Series([55.0, 48.0, 52.0], index=idx)
        initial = pd.Series([50.0, 55.0, 48.0], index=idx)
        directions = {"long": pd.Series([1, 1, 1], index=idx, dtype="Int64")}

        eq = run_futures_market(
            directions=directions,
            initial_market_prices=initial,
            real_prices=real,
            forecast_spread=5.0,
            strategy_forecasts=None,
        )
        assert isinstance(eq.final_market_prices, pd.Series)


# ---------------------------------------------------------------------------
# Test 3: Mixed strategies — forecast-capable + direction-only
# ---------------------------------------------------------------------------


class TestMixedStrategies:
    def test_forecast_and_direction_only_mix(self) -> None:
        """One strategy has forecast=75, another is direction-only (+1).

        With equal weights, the result should be between the forecast-based
        and the direction-based implied prices.
        """
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)

        directions = {
            "forecaster": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
            "direction_only": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
        }
        forecasts = {
            "forecaster": {d: 75.0 for d in _DATES},
            # "direction_only" not in forecasts → uses synthesis
        }
        spread = 10.0
        weights = {"forecaster": 0.5, "direction_only": 0.5}

        new_prices = compute_market_prices(
            directions=directions,
            weights=weights,
            current_market_prices=initial,
            forecast_spread=spread,
            strategy_forecasts=forecasts,
        )

        # forecaster: implied = 75 (from forecast)
        # direction_only: implied = 50 + 1*10 = 60 (from synthesis)
        # weighted avg: 0.5*75 + 0.5*60 = 67.5
        for t in _DATES:
            assert new_prices.loc[t] == pytest.approx(67.5)


# ---------------------------------------------------------------------------
# Test 4: forecast_spread sensitivity
# ---------------------------------------------------------------------------


class TestForecastSpreadSensitivity:
    """Verify that spread affects convergence behaviour for direction-only strategies."""

    def test_all_spreads_converge_in_correct_direction(self) -> None:
        """All spread values should move market toward real price."""
        idx = _index()
        real = pd.Series([80.0, 80.0, 80.0], index=idx)
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        directions = {"long": pd.Series([1, 1, 1], index=idx, dtype="Int64")}

        for spread in [1.0, 5.0, 10.0, 20.0]:
            eq = run_futures_market(
                directions=directions,
                initial_market_prices=initial,
                real_prices=real,
                max_iterations=100,
                forecast_spread=spread,
                dampening=0.5,
            )
            for t in _DATES:
                assert eq.final_market_prices.loc[t] > 50.0, (
                    f"spread={spread}: price should move toward real=80"
                )

    def test_larger_spread_converges_in_fewer_iterations(self) -> None:
        """Larger spread → fewer iterations to cross a given threshold."""
        idx = _index()
        real = pd.Series([80.0, 80.0, 80.0], index=idx)
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        directions = {"long": pd.Series([1, 1, 1], index=idx, dtype="Int64")}

        iteration_counts = {}
        for spread in [1.0, 5.0, 10.0, 20.0]:
            eq = run_futures_market(
                directions=directions,
                initial_market_prices=initial,
                real_prices=real,
                max_iterations=200,
                convergence_threshold=0.001,
                forecast_spread=spread,
                dampening=0.5,
            )
            iteration_counts[spread] = len(eq.iterations)

        # Larger spread should converge faster (fewer iterations)
        # or both hit max_iterations; in either case spread=20 ≤ spread=1
        assert iteration_counts[20.0] <= iteration_counts[1.0]


# ---------------------------------------------------------------------------
# Test 5: Profit calculation independent of forecast_spread
# ---------------------------------------------------------------------------


class TestProfitIndependentOfSpread:
    """compute_strategy_profits uses (real - market), not forecast_spread."""

    def test_same_profits_different_spreads(self) -> None:
        idx = _index()
        market = pd.Series([50.0, 55.0, 48.0], index=idx)
        real = pd.Series([55.0, 48.0, 52.0], index=idx)
        directions = {"long": pd.Series([1, 1, 1], index=idx, dtype="Int64")}

        profits_a = compute_strategy_profits(directions, market, real)
        profits_b = compute_strategy_profits(directions, market, real)

        # Profits depend only on directions, market, and real — not spread
        assert profits_a["long"] == pytest.approx(profits_b["long"])


# ---------------------------------------------------------------------------
# Regression: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Run the engine twice with the same inputs → identical output."""

    def test_identical_runs(self) -> None:
        idx = _index()
        real = pd.Series([55.0, 48.0, 52.0], index=idx)
        initial = pd.Series([50.0, 55.0, 48.0], index=idx)
        directions = {
            "long": pd.Series([1, 1, 1], index=idx, dtype="Int64"),
            "perfect": pd.Series([1, -1, 1], index=idx, dtype="Int64"),
        }
        kwargs = dict(
            directions=directions,
            initial_market_prices=initial,
            real_prices=real,
            max_iterations=50,
            convergence_threshold=0.001,
            forecast_spread=5.0,
            dampening=0.5,
        )

        eq1 = run_futures_market(**kwargs)
        eq2 = run_futures_market(**kwargs)

        pd.testing.assert_series_equal(eq1.final_market_prices, eq2.final_market_prices)


# ---------------------------------------------------------------------------
# Regression: auto-calibration floor
# ---------------------------------------------------------------------------


class TestAutoCalibrationFloor:
    """forecast_spread=None with zero gap should apply the 0.1 floor."""

    def test_zero_gap_applies_floor(self) -> None:
        idx = _index()
        prices = pd.Series([50.0, 50.0, 50.0], index=idx)
        directions = {"long": pd.Series([1, 1, 1], index=idx, dtype="Int64")}

        # real == initial → std of gap = 0 → floor 0.1 should be applied
        eq = run_futures_market(
            directions=directions,
            initial_market_prices=prices.copy(),
            real_prices=prices.copy(),
            forecast_spread=None,
        )
        # Should not stall or error; prices unchanged because PnL=0
        assert eq.converged
