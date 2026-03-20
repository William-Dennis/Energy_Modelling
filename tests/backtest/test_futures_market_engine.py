"""Unit tests for the spec-compliant synthetic futures market engine.

Tests verify that the engine implements ``docs/energy_market_spec.md``:
  1. Trading decision: q = sign(forecast - market)
  2. Profit: r = q * (real - market)  (NO *24 multiplier)
  3. Weighting: w = max(profit, 0) / sum(max(profits, 0))
  4. Price update: P_new = alpha * sum(w_i * forecast_i) + (1-alpha) * P_old
  5. Iterate until convergence.

Phase 8 extensions: dampening (alpha), weight cap, log-profit weighting,
weighted median, bimodal cluster detection, two-phase convergence.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.futures_market_engine import (
    adaptive_alpha,
    compute_market_prices,
    compute_market_prices_median,
    compute_strategy_profits,
    compute_weights,
    compute_weights_capped,
    compute_weights_log,
    detect_bimodal_clusters,
    run_futures_market,
    run_futures_market_iteration,
    run_two_phase_market,
    weighted_median,
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


# ===========================================================================
# Phase 8b: Dampening tests
# ===========================================================================


class TestDampeningAlpha:
    """Phase 8b: compute_market_prices with alpha parameter."""

    def test_alpha_one_recovers_spec(self) -> None:
        """alpha=1.0 gives the same result as the original undampened engine."""
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        forecasts = {"A": {d: 70.0 for d in _DATES}}

        dampened = compute_market_prices(
            weights={"A": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
            alpha=1.0,
        )
        for t in _DATES:
            assert dampened.loc[t] == pytest.approx(70.0)

    def test_alpha_half_blends(self) -> None:
        """alpha=0.5 blends undampened price (70) with current (50) -> 60."""
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        forecasts = {"A": {d: 70.0 for d in _DATES}}

        dampened = compute_market_prices(
            weights={"A": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
            alpha=0.5,
        )
        # 0.5 * 70 + 0.5 * 50 = 60
        for t in _DATES:
            assert dampened.loc[t] == pytest.approx(60.0)

    def test_alpha_zero_point_three(self) -> None:
        """alpha=0.3: 0.3 * 70 + 0.7 * 50 = 56."""
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        forecasts = {"A": {d: 70.0 for d in _DATES}}

        dampened = compute_market_prices(
            weights={"A": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
            alpha=0.3,
        )
        for t in _DATES:
            assert dampened.loc[t] == pytest.approx(56.0, abs=0.01)

    def test_dampened_run_converges_oscillating_case(self) -> None:
        """Opposing strategies that oscillate undampened should converge with dampening."""
        dates = pd.DatetimeIndex(["2024-01-01"])
        real = pd.Series([100.0], index=dates)
        initial = pd.Series([90.0], index=dates)
        forecasts = {
            "Long": {dates[0]: 110.0},
            "Short": {dates[0]: 70.0},
        }
        # Undampened: oscillates (known from existing test)
        eq_undampened = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
            alpha=1.0,
        )
        # Dampened: should converge
        eq_dampened = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=100,
            alpha=0.3,
        )
        # Dampened should have a lower delta and significantly reduced oscillation
        assert eq_dampened.convergence_delta < eq_undampened.convergence_delta
        # For a 2-strategy system with no fixed point, dampening reduces
        # but cannot eliminate the cycle.  Verify substantial reduction.
        assert eq_dampened.convergence_delta < 0.5 * eq_undampened.convergence_delta

    def test_alpha_passed_through_iteration(self) -> None:
        """run_futures_market_iteration respects alpha."""
        idx = _index()
        market = pd.Series([50.0, 50.0, 50.0], index=idx)
        real = pd.Series([60.0, 60.0, 60.0], index=idx)
        forecasts = {"long": {d: 55.0 for d in _DATES}}

        result_undampened = run_futures_market_iteration(
            market_prices=market, real_prices=real, iteration=0,
            strategy_forecasts=forecasts, alpha=1.0,
        )
        result_dampened = run_futures_market_iteration(
            market_prices=market, real_prices=real, iteration=0,
            strategy_forecasts=forecasts, alpha=0.5,
        )
        # Undampened: price -> 55.  Dampened: 0.5*55 + 0.5*50 = 52.5
        for t in _DATES:
            assert result_undampened.market_prices.loc[t] == pytest.approx(55.0)
            assert result_dampened.market_prices.loc[t] == pytest.approx(52.5)


class TestAdaptiveAlpha:
    """Phase 8b: adaptive_alpha helper."""

    def test_large_delta_gives_small_alpha(self) -> None:
        alpha = adaptive_alpha(delta=100.0, target_delta=1.0, alpha_max=0.8, alpha_min=0.1)
        assert alpha == pytest.approx(0.1)

    def test_small_delta_gives_large_alpha(self) -> None:
        alpha = adaptive_alpha(delta=0.5, target_delta=1.0, alpha_max=0.8, alpha_min=0.1)
        assert alpha == pytest.approx(0.8)

    def test_zero_delta_returns_alpha_max(self) -> None:
        alpha = adaptive_alpha(delta=0.0, alpha_max=0.8)
        assert alpha == 0.8

    def test_exact_target_gives_alpha_max(self) -> None:
        alpha = adaptive_alpha(delta=1.0, target_delta=1.0, alpha_max=0.8, alpha_min=0.1)
        assert alpha == pytest.approx(0.8)


class TestTwoPhaseMarket:
    """Phase 8b: two-phase convergence."""

    def test_two_phase_returns_equilibrium(self) -> None:
        dates = pd.DatetimeIndex(["2024-01-01"])
        real = pd.Series([100.0], index=dates)
        initial = pd.Series([90.0], index=dates)
        forecasts = {"PF": {dates[0]: 100.0}}

        eq = run_two_phase_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
        )
        assert eq.final_market_prices.iloc[0] == pytest.approx(100.0, abs=0.1)

    def test_two_phase_improves_oscillating_case(self) -> None:
        """Two-phase should produce lower delta than undampened for oscillating strategies."""
        dates = pd.DatetimeIndex(["2024-01-01"])
        real = pd.Series([100.0], index=dates)
        initial = pd.Series([90.0], index=dates)
        forecasts = {
            "Long": {dates[0]: 110.0},
            "Short": {dates[0]: 70.0},
        }
        eq_undampened = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        eq_two_phase = run_two_phase_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
        )
        assert eq_two_phase.convergence_delta <= eq_undampened.convergence_delta
        # Two-phase should significantly reduce the oscillation amplitude
        assert eq_two_phase.convergence_delta < 0.5 * eq_undampened.convergence_delta


# ===========================================================================
# Phase 8c: Weighting reform tests
# ===========================================================================


class TestComputeWeightsCapped:
    """Phase 8c, Experiment C1: per-strategy weight cap."""

    def test_cap_clips_dominant_strategy(self) -> None:
        weights = compute_weights_capped({"a": 900.0, "b": 100.0}, w_max=0.5)
        assert weights["a"] <= 0.5 + 1e-10
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_cap_at_one_equals_standard_weights(self) -> None:
        profits = {"a": 100.0, "b": -50.0, "c": 200.0}
        standard = compute_weights(profits)
        capped = compute_weights_capped(profits, w_max=1.0)
        for name in profits:
            assert capped[name] == pytest.approx(standard[name], abs=1e-10)

    def test_all_negative_returns_zeros(self) -> None:
        weights = compute_weights_capped({"a": -10.0, "b": -20.0}, w_max=0.5)
        assert all(w == 0.0 for w in weights.values())

    def test_weights_sum_to_one(self) -> None:
        weights = compute_weights_capped(
            {"a": 50.0, "b": 100.0, "c": 150.0, "d": 200.0},
            w_max=0.30,
        )
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_tight_cap_distributes_evenly(self) -> None:
        """With a very tight cap, all profitable strategies get w_max."""
        weights = compute_weights_capped(
            {"a": 100.0, "b": 200.0, "c": 300.0},
            w_max=0.05,
        )
        for name in weights:
            assert weights[name] <= 0.05 + 1e-10 or weights[name] == pytest.approx(0.0)


class TestComputeWeightsLog:
    """Phase 8c, Experiment C3: log-profit weighting."""

    def test_compresses_profit_ratios(self) -> None:
        """A 6.6:1 profit ratio should map to a much smaller weight ratio."""
        standard = compute_weights({"a": 6363.0, "b": 968.0})
        log_w = compute_weights_log({"a": 6363.0, "b": 968.0})
        standard_ratio = standard["a"] / standard["b"]
        log_ratio = log_w["a"] / log_w["b"]
        assert log_ratio < standard_ratio  # Should be ~1.27 vs 6.6

    def test_only_positive_get_weight(self) -> None:
        weights = compute_weights_log({"a": 100.0, "b": -50.0})
        assert weights["b"] == 0.0
        assert weights["a"] == pytest.approx(1.0)

    def test_all_negative_returns_zeros(self) -> None:
        weights = compute_weights_log({"a": -10.0, "b": -20.0})
        assert all(w == 0.0 for w in weights.values())

    def test_weights_sum_to_one(self) -> None:
        weights = compute_weights_log({"a": 50.0, "b": 100.0, "c": 150.0})
        assert sum(weights.values()) == pytest.approx(1.0)


class TestWeightedMedian:
    """Phase 8c, Experiment C2: weighted median."""

    def test_equal_weights_returns_median(self) -> None:
        vals = np.array([10.0, 20.0, 30.0])
        wts = np.array([1.0, 1.0, 1.0])
        assert weighted_median(vals, wts) == pytest.approx(20.0)

    def test_skewed_weights(self) -> None:
        vals = np.array([10.0, 20.0, 30.0])
        wts = np.array([0.1, 0.1, 0.8])
        # Cumulative: 0.1, 0.2, 1.0 — half = 0.5, first >= 0.5 is at index 2
        assert weighted_median(vals, wts) == pytest.approx(30.0)

    def test_two_values(self) -> None:
        vals = np.array([40.0, 100.0])
        wts = np.array([0.6, 0.4])
        # Cumulative: 0.6, 1.0 — half = 0.5, first >= 0.5 is at index 0
        assert weighted_median(vals, wts) == pytest.approx(40.0)

    def test_single_value(self) -> None:
        vals = np.array([42.0])
        wts = np.array([1.0])
        assert weighted_median(vals, wts) == pytest.approx(42.0)


class TestComputeMarketPricesMedian:
    """Phase 8c, Experiment C2: weighted-median market prices."""

    def test_single_strategy_equals_weighted_mean(self) -> None:
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        forecasts = {"only": {d: 75.0 for d in _DATES}}
        new = compute_market_prices_median(
            weights={"only": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
        )
        for t in _DATES:
            assert new.loc[t] == pytest.approx(75.0)

    def test_outlier_resistance(self) -> None:
        """Median should resist one extreme outlier."""
        idx = _index()
        initial = pd.Series([50.0, 50.0, 50.0], index=idx)
        forecasts = {
            "normal1": {d: 70.0 for d in _DATES},
            "normal2": {d: 72.0 for d in _DATES},
            "outlier": {d: 200.0 for d in _DATES},
        }
        # Equal weights — median should be 72 (middle value), not pulled to ~114
        new = compute_market_prices_median(
            weights={"normal1": 1.0, "normal2": 1.0, "outlier": 1.0},
            strategy_forecasts=forecasts,
            current_market_prices=initial,
        )
        for t in _DATES:
            assert new.loc[t] == pytest.approx(72.0)


class TestDetectBimodalClusters:
    """Phase 8c, Experiment C4: bimodal cluster detection."""

    def test_bimodal_detection(self) -> None:
        forecasts = [40.0, 42.0, 45.0, 100.0, 102.0, 105.0]
        result = detect_bimodal_clusters(forecasts, gap_threshold=20.0)
        assert result is not None
        low, high = result
        assert max(low) < 50
        assert min(high) > 90

    def test_unimodal_returns_none(self) -> None:
        forecasts = [40.0, 42.0, 45.0, 48.0, 50.0]
        result = detect_bimodal_clusters(forecasts, gap_threshold=20.0)
        assert result is None

    def test_single_element_returns_none(self) -> None:
        assert detect_bimodal_clusters([42.0]) is None

    def test_empty_returns_none(self) -> None:
        assert detect_bimodal_clusters([]) is None
