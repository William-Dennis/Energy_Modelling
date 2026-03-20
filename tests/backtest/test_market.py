"""Tests for convergence analysis module (Phase 7).

Tests the theoretical and empirical convergence properties of the
spec-compliant synthetic futures market under various strategy
configurations, including perfect foresight.

The spec model (``docs/energy_market_spec.md``) has:
  - NO dampening (price update is direct weighted average of forecasts)
  - NO direction ± spread synthesis
  - NO *24 profit multiplier in the engine
"""

from __future__ import annotations

import pandas as pd
import pytest

from energy_modelling.backtest.convergence import (
    ConvergenceTrajectory,
    compute_convergence_trajectory,
    run_forecast_foresight_market,
)
from energy_modelling.backtest.futures_market_engine import run_futures_market

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_day_setup(
    real: float = 100.0,
    initial: float = 90.0,
) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    """Create a single-day test setup."""
    dates = pd.DatetimeIndex(["2024-01-01"])
    real_prices = pd.Series([real], index=dates)
    initial_prices = pd.Series([initial], index=dates)
    return dates, real_prices, initial_prices


def _multi_day_setup() -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    """Create a multi-day test setup with mixed directions."""
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    real_prices = pd.Series([100.0, 80.0, 95.0], index=dates)
    initial_prices = pd.Series([90.0, 90.0, 90.0], index=dates)
    return dates, real_prices, initial_prices


def _build_pf_forecasts(real_prices: pd.Series) -> dict[str, dict]:
    """Build PF forecasts that equal real prices."""
    return {"PerfectForesight": {t: float(real_prices.loc[t]) for t in real_prices.index}}


def _build_constant_forecasts(
    dates: pd.DatetimeIndex,
    name: str,
    value: float,
) -> dict[str, dict]:
    """Build a strategy with constant forecast on all dates."""
    return {name: {t: value for t in dates}}


# ---------------------------------------------------------------------------
# compute_convergence_trajectory
# ---------------------------------------------------------------------------


class TestConvergenceTrajectory:
    """Test trajectory extraction from market equilibrium."""

    def test_returns_trajectory_object(self) -> None:
        dates, real, initial = _single_day_setup()
        forecasts = _build_pf_forecasts(real)
        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        traj = compute_convergence_trajectory(eq, real)
        assert isinstance(traj, ConvergenceTrajectory)

    def test_deltas_are_nonnegative(self) -> None:
        dates, real, initial = _single_day_setup()
        forecasts = _build_pf_forecasts(real)
        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        traj = compute_convergence_trajectory(eq, real)
        assert all(d >= 0 for d in traj.deltas)

    def test_final_rmse_zero_for_pf_only(self) -> None:
        """With only PF (no dampening), final RMSE should be ~0."""
        dates, real, initial = _single_day_setup()
        forecasts = _build_pf_forecasts(real)
        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        traj = compute_convergence_trajectory(eq, real)
        # No dampening => PF drives price to real in one step => RMSE ≈ 0
        assert traj.final_rmse < 0.01

    def test_multi_day_trajectory(self) -> None:
        dates, real, initial = _multi_day_setup()
        forecasts = _build_pf_forecasts(real)
        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        traj = compute_convergence_trajectory(eq, real)
        assert traj.converged
        assert traj.n_iterations <= 50
        assert traj.final_rmse < 0.01


# ---------------------------------------------------------------------------
# run_forecast_foresight_market (PF-only convergence)
# ---------------------------------------------------------------------------


class TestForecastForesightMarket:
    """Test convergence with PF providing real-price forecasts."""

    def test_converges_for_single_day(self) -> None:
        dates, real, initial = _single_day_setup(real=100.0, initial=90.0)
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=100,
        )
        assert eq.converged

    def test_converges_to_real_price_single_day(self) -> None:
        """With PF forecast = real price (no dampening), converges to P_real."""
        dates, real, initial = _single_day_setup(real=100.0, initial=90.0)
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=100,
        )
        assert eq.final_market_prices.iloc[0] == pytest.approx(100.0, abs=0.01)

    def test_converges_multi_day(self) -> None:
        dates, real, initial = _multi_day_setup()
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=100,
        )
        assert eq.converged
        for t in dates:
            assert eq.final_market_prices.loc[t] == pytest.approx(real.loc[t], abs=0.01)

    def test_instant_convergence_no_dampening(self) -> None:
        """Without dampening, PF-only market should converge in <=2 iterations."""
        dates, real, initial = _single_day_setup(real=100.0, initial=90.0)
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=50,
        )
        # Iteration 0: PF forecast=100, market=90, sign(100-90)=+1,
        #   profit = +1*(100-90) = 10 > 0 => weight=1.0
        #   New price = 1.0 * 100 = 100.
        # Iteration 1: sign(100-100)=0, profit=0 => weights all zero
        #   => price carries forward. Delta = 0 < threshold => converged.
        assert eq.converged
        assert len(eq.iterations) <= 2

    def test_no_oscillation_regardless_of_distance(self) -> None:
        """Without dampening, there's no oscillation."""
        dates, real, initial = _single_day_setup(real=100.0, initial=90.0)
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=200,
        )
        assert eq.converged
        assert eq.final_market_prices.iloc[0] == pytest.approx(100.0, abs=0.01)

    def test_large_price_gap_still_converges(self) -> None:
        """Even with a huge initial gap, PF converges in one step."""
        dates = pd.DatetimeIndex(["2024-01-01"])
        real = pd.Series([500.0], index=dates)
        initial = pd.Series([50.0], index=dates)
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=200,
        )
        assert eq.converged
        assert eq.final_market_prices.iloc[0] == pytest.approx(500.0, abs=0.01)


# ---------------------------------------------------------------------------
# PF with other strategies
# ---------------------------------------------------------------------------


class TestPFWithOtherStrategies:
    """PF mixed with other strategies: convergence and dominance."""

    def test_pf_dominates_wrong_strategy(self) -> None:
        """When PF competes with a consistently wrong strategy, PF gets all weight."""
        dates, real, initial = _single_day_setup(real=100.0, initial=90.0)

        # Wrong strategy forecasts 70 (below market of 90, goes short on a day
        # where real > market => loses money)
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=50,
            other_forecasts={"wrong": {dates[0]: 70.0}},
        )

        # PF should dominate
        last = eq.iterations[-1]
        assert last.strategy_weights.get("PerfectForesight", 0.0) >= last.strategy_weights.get(
            "wrong", 0.0
        )

    def test_pf_with_other_forecaster_converges(self) -> None:
        """PF + another forecaster still converges to a fixed point."""
        dates, real, initial = _multi_day_setup()

        other = {t: 95.0 for t in dates}  # constant forecast of 95
        eq = run_forecast_foresight_market(
            real_prices=real,
            initial_market_prices=initial,
            max_iterations=100,
            other_forecasts={"constant_95": other},
        )
        assert eq.converged

    def test_convergence_with_all_strategies_same_forecast(self) -> None:
        """When all strategies forecast the same value, price = that value."""
        dates = pd.DatetimeIndex(["2024-01-01"])
        real = pd.Series([100.0], index=dates)
        initial = pd.Series([90.0], index=dates)
        forecasts = {
            "a": {dates[0]: 95.0},
            "b": {dates[0]: 95.0},
        }

        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        # Both forecast 95, both have same direction, both have same profit
        # => equal weight => weighted avg = 95
        assert eq.final_market_prices.iloc[0] == pytest.approx(95.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration: market behaviour under various configurations
# ---------------------------------------------------------------------------


class TestMarketBehaviour:
    """Integration tests for market-level behaviour."""

    def test_opposing_strategies_oscillate_when_both_overshoot(self) -> None:
        """Long(110) vs Short(70) with real=100: oscillates because each
        forecast overshoots past real, handing profit to the other.

        Iter 0: Long wins (price→110). Iter 1: Short wins (price→70).
        This cycle repeats — the market does NOT converge.
        """
        dates = pd.DatetimeIndex(["2024-01-01"])
        real = pd.Series([100.0], index=dates)
        initial = pd.Series([90.0], index=dates)
        forecasts = {
            "Long": {dates[0]: 110.0},
            "Short": {dates[0]: 70.0},
        }
        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        # With both forecasts on opposite sides of real, market oscillates
        assert not eq.converged or len(eq.iterations) == 50

    def test_two_profitable_strategies_weighted_average(self) -> None:
        """When both strategies are profitable, price is their weighted avg."""
        dates = pd.DatetimeIndex(["2024-01-01"])
        real = pd.Series([100.0], index=dates)
        initial = pd.Series([80.0], index=dates)

        # Both forecast above market, both go long, both profitable
        forecasts = {
            "a": {dates[0]: 95.0},  # sign(95-80)=+1, profit = +1*(100-80)=20
            "b": {dates[0]: 105.0},  # sign(105-80)=+1, profit = +1*(100-80)=20
        }
        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=1,
        )
        # Equal profit => equal weight => price = (95+105)/2 = 100
        assert eq.iterations[0].market_prices.iloc[0] == pytest.approx(100.0, abs=0.01)

    def test_multi_day_mixed_directions(self) -> None:
        """Multi-day test: strategies have different forecasts per day."""
        dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
        real = pd.Series([110.0, 80.0], index=dates)
        initial = pd.Series([100.0, 100.0], index=dates)

        forecasts = {
            "long": {dates[0]: 115.0, dates[1]: 110.0},
            "short": {dates[0]: 85.0, dates[1]: 75.0},
        }
        eq = run_futures_market(
            initial_market_prices=initial,
            real_prices=real,
            strategy_forecasts=forecasts,
            max_iterations=50,
        )
        assert len(eq.iterations) > 0
        assert isinstance(eq.final_market_prices, pd.Series)
        assert len(eq.final_market_prices) == 2
