"""Tests for strategy.runner."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import DayState, Signal, Trade
from energy_modelling.strategy.base import Strategy
from energy_modelling.strategy.naive_copy import NaiveCopyStrategy
from energy_modelling.strategy.perfect_foresight import PerfectForesightStrategy
from energy_modelling.strategy.runner import BacktestResult, BacktestRunner


def _make_hourly_csv(path: Path, days: int = 10) -> Path:
    """Create a minimal hourly CSV for runner tests."""
    timestamps = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(77)
    prices = []
    for d in range(days):
        day_mean = 40.0 + d * 3.0
        prices.extend([day_mean + rng.normal(0, 2) for _ in range(24)])
    data = {
        "timestamp_utc": timestamps,
        "price_eur_mwh": prices,
        "gen_solar_mw": rng.uniform(0, 5000, len(timestamps)),
        "gen_wind_onshore_mw": rng.uniform(0, 20000, len(timestamps)),
        "gen_wind_offshore_mw": rng.uniform(0, 6000, len(timestamps)),
        "gen_fossil_gas_mw": rng.uniform(1000, 10000, len(timestamps)),
        "gen_fossil_hard_coal_mw": rng.uniform(500, 8000, len(timestamps)),
        "gen_fossil_brown_coal_lignite_mw": rng.uniform(2000, 15000, len(timestamps)),
        "gen_nuclear_mw": rng.uniform(0, 9000, len(timestamps)),
        "load_actual_mw": rng.uniform(35000, 70000, len(timestamps)),
        "load_forecast_mw": rng.uniform(35000, 70000, len(timestamps)),
        "forecast_solar_mw": rng.uniform(0, 5000, len(timestamps)),
        "forecast_wind_onshore_mw": rng.uniform(0, 20000, len(timestamps)),
        "forecast_wind_offshore_mw": rng.uniform(0, 6000, len(timestamps)),
        "weather_temperature_2m_degc": rng.normal(10, 5, len(timestamps)),
        "weather_wind_speed_10m_kmh": rng.uniform(0, 50, len(timestamps)),
        "weather_shortwave_radiation_wm2": rng.uniform(0, 800, len(timestamps)),
        "price_fr_eur_mwh": rng.normal(55, 15, len(timestamps)),
        "price_nl_eur_mwh": rng.normal(52, 12, len(timestamps)),
        "price_at_eur_mwh": rng.normal(50, 10, len(timestamps)),
        "price_pl_eur_mwh": rng.normal(60, 20, len(timestamps)),
        "price_cz_eur_mwh": rng.normal(48, 10, len(timestamps)),
        "price_dk_1_eur_mwh": rng.normal(45, 15, len(timestamps)),
        "flow_fr_net_import_mw": rng.normal(-500, 1000, len(timestamps)),
        "flow_nl_net_import_mw": rng.normal(-1000, 800, len(timestamps)),
        "carbon_price_usd": rng.uniform(20, 100, len(timestamps)),
        "gas_price_usd": rng.uniform(15, 50, len(timestamps)),
    }
    df = pd.DataFrame(data)
    csv_path = path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class _SkipStrategy(Strategy):
    """A strategy that skips every trade (returns None)."""

    def act(self, state: DayState) -> Signal | None:
        return None


class _ShortStrategy(Strategy):
    """A strategy that always signals short (-1)."""

    def act(self, state: DayState) -> Signal | None:
        return Signal(delivery_date=state.delivery_date, direction=-1)


class TestBacktestRunner:
    """Tests for BacktestRunner."""

    def test_run_returns_backtest_result(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()
        assert isinstance(result, BacktestResult)

    def test_result_has_settlements(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()
        # 5 days minus 1 (first day has no prior settlement) = 4 trades
        assert len(result.settlements) == 4

    def test_daily_pnl_length(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()
        assert len(result.daily_pnl) == 4

    def test_cumulative_pnl_is_cumsum(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()
        pd.testing.assert_series_equal(
            result.cumulative_pnl,
            result.daily_pnl.cumsum(),
        )

    def test_skip_strategy_gives_empty(self, tmp_path: Path) -> None:
        """A strategy that always returns None should yield no settlements."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = _SkipStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()
        assert len(result.settlements) == 0
        assert len(result.daily_pnl) == 0

    def test_strategy_reset_called(self, tmp_path: Path) -> None:
        """Runner should call strategy.reset() before starting."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        # Monkey-patch reset to track calls
        reset_called = []
        original_reset = strategy.reset

        def tracked_reset() -> None:
            reset_called.append(True)
            original_reset()

        strategy.reset = tracked_reset  # type: ignore[assignment]
        runner = BacktestRunner(market, strategy)
        runner.run()
        assert len(reset_called) == 1


class TestRunnerEntryPriceInvariant:
    """Tests that entry price is always fixed to the prior day's DA settlement.

    This is the core correctness guarantee of the refactored architecture.
    Strategies emit a Signal (direction only); the runner constructs the Trade
    using state.last_settlement_price as entry price.
    """

    def test_entry_price_equals_prior_day_settlement(self, tmp_path: Path) -> None:
        """Every trade's entry price must equal the prior day's settlement."""
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        # Build a map of date → settlement from the market
        # (same as what the runner sees via DayState.last_settlement_price)
        delivery_dates = market.delivery_dates
        for settlement in result.settlements:
            trade = settlement.trade
            state = market.get_state(trade.delivery_date)
            assert trade.entry_price == pytest.approx(state.last_settlement_price), (
                f"On {trade.delivery_date}: entry_price {trade.entry_price} != "
                f"last_settlement {state.last_settlement_price}"
            )

    def test_entry_price_equals_prior_day_settlement_short_strategy(self, tmp_path: Path) -> None:
        """Entry price invariant holds for short signals too."""
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        strategy = _ShortStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            trade = settlement.trade
            state = market.get_state(trade.delivery_date)
            assert trade.entry_price == pytest.approx(state.last_settlement_price)

    def test_entry_price_invariant_with_perfect_foresight(self, tmp_path: Path) -> None:
        """Entry price invariant holds for PerfectForesightStrategy too."""
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        from energy_modelling.market_simulation.data import (
            compute_daily_settlement,
            load_dataset,
        )

        df = load_dataset(csv_path)
        settlements_map = compute_daily_settlement(df)
        strategy = PerfectForesightStrategy(settlements_map)
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            trade = settlement.trade
            state = market.get_state(trade.delivery_date)
            assert trade.entry_price == pytest.approx(state.last_settlement_price)


class TestRunnerQuantityInvariant:
    """Tests that position size is always fixed at 1 MW."""

    def test_position_size_is_1mw_long(self, tmp_path: Path) -> None:
        """Long signals produce position_mw = +1.0."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()  # always long
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            assert settlement.trade.position_mw == pytest.approx(1.0), (
                f"Expected 1.0 MW, got {settlement.trade.position_mw}"
            )

    def test_position_size_is_1mw_short(self, tmp_path: Path) -> None:
        """Short signals produce position_mw = -1.0."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = _ShortStrategy()  # always short
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            assert settlement.trade.position_mw == pytest.approx(-1.0), (
                f"Expected -1.0 MW, got {settlement.trade.position_mw}"
            )

    def test_absolute_position_always_1mw(self, tmp_path: Path) -> None:
        """Regardless of direction, |position_mw| must always be 1.0."""
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        from energy_modelling.market_simulation.data import (
            compute_daily_settlement,
            load_dataset,
        )

        df = load_dataset(csv_path)
        settlements_map = compute_daily_settlement(df)
        strategy = PerfectForesightStrategy(settlements_map)  # mixes long and short
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            assert abs(settlement.trade.position_mw) == pytest.approx(1.0)


class TestRunnerDirectionMatchesSignal:
    """Tests that the trade direction matches the strategy's signal."""

    def test_long_signal_produces_positive_position(self, tmp_path: Path) -> None:
        """Signal direction +1 must produce position_mw > 0."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()  # always direction=+1
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            assert settlement.trade.position_mw > 0

    def test_short_signal_produces_negative_position(self, tmp_path: Path) -> None:
        """Signal direction -1 must produce position_mw < 0."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = _ShortStrategy()  # always direction=-1
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            assert settlement.trade.position_mw < 0

    def test_sign_of_position_matches_direction(self, tmp_path: Path) -> None:
        """For perfect foresight, sign(position_mw) == signal.direction for
        every trade, validating that the runner correctly maps signal to trade."""
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        from energy_modelling.market_simulation.data import (
            compute_daily_settlement,
            load_dataset,
        )

        df = load_dataset(csv_path)
        settlements_map = compute_daily_settlement(df)
        strategy = PerfectForesightStrategy(settlements_map)

        # Run with tracking: capture the signals emitted by the strategy
        signals_emitted: dict[date, int] = {}
        original_act = strategy.act

        def tracked_act(state: DayState) -> Signal | None:
            signal = original_act(state)
            if signal is not None:
                signals_emitted[signal.delivery_date] = signal.direction
            return signal

        strategy.act = tracked_act  # type: ignore[method-assign]

        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            trade = settlement.trade
            expected_direction = signals_emitted[trade.delivery_date]
            actual_sign = 1 if trade.position_mw > 0 else -1
            assert actual_sign == expected_direction, (
                f"On {trade.delivery_date}: position sign {actual_sign} != "
                f"signal direction {expected_direction}"
            )


class TestRunnerHoursInvariant:
    """Tests that all trades use 24-hour base-load contracts."""

    def test_hours_always_24(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            assert settlement.trade.hours == 24


class TestRunnerPnlFormula:
    """Tests that PnL = (settlement - entry) * position_mw * 24 is correct."""

    def test_pnl_formula_correctness(self, tmp_path: Path) -> None:
        """Verify every settlement PnL matches the explicit formula."""
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for i, settlement in enumerate(result.settlements):
            trade = settlement.trade
            expected_pnl = (
                (settlement.settlement_price - trade.entry_price) * trade.position_mw * trade.hours
            )
            assert settlement.pnl == pytest.approx(expected_pnl), (
                f"Trade {i}: pnl {settlement.pnl} != expected {expected_pnl}"
            )

    def test_naive_copy_pnl_is_daily_price_change_times_24(self, tmp_path: Path) -> None:
        """For NaiveCopy (always long, 1 MW), PnL = (today - yesterday) * 24.

        This verifies the economic interpretation: the naive strategy earns
        the daily price change.
        """
        csv_path = _make_hourly_csv(tmp_path, days=5)
        market = MarketEnvironment(csv_path)
        strategy = NaiveCopyStrategy()
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            trade = settlement.trade
            state = market.get_state(trade.delivery_date)
            expected = (settlement.settlement_price - state.last_settlement_price) * 24.0
            assert settlement.pnl == pytest.approx(expected)

    def test_perfect_foresight_pnl_always_non_negative(self, tmp_path: Path) -> None:
        """PerfectForesight PnL is always >= 0 (it always picks the right side)."""
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        from energy_modelling.market_simulation.data import (
            compute_daily_settlement,
            load_dataset,
        )

        df = load_dataset(csv_path)
        settlements_map = compute_daily_settlement(df)
        strategy = PerfectForesightStrategy(settlements_map)
        runner = BacktestRunner(market, strategy)
        result = runner.run()

        for settlement in result.settlements:
            assert settlement.pnl >= 0.0, (
                f"PerfectForesight PnL was negative: {settlement.pnl} "
                f"on {settlement.trade.delivery_date}"
            )

    def test_perfect_foresight_pnl_dominates_naive(self, tmp_path: Path) -> None:
        """Total PnL of PerfectForesight >= |Total PnL of NaiveCopy|.

        Perfect foresight is the upper bound: its total PnL is the sum of
        absolute daily moves, which is always >= the sum of signed moves.
        """
        csv_path = _make_hourly_csv(tmp_path, days=10)
        market = MarketEnvironment(csv_path)
        from energy_modelling.market_simulation.data import (
            compute_daily_settlement,
            load_dataset,
        )

        df = load_dataset(csv_path)
        settlements_map = compute_daily_settlement(df)

        naive_runner = BacktestRunner(market, NaiveCopyStrategy())
        pf_runner = BacktestRunner(market, PerfectForesightStrategy(settlements_map))

        naive_result = naive_runner.run()
        pf_result = pf_runner.run()

        naive_total = naive_result.daily_pnl.sum()
        pf_total = pf_result.daily_pnl.sum()

        assert pf_total >= abs(naive_total), (
            f"PF total {pf_total} < |naive total| {abs(naive_total)}"
        )
