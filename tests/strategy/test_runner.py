"""Tests for strategy.runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import DayState, Trade
from energy_modelling.strategy.base import Strategy
from energy_modelling.strategy.naive_copy import NaiveCopyStrategy
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

    def act(self, state: DayState) -> Trade | None:
        return None


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
