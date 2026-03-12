"""Tests for market_simulation.market."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import DayState, Settlement, Trade


def _make_hourly_csv(path: Path, days: int = 5) -> Path:
    """Create a minimal hourly CSV for market environment tests."""
    timestamps = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(99)
    # Use deterministic prices: each day has a distinct mean
    prices = []
    for d in range(days):
        day_mean = 40.0 + d * 5.0
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


class TestMarketEnvironmentInit:
    """Tests for MarketEnvironment construction."""

    def test_creates_from_csv(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path)
        env = MarketEnvironment(csv_path)
        assert env is not None

    def test_default_date_range(self, tmp_path: Path) -> None:
        """Without explicit dates, should cover the full range minus first day."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        dates = env.delivery_dates
        # First delivery day needs a prior settlement, so starts at day 2
        assert dates[0] == date(2024, 1, 2)
        assert dates[-1] == date(2024, 1, 5)

    def test_custom_date_range(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(
            csv_path,
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 4),
        )
        dates = env.delivery_dates
        assert dates == [date(2024, 1, 3), date(2024, 1, 4)]


class TestMarketEnvironmentGetState:
    """Tests for MarketEnvironment.get_state()."""

    def test_returns_day_state(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        state = env.get_state(date(2024, 1, 2))
        assert isinstance(state, DayState)

    def test_delivery_date_matches(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        target = date(2024, 1, 3)
        state = env.get_state(target)
        assert state.delivery_date == target

    def test_last_settlement_is_prior_day(self, tmp_path: Path) -> None:
        """The last_settlement_price should be from the day before."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        state = env.get_state(date(2024, 1, 3))
        # Should be the settlement for Jan 2, not Jan 3
        assert isinstance(state.last_settlement_price, float)

    def test_features_are_lagged(self, tmp_path: Path) -> None:
        """Features should not contain data from the delivery day itself."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        state = env.get_state(date(2024, 1, 3))
        assert isinstance(state.features, pd.DataFrame)
        assert len(state.features) == 1

    def test_neighbor_prices_populated(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        state = env.get_state(date(2024, 1, 3))
        assert isinstance(state.neighbor_prices, dict)
        assert len(state.neighbor_prices) > 0

    def test_invalid_date_raises(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        with pytest.raises(ValueError, match="outside"):
            env.get_state(date(2099, 1, 1))


class TestMarketEnvironmentSettle:
    """Tests for MarketEnvironment.settle()."""

    def test_returns_settlement(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        trade = Trade(
            delivery_date=date(2024, 1, 2),
            entry_price=45.0,
            position_mw=1.0,
        )
        result = env.settle(trade)
        assert isinstance(result, Settlement)

    def test_settlement_price_is_day_mean(self, tmp_path: Path) -> None:
        """Settled price should equal the day's average hourly price."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        trade = Trade(
            delivery_date=date(2024, 1, 2),
            entry_price=45.0,
            position_mw=1.0,
        )
        result = env.settle(trade)
        assert isinstance(result.settlement_price, float)
        assert result.settlement_price != 0.0

    def test_pnl_computation(self, tmp_path: Path) -> None:
        """PnL should follow (settlement - entry) * position * hours."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        trade = Trade(
            delivery_date=date(2024, 1, 2),
            entry_price=45.0,
            position_mw=1.0,
            hours=24,
        )
        result = env.settle(trade)
        expected_pnl = (result.settlement_price - 45.0) * 1.0 * 24
        assert result.pnl == pytest.approx(expected_pnl)

    def test_invalid_delivery_date_raises(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        trade = Trade(
            delivery_date=date(2099, 1, 1),
            entry_price=45.0,
            position_mw=1.0,
        )
        with pytest.raises(ValueError):
            env.settle(trade)


class TestMarketEnvironmentSettlementPrices:
    """Tests for MarketEnvironment.settlement_prices property."""

    def test_returns_series(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        prices = env.settlement_prices
        assert isinstance(prices, pd.Series)

    def test_covers_all_dataset_days(self, tmp_path: Path) -> None:
        """settlement_prices covers the full dataset, not just the sim range."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        prices = env.settlement_prices
        assert len(prices) == 5

    def test_index_contains_delivery_dates(self, tmp_path: Path) -> None:
        """Every simulation delivery date must be present in settlement_prices."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        prices = env.settlement_prices
        for d in env.delivery_dates:
            assert d in prices.index, f"{d} missing from settlement_prices"

    def test_values_are_floats(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        prices = env.settlement_prices
        assert all(isinstance(v, float) for v in prices.values)

    def test_returns_copy(self, tmp_path: Path) -> None:
        """Mutating the returned series must not affect the market's internal state."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        prices = env.settlement_prices
        original_value = float(prices.iloc[0])
        prices.iloc[0] = 9999.0
        fresh = env.settlement_prices
        assert float(fresh.iloc[0]) == pytest.approx(original_value)

    def test_consistent_with_get_state_last_settlement(self, tmp_path: Path) -> None:
        """settlement_prices[prior_date] must equal state.last_settlement_price."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        env = MarketEnvironment(csv_path)
        prices = env.settlement_prices
        # Jan 3 state should have last_settlement = settlement of Jan 2
        state = env.get_state(date(2024, 1, 3))
        assert prices[date(2024, 1, 2)] == pytest.approx(state.last_settlement_price)
