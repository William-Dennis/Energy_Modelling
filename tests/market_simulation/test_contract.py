"""Tests for market_simulation.contract."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.market_simulation.contract import compute_pnl, compute_settlement_price
from energy_modelling.market_simulation.types import Trade


class TestComputeSettlementPrice:
    """Tests for compute_settlement_price()."""

    def test_uniform_prices(self) -> None:
        """All hours at the same price gives that price."""
        prices = pd.Series([50.0] * 24)
        assert compute_settlement_price(prices) == 50.0

    def test_mean_of_varying_prices(self) -> None:
        """Settlement is the arithmetic mean of hourly prices."""
        prices = pd.Series([10.0, 20.0, 30.0, 40.0])
        assert compute_settlement_price(prices) == 25.0

    def test_handles_negative_prices(self) -> None:
        """Negative prices are common in renewable-heavy hours."""
        prices = pd.Series([-50.0, 50.0])
        assert compute_settlement_price(prices) == 0.0

    def test_single_price(self) -> None:
        """A single observation should return that value."""
        prices = pd.Series([42.5])
        assert compute_settlement_price(prices) == 42.5

    def test_empty_series_raises(self) -> None:
        """Empty series should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_settlement_price(pd.Series([], dtype=float))

    def test_all_nan_raises(self) -> None:
        """All-NaN series should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_settlement_price(pd.Series([np.nan, np.nan]))

    def test_nan_ignored(self) -> None:
        """NaN values should be excluded from the mean."""
        prices = pd.Series([10.0, np.nan, 30.0])
        assert compute_settlement_price(prices) == 20.0


class TestComputePnl:
    """Tests for compute_pnl()."""

    def test_long_profit(self) -> None:
        """Long position profits when settlement > entry."""
        trade = Trade(
            delivery_date=date(2024, 1, 15),
            entry_price=50.0,
            position_mw=1.0,
            hours=24,
        )
        pnl = compute_pnl(trade, settlement_price=60.0)
        assert pnl == pytest.approx(240.0)  # (60 - 50) * 1 * 24

    def test_long_loss(self) -> None:
        """Long position loses when settlement < entry."""
        trade = Trade(
            delivery_date=date(2024, 1, 15),
            entry_price=60.0,
            position_mw=1.0,
            hours=24,
        )
        pnl = compute_pnl(trade, settlement_price=50.0)
        assert pnl == pytest.approx(-240.0)

    def test_short_profit(self) -> None:
        """Short position profits when settlement < entry."""
        trade = Trade(
            delivery_date=date(2024, 1, 15),
            entry_price=60.0,
            position_mw=-1.0,
            hours=24,
        )
        pnl = compute_pnl(trade, settlement_price=50.0)
        assert pnl == pytest.approx(240.0)  # (50 - 60) * -1 * 24

    def test_zero_pnl_when_flat(self) -> None:
        """PnL is zero when settlement equals entry."""
        trade = Trade(
            delivery_date=date(2024, 1, 15),
            entry_price=50.0,
            position_mw=1.0,
            hours=24,
        )
        pnl = compute_pnl(trade, settlement_price=50.0)
        assert pnl == pytest.approx(0.0)

    def test_peak_hours(self) -> None:
        """Peak contract with 12 hours gives half the base PnL."""
        trade = Trade(
            delivery_date=date(2024, 1, 15),
            entry_price=50.0,
            position_mw=1.0,
            hours=12,
        )
        pnl = compute_pnl(trade, settlement_price=60.0)
        assert pnl == pytest.approx(120.0)  # (60 - 50) * 1 * 12
