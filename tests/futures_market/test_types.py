"""Tests for market_simulation.types -- specifically the Signal dataclass."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from energy_modelling.futures_market.types import DayState, Signal, Trade


_DATE = date(2024, 3, 15)


class TestSignalConstruction:
    """Tests for Signal dataclass construction and validation."""

    def test_creates_long_signal(self) -> None:
        signal = Signal(delivery_date=_DATE, direction=1)
        assert signal.delivery_date == _DATE
        assert signal.direction == 1

    def test_creates_short_signal(self) -> None:
        signal = Signal(delivery_date=_DATE, direction=-1)
        assert signal.delivery_date == _DATE
        assert signal.direction == -1

    def test_rejects_zero_direction(self) -> None:
        """direction=0 is not a valid signal."""
        with pytest.raises(ValueError, match="direction must be"):
            Signal(delivery_date=_DATE, direction=0)

    def test_rejects_direction_plus_2(self) -> None:
        """direction=2 is not a valid signal."""
        with pytest.raises(ValueError, match="direction must be"):
            Signal(delivery_date=_DATE, direction=2)

    def test_rejects_direction_minus_2(self) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            Signal(delivery_date=_DATE, direction=-2)

    def test_rejects_large_positive_direction(self) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            Signal(delivery_date=_DATE, direction=100)

    def test_is_frozen(self) -> None:
        """Signal is a frozen dataclass -- fields cannot be reassigned."""
        signal = Signal(delivery_date=_DATE, direction=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            signal.direction = -1  # type: ignore[misc]

    def test_frozen_delivery_date(self) -> None:
        signal = Signal(delivery_date=_DATE, direction=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            signal.delivery_date = date(2024, 6, 1)  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two Signal objects with same fields are equal."""
        a = Signal(delivery_date=_DATE, direction=1)
        b = Signal(delivery_date=_DATE, direction=1)
        assert a == b

    def test_inequality_direction(self) -> None:
        a = Signal(delivery_date=_DATE, direction=1)
        b = Signal(delivery_date=_DATE, direction=-1)
        assert a != b

    def test_inequality_date(self) -> None:
        a = Signal(delivery_date=_DATE, direction=1)
        b = Signal(delivery_date=date(2024, 3, 16), direction=1)
        assert a != b

    def test_hashable(self) -> None:
        """Frozen dataclasses must be hashable (usable as dict keys)."""
        signal = Signal(delivery_date=_DATE, direction=1)
        d = {signal: "test"}
        assert d[signal] == "test"

    def test_no_entry_price_field(self) -> None:
        """Signal must NOT have an entry_price field -- that is the market's
        responsibility."""
        signal = Signal(delivery_date=_DATE, direction=1)
        assert not hasattr(signal, "entry_price")

    def test_no_position_mw_field(self) -> None:
        """Signal must NOT have a position_mw field -- quantity is fixed by
        the runner."""
        signal = Signal(delivery_date=_DATE, direction=1)
        assert not hasattr(signal, "position_mw")

    def test_no_hours_field(self) -> None:
        """Signal must NOT have an hours field."""
        signal = Signal(delivery_date=_DATE, direction=1)
        assert not hasattr(signal, "hours")


class TestSignalRepr:
    """Tests for Signal string representation (useful for debugging)."""

    def test_repr_contains_direction(self) -> None:
        signal = Signal(delivery_date=_DATE, direction=1)
        assert "1" in repr(signal)

    def test_repr_contains_date(self) -> None:
        signal = Signal(delivery_date=_DATE, direction=-1)
        assert "2024" in repr(signal)


class TestSignalVsTradeSemantics:
    """Semantic tests verifying Signal and Trade have the right separation."""

    def test_signal_has_no_price_information(self) -> None:
        """Signal fields must be limited to delivery_date and direction.
        No price-related fields allowed."""
        signal = Signal(delivery_date=_DATE, direction=1)
        # Only these two fields should exist
        assert set(signal.__dataclass_fields__.keys()) == {"delivery_date", "direction"}

    def test_trade_preserves_price_fields(self) -> None:
        """Trade still carries entry_price -- it is set by the runner, not strategy."""
        trade = Trade(
            delivery_date=_DATE,
            entry_price=55.0,
            position_mw=1.0,
            hours=24,
        )
        assert trade.entry_price == pytest.approx(55.0)

    def test_runner_maps_direction_to_position_sign(self) -> None:
        """Verify the mapping the runner applies: direction +1 → position +1.0,
        direction -1 → position -1.0."""
        for direction in (1, -1):
            signal = Signal(delivery_date=_DATE, direction=direction)
            # Simulate what the challenge runner does
            position_mw = float(signal.direction) * 1.0
            assert position_mw == float(direction)
