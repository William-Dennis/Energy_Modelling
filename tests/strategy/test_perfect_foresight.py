"""Tests for strategy.perfect_foresight."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.market_simulation.contract import compute_pnl
from energy_modelling.market_simulation.types import DayState, Signal, Trade
from energy_modelling.strategy.perfect_foresight import PerfectForesightStrategy


def _make_day_state(
    delivery_date: date = date(2024, 1, 15),
    last_settlement: float = 50.0,
) -> DayState:
    """Create a minimal DayState for testing."""
    return DayState(
        delivery_date=delivery_date,
        last_settlement_price=last_settlement,
        features=pd.DataFrame({"load_actual_mw_mean": [55000.0]}, index=[delivery_date]),
        neighbor_prices={"FR": 55.0, "NL": 52.0},
    )


_DATE = date(2024, 1, 15)


class TestPerfectForesightStrategyInit:
    """Tests for PerfectForesightStrategy construction."""

    def test_accepts_dict(self) -> None:
        strategy = PerfectForesightStrategy({_DATE: 60.0})
        assert isinstance(strategy, PerfectForesightStrategy)

    def test_accepts_pandas_series(self) -> None:
        prices = pd.Series({_DATE: 60.0})
        strategy = PerfectForesightStrategy(prices)
        assert isinstance(strategy, PerfectForesightStrategy)

    def test_series_and_dict_equivalent(self) -> None:
        prices_dict = {_DATE: 60.0}
        prices_series = pd.Series(prices_dict)
        s_dict = PerfectForesightStrategy(prices_dict)
        s_series = PerfectForesightStrategy(prices_series)
        state = _make_day_state(last_settlement=50.0)
        assert s_dict.act(state) == s_series.act(state)


class TestPerfectForesightStrategyAct:
    """Tests for PerfectForesightStrategy.act()."""

    def test_returns_signal_instance(self) -> None:
        strategy = PerfectForesightStrategy({_DATE: 60.0})
        state = _make_day_state()
        signal = strategy.act(state)
        assert isinstance(signal, Signal)

    def test_goes_long_when_settlement_above_entry(self) -> None:
        """When true settlement > entry price, direction should be +1."""
        strategy = PerfectForesightStrategy({_DATE: 60.0})
        state = _make_day_state(last_settlement=50.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == 1

    def test_goes_short_when_settlement_below_entry(self) -> None:
        """When true settlement < entry price, direction should be -1."""
        strategy = PerfectForesightStrategy({_DATE: 40.0})
        state = _make_day_state(last_settlement=50.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == -1

    def test_skips_when_settlement_equals_entry(self) -> None:
        """When true settlement == entry price, act() should return None."""
        strategy = PerfectForesightStrategy({_DATE: 50.0})
        state = _make_day_state(last_settlement=50.0)
        signal = strategy.act(state)
        assert signal is None

    def test_delivery_date_matches_state(self) -> None:
        target = date(2024, 3, 10)
        strategy = PerfectForesightStrategy({target: 80.0})
        state = _make_day_state(delivery_date=target, last_settlement=50.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.delivery_date == target

    def test_raises_key_error_for_missing_date(self) -> None:
        """act() must raise KeyError if delivery date not in settlement prices."""
        strategy = PerfectForesightStrategy({})
        state = _make_day_state()
        with pytest.raises(KeyError):
            strategy.act(state)

    def test_negative_prices_long(self) -> None:
        """Works correctly when both settlement and entry are negative."""
        strategy = PerfectForesightStrategy({_DATE: -5.0})
        state = _make_day_state(last_settlement=-10.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == 1  # -5 > -10 → long

    def test_negative_prices_short(self) -> None:
        """Works correctly when true settlement is more negative."""
        strategy = PerfectForesightStrategy({_DATE: -15.0})
        state = _make_day_state(last_settlement=-10.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == -1  # -15 < -10 → short

    def test_no_entry_price_on_signal(self) -> None:
        """Signal must not expose an entry_price attribute -- entry price is
        the market's responsibility, not the strategy's."""
        strategy = PerfectForesightStrategy({_DATE: 70.0})
        state = _make_day_state(last_settlement=50.0)
        signal = strategy.act(state)
        assert signal is not None
        assert not hasattr(signal, "entry_price")

    def test_direction_only_long_and_short(self) -> None:
        """direction must only ever be +1 or -1 (never 0 or other values)."""
        entry = 50.0
        test_cases = [
            (60.0, 1),  # settlement above entry → long
            (40.0, -1),  # settlement below entry → short
            (50.1, 1),  # tiny upward move → long
            (49.9, -1),  # tiny downward move → short
        ]
        for settlement, expected_direction in test_cases:
            strategy = PerfectForesightStrategy({_DATE: settlement})
            state = _make_day_state(last_settlement=entry)
            signal = strategy.act(state)
            assert signal is not None
            assert signal.direction == expected_direction, (
                f"settlement={settlement}: expected {expected_direction}, got {signal.direction}"
            )


class TestPerfectForesightPnl:
    """Tests that PnL is always non-negative (the key upper-bound property).

    Since entry price is set by the runner (= last_settlement_price),
    we manually construct Trade objects here to verify the PnL arithmetic.
    """

    def _make_trade(
        self,
        signal: Signal,
        entry_price: float,
        hours: int = 24,
    ) -> Trade:
        """Simulate what BacktestRunner does: build Trade from Signal + market state."""
        return Trade(
            delivery_date=signal.delivery_date,
            entry_price=entry_price,
            position_mw=float(signal.direction) * 1.0,
            hours=hours,
        )

    def test_pnl_non_negative_long(self) -> None:
        """Long trade profit: (60 - 50) * +1 * 24 = 240."""
        entry = 50.0
        settlement = 60.0
        strategy = PerfectForesightStrategy({_DATE: settlement})
        state = _make_day_state(last_settlement=entry)
        signal = strategy.act(state)
        assert signal is not None
        trade = self._make_trade(signal, entry_price=entry)
        pnl = compute_pnl(trade, settlement_price=settlement)
        assert pnl >= 0.0
        assert pnl == pytest.approx(240.0)

    def test_pnl_non_negative_short(self) -> None:
        """Short trade profit: (40 - 50) * -1 * 24 = 240."""
        entry = 50.0
        settlement = 40.0
        strategy = PerfectForesightStrategy({_DATE: settlement})
        state = _make_day_state(last_settlement=entry)
        signal = strategy.act(state)
        assert signal is not None
        trade = self._make_trade(signal, entry_price=entry)
        pnl = compute_pnl(trade, settlement_price=settlement)
        assert pnl >= 0.0
        assert pnl == pytest.approx(240.0)

    def test_pnl_equals_abs_move_times_24(self) -> None:
        """PnL should always equal |P_DA - entry| * 24 MW*h."""
        entry = 55.0
        settlement = 38.0
        strategy = PerfectForesightStrategy({_DATE: settlement})
        state = _make_day_state(last_settlement=entry)
        signal = strategy.act(state)
        assert signal is not None
        trade = self._make_trade(signal, entry_price=entry)
        pnl = compute_pnl(trade, settlement_price=settlement)
        expected = abs(settlement - entry) * 24
        assert pnl == pytest.approx(expected)

    def test_pnl_always_non_negative_for_many_prices(self) -> None:
        """PnL >= 0 for a range of settlement prices."""
        entry = 50.0
        test_settlements = [10.0, 30.0, 49.9, 50.1, 70.0, 100.0, -10.0]
        for s in test_settlements:
            if s == entry:
                continue
            strategy = PerfectForesightStrategy({_DATE: s})
            state = _make_day_state(last_settlement=entry)
            signal = strategy.act(state)
            assert signal is not None
            trade = self._make_trade(signal, entry_price=entry)
            pnl = compute_pnl(trade, settlement_price=s)
            assert pnl >= 0.0, f"Expected non-negative PnL for settlement={s}, got {pnl}"

    def test_pnl_is_upper_bound_over_naive(self) -> None:
        """Perfect foresight PnL >= |naive copy PnL| for any price move.

        The naive always-long strategy earns (settlement - entry) * 24.
        Perfect foresight earns |settlement - entry| * 24.  So PF PnL is
        always >= the absolute value of naive PnL, meaning it is the upper
        bound on any single-direction strategy.
        """
        from energy_modelling.strategy.naive_copy import NaiveCopyStrategy

        entry = 50.0
        test_settlements = [30.0, 45.0, 55.0, 80.0]
        naive = NaiveCopyStrategy()

        for s in test_settlements:
            pf_strategy = PerfectForesightStrategy({_DATE: s})
            state = _make_day_state(last_settlement=entry)

            pf_signal = pf_strategy.act(state)
            naive_signal = naive.act(state)

            assert naive_signal is not None
            naive_trade = self._make_trade(naive_signal, entry_price=entry)
            naive_pnl = compute_pnl(naive_trade, settlement_price=s)

            if pf_signal is not None:
                pf_trade = self._make_trade(pf_signal, entry_price=entry)
                pf_pnl = compute_pnl(pf_trade, settlement_price=s)
                assert pf_pnl >= abs(naive_pnl), (
                    f"settlement={s}: PF PnL {pf_pnl} < |naive PnL| {abs(naive_pnl)}"
                )


class TestPerfectForesightReset:
    """Tests for the reset() method."""

    def test_reset_is_noop(self) -> None:
        """reset() should not raise and strategy should work identically after."""
        strategy = PerfectForesightStrategy({_DATE: 60.0})
        strategy.reset()
        state = _make_day_state()
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == 1

    def test_reset_does_not_clear_prices(self) -> None:
        """Settlement prices must still be accessible after reset()."""
        strategy = PerfectForesightStrategy({_DATE: 40.0})
        strategy.reset()
        state = _make_day_state(last_settlement=50.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == -1  # 40 < 50 → short
