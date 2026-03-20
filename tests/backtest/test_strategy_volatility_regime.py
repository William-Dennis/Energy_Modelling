"""Tests for VolatilityRegimeStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.volatility_regime import VolatilityRegimeStrategy

_STD_COL = "price_std"
_CHANGE_COL = "price_change_eur_mwh"


def _make_train(
    std_vals: list[float] | None = None,
    change_vals: list[float] | None = None,
) -> pd.DataFrame:
    # std values: 4 low, 1 high (P75 = 4.0 boundary)
    std = std_vals or [2.0, 3.0, 4.0, 5.0, 10.0]
    change = change_vals or [1.0, -2.0, 3.0, -1.0, 2.0]
    return pd.DataFrame({_STD_COL: std, _CHANGE_COL: change})


def _make_state(
    price_std: float,
    price_change: float,
    last_price: float = 50.0,
) -> BacktestState:
    # price_change_eur_mwh is an excluded column in the real backtest runner,
    # so it appears in history (yesterday's row), not in features.
    history = pd.DataFrame(
        {_CHANGE_COL: [price_change]},
        index=pd.Index([date(2024, 2, 19)], name="delivery_date"),
    )
    return BacktestState(
        delivery_date=date(2024, 2, 20),
        last_settlement_price=last_price,
        features=pd.Series({_STD_COL: price_std}),
        history=history,
    )


class TestVolatilityRegimeInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(VolatilityRegimeStrategy, BacktestStrategy)

    def test_fit_sets_high_vol_threshold(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        assert s._vol_threshold is not None

    def test_reset_preserves_thresholds(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        vt = s._vol_threshold
        s.reset()
        assert s._vol_threshold == vt

    def test_raises_before_fit(self) -> None:
        s = VolatilityRegimeStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state(price_std=5.0, price_change=2.0))


class TestVolatilityRegimeThreshold:
    def test_vol_threshold_is_75th_percentile(self) -> None:
        import numpy as np

        train = _make_train()
        s = VolatilityRegimeStrategy()
        s.fit(train)
        expected = float(np.percentile(train[_STD_COL], 75))
        assert abs(s._vol_threshold - expected) < 1e-9


class TestVolatilityRegimeSignal:
    """High vol + up move -> short (mean-revert); high vol + down -> long.
    Low vol + up move -> long (momentum); low vol + down -> short.
    """

    def test_high_vol_price_up_short(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        # std=10 > P75 (high vol), price went up -> mean-revert -> short
        assert s.act(_make_state(price_std=10.0, price_change=5.0)) == -1

    def test_high_vol_price_down_long(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        # std=10 > P75 (high vol), price went down -> mean-revert -> long
        assert s.act(_make_state(price_std=10.0, price_change=-5.0)) == 1

    def test_low_vol_price_up_long(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        # std=2 < P75 (low vol), price went up -> momentum -> long
        assert s.act(_make_state(price_std=2.0, price_change=3.0)) == 1

    def test_low_vol_price_down_short(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        # std=2 < P75 (low vol), price went down -> momentum -> short
        assert s.act(_make_state(price_std=2.0, price_change=-3.0)) == -1

    def test_result_is_int(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        result = s.act(_make_state(price_std=5.0, price_change=2.0))
        assert isinstance(result, int)

    def test_zero_change_in_high_vol_treated_as_non_negative(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        # zero change -> short (mean-revert of non-negative move)
        assert s.act(_make_state(price_std=10.0, price_change=0.0)) == -1

    def test_zero_change_in_low_vol_treated_as_non_negative(self) -> None:
        s = VolatilityRegimeStrategy()
        s.fit(_make_train())
        # zero change -> long (momentum)
        assert s.act(_make_state(price_std=2.0, price_change=0.0)) == 1
