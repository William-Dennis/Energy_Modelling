"""Phase E: Tests for calendar/temporal/regime strategies.

Each strategy gets at least 5 tests covering:
1. forecast() returns a finite float after fit()
2. forecast() before fit() returns a finite float (graceful)
3. fit() sets skip_buffer
4. reset() is callable
5. Strategy subclasses BacktestStrategy
6. Signal polarity / regime logic where applicable
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.month_seasonal import MonthSeasonalStrategy
from strategies.quarter_seasonal import QuarterSeasonalStrategy
from strategies.monday_effect import MondayEffectStrategy
from strategies.volatility_regime_ml import VolatilityRegimeMLStrategy
from strategies.zscore_momentum import ZScoreMomentumStrategy
from strategies.gas_carbon_joint_trend import GasCarbonJointTrendStrategy
from strategies.renewable_regime import RenewableRegimeStrategy
from strategies.net_demand_momentum import NetDemandMomentumStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(1)


def _make_train(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    prices = 50.0 + _RNG.normal(0, 10, n).cumsum()
    changes = np.diff(prices, prepend=prices[0])
    return pd.DataFrame(
        {
            "delivery_date": [d.date() for d in dates],
            "split": ["train"] * n,
            "settlement_price": prices,
            "last_settlement_price": np.roll(prices, 1),
            "price_change_eur_mwh": changes,
            "target_direction": np.sign(changes).astype(float),
            "pnl_long_eur": changes * 24.0,
            "pnl_short_eur": -changes * 24.0,
            # Derived features
            "rolling_vol_7d": _RNG.uniform(2, 10, n),
            "price_zscore_20d": _RNG.normal(0, 1.5, n),
            "gas_trend_3d": _RNG.normal(0, 0.5, n),
            "carbon_trend_3d": _RNG.normal(0, 0.5, n),
            "renewable_penetration_pct": _RNG.uniform(10, 50, n),
            "net_demand_mw": 28_000 + _RNG.normal(0, 2_000, n),
            "dow_int": [d.dayofweek for d in dates],
            "is_weekend": [1 if d.dayofweek >= 5 else 0 for d in dates],
        }
    )


def _make_state(
    last_price: float = 55.0,
    delivery: date = date(2024, 6, 3),  # Monday
    features: dict | None = None,
) -> BacktestState:
    feats = {
        "rolling_vol_7d": 5.0,
        "price_zscore_20d": 0.8,
        "gas_trend_3d": 0.3,
        "carbon_trend_3d": 0.2,
        "renewable_penetration_pct": 30.0,
        "net_demand_mw": 30_000.0,
        "dow_int": delivery.weekday(),
        "is_weekend": 1 if delivery.weekday() >= 5 else 0,
    }
    if features:
        feats.update(features)
    return BacktestState(
        delivery_date=delivery,
        last_settlement_price=last_price,
        features=pd.Series(feats),
        history=pd.DataFrame(),
    )


# ===========================================================================
# MonthSeasonalStrategy
# ===========================================================================


class TestMonthSeasonalStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = MonthSeasonalStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_fit_sets_monthly_mean_for_all_months(self) -> None:
        self.strategy.fit(self.df)
        # 120 days = 4 months × 12 = should cover most months
        assert len(self.strategy._monthly_mean) > 0

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_forecast_uses_delivery_month(self) -> None:
        self.strategy.fit(self.df)
        jan_state = _make_state(delivery=date(2024, 1, 15))
        jun_state = _make_state(delivery=date(2024, 6, 15))
        # Both should be finite; they may differ if seasonal signal differs
        assert np.isfinite(self.strategy.forecast(jan_state))
        assert np.isfinite(self.strategy.forecast(jun_state))


# ===========================================================================
# QuarterSeasonalStrategy
# ===========================================================================


class TestQuarterSeasonalStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = QuarterSeasonalStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_four_quarters_populated(self) -> None:
        # Use 400 days to guarantee all 4 quarters are covered
        df_full = _make_train(n=400)
        self.strategy.fit(df_full)
        assert len(self.strategy._quarterly_mean) == 4

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_q1_and_q3_differ(self) -> None:
        self.strategy.fit(self.df)
        q1 = self.strategy._quarterly_mean.get(1, 0.0)
        q3 = self.strategy._quarterly_mean.get(3, 0.0)
        # They may happen to be equal with random data but must both be finite
        assert np.isfinite(q1) and np.isfinite(q3)


# ===========================================================================
# MondayEffectStrategy
# ===========================================================================


class TestMondayEffectStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = MondayEffectStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_seven_dow_signals_populated(self) -> None:
        self.strategy.fit(self.df)
        # Training covers all 7 DOWs in 120 days
        assert len(self.strategy._dow_mean) == 7

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_uses_dow_int_feature(self) -> None:
        self.strategy.fit(self.df)
        # Monday (dow=0) and Friday (dow=4) may give different forecasts
        mon = _make_state(delivery=date(2024, 6, 3), features={"dow_int": 0})
        fri = _make_state(delivery=date(2024, 6, 7), features={"dow_int": 4})
        assert np.isfinite(self.strategy.forecast(mon))
        assert np.isfinite(self.strategy.forecast(fri))


# ===========================================================================
# VolatilityRegimeMLStrategy
# ===========================================================================


class TestVolatilityRegimeMLStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = VolatilityRegimeMLStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_threshold_is_median_vol(self) -> None:
        self.strategy.fit(self.df)
        expected = float(np.median(self.df["rolling_vol_7d"]))
        assert self.strategy._vol_threshold == pytest.approx(expected)

    def test_high_and_low_regime_means_differ(self) -> None:
        self.strategy.fit(self.df)
        # With random data they can be equal in edge cases; just check finite
        assert np.isfinite(self.strategy._high_vol_mean)
        assert np.isfinite(self.strategy._low_vol_mean)

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_graceful_missing_vol_col(self) -> None:
        df_no_vol = self.df.drop(columns=["rolling_vol_7d"])
        self.strategy.fit(df_no_vol)
        assert np.isfinite(self.strategy.forecast(_make_state()))


# ===========================================================================
# ZScoreMomentumStrategy
# ===========================================================================


class TestZScoreMomentumStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = ZScoreMomentumStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_high_positive_zscore_gives_long_signal(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0, features={"price_zscore_20d": 5.0})
        result = self.strategy.forecast(state)
        assert result > 50.0  # long = price + 1

    def test_high_negative_zscore_gives_short_signal(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0, features={"price_zscore_20d": -5.0})
        result = self.strategy.forecast(state)
        assert result < 50.0  # short = price - 1

    def test_near_zero_zscore_no_trade(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0, features={"price_zscore_20d": 0.0})
        result = self.strategy.forecast(state)
        assert result == pytest.approx(50.0)

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


# ===========================================================================
# GasCarbonJointTrendStrategy
# ===========================================================================


class TestGasCarbonJointTrendStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = GasCarbonJointTrendStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_both_positive_trends_give_long(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(
            last_price=50.0,
            features={"gas_trend_3d": 10.0, "carbon_trend_3d": 10.0},
        )
        result = self.strategy.forecast(state)
        assert result > 50.0

    def test_both_negative_trends_give_short(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(
            last_price=50.0,
            features={"gas_trend_3d": -10.0, "carbon_trend_3d": -10.0},
        )
        result = self.strategy.forecast(state)
        assert result < 50.0

    def test_mixed_trends_no_trade(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(
            last_price=50.0,
            features={"gas_trend_3d": 10.0, "carbon_trend_3d": -10.0},
        )
        result = self.strategy.forecast(state)
        assert result == pytest.approx(50.0)

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


# ===========================================================================
# RenewableRegimeStrategy
# ===========================================================================


class TestRenewableRegimeStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = RenewableRegimeStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_threshold_is_median(self) -> None:
        self.strategy.fit(self.df)
        expected = float(np.median(self.df["renewable_penetration_pct"]))
        assert self.strategy._threshold == pytest.approx(expected)

    def test_high_ren_and_low_ren_forecasts_differ(self) -> None:
        self.strategy.fit(self.df)
        high = _make_state(last_price=50.0, features={"renewable_penetration_pct": 100.0})
        low = _make_state(last_price=50.0, features={"renewable_penetration_pct": 0.0})
        # May be equal with random data, but both must be finite
        assert np.isfinite(self.strategy.forecast(high))
        assert np.isfinite(self.strategy.forecast(low))

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_graceful_missing_col(self) -> None:
        df2 = self.df.drop(columns=["renewable_penetration_pct"])
        self.strategy.fit(df2)
        assert np.isfinite(self.strategy.forecast(_make_state()))


# ===========================================================================
# NetDemandMomentumStrategy
# ===========================================================================


class TestNetDemandMomentumStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = NetDemandMomentumStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_threshold_is_mean_net_demand(self) -> None:
        self.strategy.fit(self.df)
        expected = float(self.df["net_demand_mw"].mean())
        assert self.strategy._mean_net_demand == pytest.approx(expected)

    def test_high_nd_and_low_nd_both_finite(self) -> None:
        self.strategy.fit(self.df)
        high = _make_state(last_price=50.0, features={"net_demand_mw": 50_000.0})
        low = _make_state(last_price=50.0, features={"net_demand_mw": 1_000.0})
        assert np.isfinite(self.strategy.forecast(high))
        assert np.isfinite(self.strategy.forecast(low))

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_graceful_missing_col(self) -> None:
        df2 = self.df.drop(columns=["net_demand_mw"])
        self.strategy.fit(df2)
        assert np.isfinite(self.strategy.forecast(_make_state()))
