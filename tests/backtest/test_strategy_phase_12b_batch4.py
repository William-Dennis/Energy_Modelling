"""Tests for Phase 12B Batch 4 strategies (90-94).

Covers:
1. SeasonalRegimeSwitchStrategy
2. WeekendMeanReversionStrategy
3. HighVolSkipStrategy
4. RadiationRegimeStrategy
5. IndependentVoteStrategy
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.high_vol_skip import HighVolSkipStrategy
from strategies.independent_vote import IndependentVoteStrategy
from strategies.radiation_regime import RadiationRegimeStrategy
from strategies.seasonal_regime_switch import SeasonalRegimeSwitchStrategy
from strategies.weekend_mean_reversion import WeekendMeanReversionStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_multi(
    features: dict[str, float],
    last_price: float = 50.0,
    delivery_date: date | None = None,
) -> BacktestState:
    return BacktestState(
        delivery_date=delivery_date or date(2024, 7, 15),
        last_settlement_price=last_price,
        features=pd.Series(features),
        history=pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# 1. SeasonalRegimeSwitchStrategy
# ---------------------------------------------------------------------------


class TestSeasonalRegimeSwitchStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(SeasonalRegimeSwitchStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = SeasonalRegimeSwitchStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_winter_rising_gas_goes_long(self) -> None:
        s = SeasonalRegimeSwitchStrategy()
        df = pd.DataFrame(
            {
                "renewable_penetration_pct": [20, 30, 40, 50, 60],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        features = {"gas_trend_3d": 2.0}
        forecast = s.forecast(_state_multi(features, delivery_date=date(2024, 1, 15)))
        assert forecast > 50.0  # long

    def test_winter_falling_gas_goes_short(self) -> None:
        s = SeasonalRegimeSwitchStrategy()
        df = pd.DataFrame(
            {
                "renewable_penetration_pct": [20, 30, 40, 50, 60],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        features = {"gas_trend_3d": -2.0}
        forecast = s.forecast(_state_multi(features, delivery_date=date(2024, 12, 15)))
        assert forecast < 50.0  # short

    def test_summer_high_renewables_goes_short(self) -> None:
        s = SeasonalRegimeSwitchStrategy()
        df = pd.DataFrame(
            {
                "renewable_penetration_pct": [20, 30, 40, 50, 60],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median renew = 40
        features = {"renewable_penetration_pct": 70.0}
        forecast = s.forecast(_state_multi(features, delivery_date=date(2024, 7, 15)))
        assert forecast < 50.0  # short

    def test_summer_low_renewables_goes_long(self) -> None:
        s = SeasonalRegimeSwitchStrategy()
        df = pd.DataFrame(
            {
                "renewable_penetration_pct": [20, 30, 40, 50, 60],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median renew = 40
        features = {"renewable_penetration_pct": 10.0}
        forecast = s.forecast(_state_multi(features, delivery_date=date(2024, 6, 15)))
        assert forecast > 50.0  # long

    def test_reset_is_noop(self) -> None:
        s = SeasonalRegimeSwitchStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 2. WeekendMeanReversionStrategy
# ---------------------------------------------------------------------------


class TestWeekendMeanReversionStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(WeekendMeanReversionStrategy, BacktestStrategy)

    def test_fit_computes_averages(self) -> None:
        s = WeekendMeanReversionStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [60, 60, 60, 60, 60, 40, 40],  # weekday=60, weekend=40
                "is_weekend": [0, 0, 0, 0, 0, 1, 1],
                "price_change_eur_mwh": [1.0] * 7,
            }
        )
        s.fit(df)
        assert s._weekday_avg == pytest.approx(60.0)
        assert s._weekend_avg == pytest.approx(40.0)

    def test_raises_before_fit(self) -> None:
        s = WeekendMeanReversionStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_weekend_above_avg_goes_short(self) -> None:
        s = WeekendMeanReversionStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [60, 60, 60, 60, 60, 40, 40],
                "is_weekend": [0, 0, 0, 0, 0, 1, 1],
                "price_change_eur_mwh": [1.0] * 7,
            }
        )
        s.fit(df)
        # Weekend, price 55 > weekend_avg 40
        forecast = s.forecast(_state_multi({"is_weekend": 1}, last_price=55.0))
        assert forecast < 55.0

    def test_weekday_below_avg_goes_long(self) -> None:
        s = WeekendMeanReversionStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [60, 60, 60, 60, 60, 40, 40],
                "is_weekend": [0, 0, 0, 0, 0, 1, 1],
                "price_change_eur_mwh": [1.0] * 7,
            }
        )
        s.fit(df)
        # Weekday, price 50 < weekday_avg 60
        forecast = s.forecast(_state_multi({"is_weekend": 0}, last_price=50.0))
        assert forecast > 50.0

    def test_reset_is_noop(self) -> None:
        s = WeekendMeanReversionStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 3. HighVolSkipStrategy
# ---------------------------------------------------------------------------


class TestHighVolSkipStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(HighVolSkipStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = HighVolSkipStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_high_vol_returns_neutral(self) -> None:
        s = HighVolSkipStrategy()
        df = pd.DataFrame(
            {
                "rolling_vol_14d": [2.0, 4.0, 6.0, 8.0],
                "price_mean": [50.0, 55.0, 45.0, 60.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)  # p75_vol = 7.0
        # High vol -> skip
        features = {"rolling_vol_14d": 20.0}
        forecast = s.forecast(_state_multi(features, last_price=60.0))
        assert forecast == 60.0  # neutral

    def test_low_vol_above_median_goes_short(self) -> None:
        s = HighVolSkipStrategy()
        df = pd.DataFrame(
            {
                "rolling_vol_14d": [2.0, 4.0, 6.0, 8.0],
                "price_mean": [40.0, 50.0, 60.0, 70.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)  # median_price = 55, p75_vol = 7
        features = {"rolling_vol_14d": 3.0}
        forecast = s.forecast(_state_multi(features, last_price=65.0))
        assert forecast < 65.0  # short (above median)

    def test_low_vol_below_median_goes_long(self) -> None:
        s = HighVolSkipStrategy()
        df = pd.DataFrame(
            {
                "rolling_vol_14d": [2.0, 4.0, 6.0, 8.0],
                "price_mean": [40.0, 50.0, 60.0, 70.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)  # median_price = 55
        features = {"rolling_vol_14d": 3.0}
        forecast = s.forecast(_state_multi(features, last_price=45.0))
        assert forecast > 45.0  # long (below median)

    def test_reset_is_noop(self) -> None:
        s = HighVolSkipStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 4. RadiationRegimeStrategy
# ---------------------------------------------------------------------------


class TestRadiationRegimeStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(RadiationRegimeStrategy, BacktestStrategy)

    def test_fit_computes_quartiles(self) -> None:
        s = RadiationRegimeStrategy()
        df = pd.DataFrame(
            {
                "weather_shortwave_radiation_wm2_mean": [
                    100,
                    150,
                    200,
                    250,
                    300,
                    350,
                    400,
                    450,
                ],
                "price_change_eur_mwh": [1.0, -1.0] * 4,
            }
        )
        s.fit(df)
        assert s._p25_rad is not None
        assert s._p75_rad is not None
        assert s._p25_rad < s._p75_rad

    def test_raises_before_fit(self) -> None:
        s = RadiationRegimeStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_high_radiation_goes_short(self) -> None:
        s = RadiationRegimeStrategy()
        df = pd.DataFrame(
            {
                "weather_shortwave_radiation_wm2_mean": [100, 200, 300, 400],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)  # p75 = 350
        features = {"weather_shortwave_radiation_wm2_mean": 500.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_low_radiation_goes_long(self) -> None:
        s = RadiationRegimeStrategy()
        df = pd.DataFrame(
            {
                "weather_shortwave_radiation_wm2_mean": [100, 200, 300, 400],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)  # p25 = 150
        features = {"weather_shortwave_radiation_wm2_mean": 50.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_middle_radiation_neutral(self) -> None:
        s = RadiationRegimeStrategy()
        df = pd.DataFrame(
            {
                "weather_shortwave_radiation_wm2_mean": [100, 200, 300, 400],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)
        features = {"weather_shortwave_radiation_wm2_mean": 250.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0

    def test_reset_is_noop(self) -> None:
        s = RadiationRegimeStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 5. IndependentVoteStrategy
# ---------------------------------------------------------------------------


class TestIndependentVoteStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(IndependentVoteStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = IndependentVoteStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_all_bullish_goes_long(self) -> None:
        s = IndependentVoteStrategy()
        df = pd.DataFrame(
            {
                "price_zscore_20d": [0.0] * 10,
                "net_demand_mw": [50000.0] * 10,
                "renewable_penetration_pct": [30.0] * 10,
                "price_change_eur_mwh": [1.0] * 10,
            }
        )
        s.fit(df)
        # All signals bullish:
        # zscore below median -> long
        # gas trend positive -> long
        # net demand above median -> long
        # weekday -> long
        # low renewables -> long
        features = {
            "price_zscore_20d": -2.0,
            "gas_trend_3d": 5.0,
            "net_demand_mw": 60000.0,
            "is_weekend": 0,
            "renewable_penetration_pct": 10.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_all_bearish_goes_short(self) -> None:
        s = IndependentVoteStrategy()
        df = pd.DataFrame(
            {
                "price_zscore_20d": [0.0] * 10,
                "net_demand_mw": [50000.0] * 10,
                "renewable_penetration_pct": [30.0] * 10,
                "price_change_eur_mwh": [1.0] * 10,
            }
        )
        s.fit(df)
        # All signals bearish:
        # zscore above median -> short
        # gas trend negative -> short
        # net demand below median -> short
        # weekend -> short
        # high renewables -> short
        features = {
            "price_zscore_20d": 2.0,
            "gas_trend_3d": -5.0,
            "net_demand_mw": 30000.0,
            "is_weekend": 1,
            "renewable_penetration_pct": 60.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_mixed_signals_majority_wins(self) -> None:
        s = IndependentVoteStrategy()
        df = pd.DataFrame(
            {
                "price_zscore_20d": [0.0] * 10,
                "net_demand_mw": [50000.0] * 10,
                "renewable_penetration_pct": [30.0] * 10,
                "price_change_eur_mwh": [1.0] * 10,
            }
        )
        s.fit(df)
        # 3 long, 2 short:
        # zscore below -> long
        # gas positive -> long
        # net demand above -> long
        # weekend -> short
        # high renewables -> short
        features = {
            "price_zscore_20d": -1.0,
            "gas_trend_3d": 2.0,
            "net_demand_mw": 60000.0,
            "is_weekend": 1,
            "renewable_penetration_pct": 50.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # majority long

    def test_reset_is_noop(self) -> None:
        s = IndependentVoteStrategy()
        s.reset()
