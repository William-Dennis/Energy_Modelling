"""Tests for Phase 12B Batch 1 strategies (75-79).

Covers:
1. RadiationSolarStrategy
2. IntradayRangeStrategy
3. OffshoreWindAnomalyStrategy
4. ForecastPriceErrorStrategy
5. PolandSpreadStrategy
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.forecast_price_error import ForecastPriceErrorStrategy
from strategies.intraday_range import IntradayRangeStrategy
from strategies.offshore_wind_anomaly import OffshoreWindAnomalyStrategy
from strategies.poland_spread import PolandSpreadStrategy
from strategies.radiation_solar import RadiationSolarStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(feature_name: str, value: float, last_price: float = 50.0) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 5, 1),
        last_settlement_price=last_price,
        features=pd.Series({feature_name: value}),
        history=pd.DataFrame(),
    )


def _state_multi(features: dict[str, float], last_price: float = 50.0) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 5, 1),
        last_settlement_price=last_price,
        features=pd.Series(features),
        history=pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# 1. RadiationSolarStrategy
# ---------------------------------------------------------------------------


class TestRadiationSolarStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(RadiationSolarStrategy, BacktestStrategy)

    def test_fit_sets_median(self) -> None:
        s = RadiationSolarStrategy()
        df = pd.DataFrame(
            {
                "weather_shortwave_radiation_wm2_mean": [100, 200, 300, 400, 500],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        assert s._median_radiation == pytest.approx(300.0)

    def test_raises_before_fit(self) -> None:
        s = RadiationSolarStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state("weather_shortwave_radiation_wm2_mean", 200.0))

    def test_high_radiation_goes_short(self) -> None:
        s = RadiationSolarStrategy()
        df = pd.DataFrame(
            {
                "weather_shortwave_radiation_wm2_mean": [100, 200, 300],
                "price_change_eur_mwh": [1.0, -1.0, 2.0],
            }
        )
        s.fit(df)
        forecast = s.forecast(_state("weather_shortwave_radiation_wm2_mean", 500.0))
        assert forecast < 50.0  # short

    def test_low_radiation_goes_long(self) -> None:
        s = RadiationSolarStrategy()
        df = pd.DataFrame(
            {
                "weather_shortwave_radiation_wm2_mean": [100, 200, 300],
                "price_change_eur_mwh": [1.0, -1.0, 2.0],
            }
        )
        s.fit(df)
        forecast = s.forecast(_state("weather_shortwave_radiation_wm2_mean", 50.0))
        assert forecast > 50.0  # long


# ---------------------------------------------------------------------------
# 2. IntradayRangeStrategy
# ---------------------------------------------------------------------------


class TestIntradayRangeStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(IntradayRangeStrategy, BacktestStrategy)

    def test_fit_sets_percentiles(self) -> None:
        s = IntradayRangeStrategy()
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "price_range": rng.uniform(2, 15, 100),
                "price_change_eur_mwh": rng.randn(100),
            }
        )
        s.fit(df)
        assert s._p25_range is not None
        assert s._p75_range is not None
        assert s._p25_range < s._p75_range

    def test_raises_before_fit(self) -> None:
        s = IntradayRangeStrategy()
        features = {"price_max": 60.0, "price_min": 40.0, "price_range": 20.0}
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_wide_range_mean_reverts(self) -> None:
        s = IntradayRangeStrategy()
        # Train with range mostly 2-5, p75 ~= 4
        df = pd.DataFrame(
            {
                "price_range": [2.0, 3.0, 3.0, 4.0, 5.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Wide range day: midpoint = 55, settlement = 50 -> mean-revert up
        features = {"price_max": 70.0, "price_min": 40.0, "price_range": 30.0}
        forecast = s.forecast(_state_multi(features, last_price=50.0))
        assert forecast > 50.0  # reverts toward midpoint (55)

    def test_narrow_range_trend_follows(self) -> None:
        s = IntradayRangeStrategy()
        df = pd.DataFrame(
            {
                "price_range": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        # Narrow range day: midpoint = 49, settlement = 50 -> trend-follow down
        features = {"price_max": 50.0, "price_min": 48.0, "price_range": 1.0}
        forecast = s.forecast(_state_multi(features, last_price=50.0))
        # Settlement (50) > midpoint (49) -> trend continuation up
        assert forecast > 50.0

    def test_middle_range_neutral(self) -> None:
        s = IntradayRangeStrategy()
        df = pd.DataFrame(
            {
                "price_range": [2.0, 4.0, 6.0, 8.0, 10.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Middle-range day -> neutral
        features = {"price_max": 55.0, "price_min": 45.0, "price_range": 5.5}
        forecast = s.forecast(_state_multi(features, last_price=50.0))
        assert forecast == 50.0  # neutral


# ---------------------------------------------------------------------------
# 3. OffshoreWindAnomalyStrategy
# ---------------------------------------------------------------------------


class TestOffshoreWindAnomalyStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(OffshoreWindAnomalyStrategy, BacktestStrategy)

    def test_fit_sets_threshold(self) -> None:
        s = OffshoreWindAnomalyStrategy()
        df = pd.DataFrame(
            {
                "gen_wind_offshore_mw_mean": [2000, 2500, 3000, 3500, 4000],
                "forecast_wind_offshore_mw_mean": [2200, 2200, 2800, 3200, 3800],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        assert s._median_abs_error is not None
        assert s._median_abs_error > 0

    def test_raises_before_fit(self) -> None:
        s = OffshoreWindAnomalyStrategy()
        features = {
            "gen_wind_offshore_mw_mean": 3000.0,
            "forecast_wind_offshore_mw_mean": 2500.0,
        }
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_surplus_goes_short(self) -> None:
        s = OffshoreWindAnomalyStrategy()
        df = pd.DataFrame(
            {
                "gen_wind_offshore_mw_mean": [2000, 2200, 2400, 2600, 2800],
                "forecast_wind_offshore_mw_mean": [2100, 2100, 2100, 2100, 2100],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Actual >> forecast -> surplus supply
        features = {
            "gen_wind_offshore_mw_mean": 5000.0,
            "forecast_wind_offshore_mw_mean": 2000.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_shortfall_goes_long(self) -> None:
        s = OffshoreWindAnomalyStrategy()
        df = pd.DataFrame(
            {
                "gen_wind_offshore_mw_mean": [2000, 2200, 2400, 2600, 2800],
                "forecast_wind_offshore_mw_mean": [2100, 2100, 2100, 2100, 2100],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Actual << forecast -> supply shortfall
        features = {
            "gen_wind_offshore_mw_mean": 1000.0,
            "forecast_wind_offshore_mw_mean": 5000.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_small_anomaly_neutral(self) -> None:
        s = OffshoreWindAnomalyStrategy()
        df = pd.DataFrame(
            {
                "gen_wind_offshore_mw_mean": [2000, 2200, 2400, 2600, 2800],
                "forecast_wind_offshore_mw_mean": [2100, 2100, 2100, 2100, 2100],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Small anomaly -> neutral
        features = {
            "gen_wind_offshore_mw_mean": 2100.0,
            "forecast_wind_offshore_mw_mean": 2100.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0  # neutral


# ---------------------------------------------------------------------------
# 4. ForecastPriceErrorStrategy
# ---------------------------------------------------------------------------


class TestForecastPriceErrorStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(ForecastPriceErrorStrategy, BacktestStrategy)

    def test_fit_resets_state(self) -> None:
        s = ForecastPriceErrorStrategy()
        s._cum_error = 10.0
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        assert s._cum_error == 0.0
        assert s._last_forecast is None

    def test_first_forecast_is_neutral(self) -> None:
        s = ForecastPriceErrorStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        state = _state("dummy", 0.0, last_price=50.0)
        forecast = s.forecast(state)
        assert forecast == 50.0  # neutral on first call

    def test_overforecast_corrects_down(self) -> None:
        s = ForecastPriceErrorStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # First forecast: neutral at 50
        s.forecast(_state("dummy", 0.0, last_price=50.0))
        # Now price dropped to 45 but we forecast 50 -> overforecast
        forecast = s.forecast(_state("dummy", 0.0, last_price=45.0))
        assert forecast < 45.0  # corrects downward

    def test_underforecast_corrects_up(self) -> None:
        s = ForecastPriceErrorStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # First forecast: neutral at 50
        s.forecast(_state("dummy", 0.0, last_price=50.0))
        # Now price rose to 55 but we forecast 50 -> underforecast
        forecast = s.forecast(_state("dummy", 0.0, last_price=55.0))
        assert forecast > 55.0  # corrects upward

    def test_reset_is_noop(self) -> None:
        s = ForecastPriceErrorStrategy()
        s.reset()  # should not raise


# ---------------------------------------------------------------------------
# 5. PolandSpreadStrategy
# ---------------------------------------------------------------------------


class TestPolandSpreadStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(PolandSpreadStrategy, BacktestStrategy)

    def test_fit_computes_median_spread(self) -> None:
        s = PolandSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_pl_eur_mwh_mean": [40, 45, 50, 55, 60],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Spread is always 10
        assert s._median_spread == pytest.approx(10.0)

    def test_raises_before_fit(self) -> None:
        s = PolandSpreadStrategy()
        features = {"price_mean": 50.0, "price_pl_eur_mwh_mean": 40.0}
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_de_expensive_goes_short(self) -> None:
        s = PolandSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_pl_eur_mwh_mean": [40, 45, 50, 55, 60],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spread = 10
        # DE much more expensive than PL -> spread = 30 > 10
        features = {"price_mean": 70.0, "price_pl_eur_mwh_mean": 40.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_de_cheap_goes_long(self) -> None:
        s = PolandSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_pl_eur_mwh_mean": [40, 45, 50, 55, 60],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spread = 10
        # DE cheap relative to PL -> spread = -5 <= 10
        features = {"price_mean": 40.0, "price_pl_eur_mwh_mean": 45.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_reset_is_noop(self) -> None:
        s = PolandSpreadStrategy()
        s.reset()  # should not raise
