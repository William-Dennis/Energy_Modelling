"""Tests for Phase 12B Batch 2 strategies (80-84).

Covers:
1. DenmarkSpreadStrategy
2. CzechAustrianMeanStrategy
3. SparkSpreadStrategy
4. CarbonGasRatioStrategy
5. WeeklyAutocorrelationStrategy
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.carbon_gas_ratio import CarbonGasRatioStrategy
from strategies.czech_austrian_mean import CzechAustrianMeanStrategy
from strategies.denmark_spread import DenmarkSpreadStrategy
from strategies.spark_spread import SparkSpreadStrategy
from strategies.weekly_autocorrelation import WeeklyAutocorrelationStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_multi(
    features: dict[str, float],
    last_price: float = 50.0,
    history: pd.DataFrame | None = None,
) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 5, 1),
        last_settlement_price=last_price,
        features=pd.Series(features),
        history=history if history is not None else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# 1. DenmarkSpreadStrategy
# ---------------------------------------------------------------------------


class TestDenmarkSpreadStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(DenmarkSpreadStrategy, BacktestStrategy)

    def test_fit_computes_median_spread(self) -> None:
        s = DenmarkSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_dk_1_eur_mwh_mean": [45, 50, 55, 60, 65],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Spread is always 5
        assert s._median_spread == pytest.approx(5.0)

    def test_raises_before_fit(self) -> None:
        s = DenmarkSpreadStrategy()
        features = {"price_mean": 50.0, "price_dk_1_eur_mwh_mean": 40.0}
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_de_expensive_goes_short(self) -> None:
        s = DenmarkSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_dk_1_eur_mwh_mean": [45, 50, 55, 60, 65],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spread = 5
        # DE much more expensive -> spread = 25 > 5
        features = {"price_mean": 70.0, "price_dk_1_eur_mwh_mean": 45.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_de_cheap_goes_long(self) -> None:
        s = DenmarkSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_dk_1_eur_mwh_mean": [45, 50, 55, 60, 65],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spread = 5
        # DE cheap vs DK1 -> spread = -10 <= 5
        features = {"price_mean": 40.0, "price_dk_1_eur_mwh_mean": 50.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_reset_is_noop(self) -> None:
        s = DenmarkSpreadStrategy()
        s.reset()  # should not raise


# ---------------------------------------------------------------------------
# 2. CzechAustrianMeanStrategy
# ---------------------------------------------------------------------------


class TestCzechAustrianMeanStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(CzechAustrianMeanStrategy, BacktestStrategy)

    def test_fit_computes_median_spread(self) -> None:
        s = CzechAustrianMeanStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_cz_eur_mwh_mean": [45, 50, 55, 60, 65],
                "price_at_eur_mwh_mean": [43, 48, 53, 58, 63],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Avg neighbour: [44, 49, 54, 59, 64]
        # Spread: [6, 6, 6, 6, 6]
        assert s._median_spread == pytest.approx(6.0)

    def test_raises_before_fit(self) -> None:
        s = CzechAustrianMeanStrategy()
        features = {
            "price_mean": 50.0,
            "price_cz_eur_mwh_mean": 40.0,
            "price_at_eur_mwh_mean": 38.0,
        }
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_de_expensive_goes_short(self) -> None:
        s = CzechAustrianMeanStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_cz_eur_mwh_mean": [45, 50, 55, 60, 65],
                "price_at_eur_mwh_mean": [43, 48, 53, 58, 63],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spread = 6
        # DE much more expensive -> spread = 70 - 45 = 25 > 6
        features = {
            "price_mean": 70.0,
            "price_cz_eur_mwh_mean": 40.0,
            "price_at_eur_mwh_mean": 50.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_de_cheap_goes_long(self) -> None:
        s = CzechAustrianMeanStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50, 55, 60, 65, 70],
                "price_cz_eur_mwh_mean": [45, 50, 55, 60, 65],
                "price_at_eur_mwh_mean": [43, 48, 53, 58, 63],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spread = 6
        # DE cheap vs CZ-AT mean -> spread = 40 - 55 = -15 <= 6
        features = {
            "price_mean": 40.0,
            "price_cz_eur_mwh_mean": 55.0,
            "price_at_eur_mwh_mean": 55.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_reset_is_noop(self) -> None:
        s = CzechAustrianMeanStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 3. SparkSpreadStrategy
# ---------------------------------------------------------------------------


class TestSparkSpreadStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(SparkSpreadStrategy, BacktestStrategy)

    def test_fit_computes_median_spark(self) -> None:
        s = SparkSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [80, 90, 100, 110, 120],
                "gas_price_usd_mean": [10, 10, 10, 10, 10],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # spark = price - gas * 7 = [10, 20, 30, 40, 50]
        assert s._median_spark == pytest.approx(30.0)

    def test_raises_before_fit(self) -> None:
        s = SparkSpreadStrategy()
        features = {"price_mean": 100.0, "gas_price_usd_mean": 10.0}
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_high_spark_goes_short(self) -> None:
        s = SparkSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [80, 90, 100, 110, 120],
                "gas_price_usd_mean": [10, 10, 10, 10, 10],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spark = 30
        # High spark: 200 - 10*7 = 130 > 30
        features = {"price_mean": 200.0, "gas_price_usd_mean": 10.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_low_spark_goes_long(self) -> None:
        s = SparkSpreadStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [80, 90, 100, 110, 120],
                "gas_price_usd_mean": [10, 10, 10, 10, 10],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median spark = 30
        # Low spark: 50 - 10*7 = -20 <= 30
        features = {"price_mean": 50.0, "gas_price_usd_mean": 10.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_reset_is_noop(self) -> None:
        s = SparkSpreadStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 4. CarbonGasRatioStrategy
# ---------------------------------------------------------------------------


class TestCarbonGasRatioStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(CarbonGasRatioStrategy, BacktestStrategy)

    def test_fit_computes_median_ratio(self) -> None:
        s = CarbonGasRatioStrategy()
        df = pd.DataFrame(
            {
                "carbon_price_usd_mean": [20, 30, 40, 50, 60],
                "gas_price_usd_mean": [10, 10, 10, 10, 10],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Ratio: [2, 3, 4, 5, 6], median = 4
        assert s._median_ratio == pytest.approx(4.0)

    def test_raises_before_fit(self) -> None:
        s = CarbonGasRatioStrategy()
        features = {"carbon_price_usd_mean": 40.0, "gas_price_usd_mean": 10.0}
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_high_ratio_goes_long(self) -> None:
        s = CarbonGasRatioStrategy()
        df = pd.DataFrame(
            {
                "carbon_price_usd_mean": [20, 30, 40, 50, 60],
                "gas_price_usd_mean": [10, 10, 10, 10, 10],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median ratio = 4
        # High ratio: 100/10 = 10 > 4
        features = {"carbon_price_usd_mean": 100.0, "gas_price_usd_mean": 10.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_low_ratio_goes_short(self) -> None:
        s = CarbonGasRatioStrategy()
        df = pd.DataFrame(
            {
                "carbon_price_usd_mean": [20, 30, 40, 50, 60],
                "gas_price_usd_mean": [10, 10, 10, 10, 10],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median ratio = 4
        # Low ratio: 10/10 = 1 <= 4
        features = {"carbon_price_usd_mean": 10.0, "gas_price_usd_mean": 10.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_zero_gas_uses_median(self) -> None:
        s = CarbonGasRatioStrategy()
        df = pd.DataFrame(
            {
                "carbon_price_usd_mean": [20, 30, 40, 50, 60],
                "gas_price_usd_mean": [10, 10, 10, 10, 10],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Gas = 0 should not crash, uses median ratio
        features = {"carbon_price_usd_mean": 40.0, "gas_price_usd_mean": 0.0}
        forecast = s.forecast(_state_multi(features))
        # ratio = median = 4, not > 4, so direction = -1 (short)
        assert forecast == pytest.approx(50.0 - s._mean_abs_change)

    def test_reset_is_noop(self) -> None:
        s = CarbonGasRatioStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 5. WeeklyAutocorrelationStrategy
# ---------------------------------------------------------------------------


class TestWeeklyAutocorrelationStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(WeeklyAutocorrelationStrategy, BacktestStrategy)

    def test_fit_computes_median_weekly_change(self) -> None:
        s = WeeklyAutocorrelationStrategy()
        prices = list(range(50, 70))  # 20 days, prices 50..69
        df = pd.DataFrame(
            {
                "price_mean": prices,
                "price_change_eur_mwh": [1.0] * 20,
            }
        )
        s.fit(df)
        # weekly changes are all 7 (each day is +1 from prev)
        assert s._median_weekly_change == pytest.approx(7.0)

    def test_raises_before_fit(self) -> None:
        s = WeeklyAutocorrelationStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({"price_mean": 50.0}))

    def test_overbought_goes_short(self) -> None:
        s = WeeklyAutocorrelationStrategy()
        prices = [50.0] * 20  # flat training -> median weekly change = 0
        df = pd.DataFrame(
            {
                "price_mean": prices,
                "price_change_eur_mwh": [1.0] * 20,
            }
        )
        s.fit(df)
        # Price 7 days ago was 50, expected = 50+0 = 50, current = 60 > 50
        history = pd.DataFrame({"price_mean": [50.0] * 10})
        forecast = s.forecast(_state_multi({"price_mean": 60.0}, last_price=60.0, history=history))
        assert forecast < 60.0  # short (overbought)

    def test_underbought_goes_long(self) -> None:
        s = WeeklyAutocorrelationStrategy()
        prices = [50.0] * 20  # flat training -> median weekly change = 0
        df = pd.DataFrame(
            {
                "price_mean": prices,
                "price_change_eur_mwh": [1.0] * 20,
            }
        )
        s.fit(df)
        # Price 7 days ago was 50, expected = 50+0 = 50, current = 40 < 50
        history = pd.DataFrame({"price_mean": [50.0] * 10})
        forecast = s.forecast(_state_multi({"price_mean": 40.0}, last_price=40.0, history=history))
        assert forecast > 40.0  # long (underbought)

    def test_no_history_returns_neutral(self) -> None:
        s = WeeklyAutocorrelationStrategy()
        prices = [50.0] * 20
        df = pd.DataFrame(
            {
                "price_mean": prices,
                "price_change_eur_mwh": [1.0] * 20,
            }
        )
        s.fit(df)
        # No history -> neutral
        forecast = s.forecast(_state_multi({"price_mean": 50.0}, last_price=50.0))
        assert forecast == 50.0

    def test_short_training_uses_default(self) -> None:
        s = WeeklyAutocorrelationStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50.0, 55.0],
                "price_change_eur_mwh": [1.0, -1.0],
            }
        )
        s.fit(df)  # < 8 rows -> default median = 0
        assert s._median_weekly_change == pytest.approx(0.0)

    def test_reset_is_noop(self) -> None:
        s = WeeklyAutocorrelationStrategy()
        s.reset()
