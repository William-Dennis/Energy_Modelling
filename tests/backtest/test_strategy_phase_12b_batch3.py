"""Tests for Phase 12B Batch 3 strategies (85-89).

Covers:
1. MonthlyMeanReversionStrategy
2. LoadGenerationGapStrategy
3. RenewableRampStrategy
4. NuclearGasSubstitutionStrategy
5. VolatilityBreakoutStrategy
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.load_generation_gap import LoadGenerationGapStrategy
from strategies.monthly_mean_reversion import MonthlyMeanReversionStrategy
from strategies.nuclear_gas_substitution import NuclearGasSubstitutionStrategy
from strategies.renewable_ramp import RenewableRampStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_multi(
    features: dict[str, float],
    last_price: float = 50.0,
    history: pd.DataFrame | None = None,
    delivery_date: date | None = None,
) -> BacktestState:
    return BacktestState(
        delivery_date=delivery_date or date(2024, 7, 15),
        last_settlement_price=last_price,
        features=pd.Series(features),
        history=history if history is not None else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# 1. MonthlyMeanReversionStrategy
# ---------------------------------------------------------------------------


class TestMonthlyMeanReversionStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(MonthlyMeanReversionStrategy, BacktestStrategy)

    def test_fit_computes_monthly_avg(self) -> None:
        s = MonthlyMeanReversionStrategy()
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        df = pd.DataFrame(
            {
                "price_mean": [50.0 + i * 0.1 for i in range(60)],
                "price_change_eur_mwh": [1.0] * 60,
            },
            index=idx,
        )
        s.fit(df)
        assert s._monthly_avg is not None
        assert 1 in s._monthly_avg  # January
        assert 2 in s._monthly_avg  # February

    def test_raises_before_fit(self) -> None:
        s = MonthlyMeanReversionStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_price_above_avg_goes_short(self) -> None:
        s = MonthlyMeanReversionStrategy()
        idx = pd.date_range("2024-07-01", periods=31, freq="D")
        df = pd.DataFrame(
            {
                "price_mean": [50.0] * 31,
                "price_change_eur_mwh": [1.0] * 31,
            },
            index=idx,
        )
        s.fit(df)
        # July avg = 50, current price = 60 -> overbought
        forecast = s.forecast(_state_multi({}, last_price=60.0, delivery_date=date(2024, 7, 15)))
        assert forecast < 60.0

    def test_price_below_avg_goes_long(self) -> None:
        s = MonthlyMeanReversionStrategy()
        idx = pd.date_range("2024-07-01", periods=31, freq="D")
        df = pd.DataFrame(
            {
                "price_mean": [50.0] * 31,
                "price_change_eur_mwh": [1.0] * 31,
            },
            index=idx,
        )
        s.fit(df)
        # July avg = 50, current price = 40 -> underbought
        forecast = s.forecast(_state_multi({}, last_price=40.0, delivery_date=date(2024, 7, 15)))
        assert forecast > 40.0

    def test_unknown_month_uses_overall(self) -> None:
        s = MonthlyMeanReversionStrategy()
        idx = pd.date_range("2024-07-01", periods=31, freq="D")
        df = pd.DataFrame(
            {
                "price_mean": [50.0] * 31,
                "price_change_eur_mwh": [1.0] * 31,
            },
            index=idx,
        )
        s.fit(df)
        # December not in training data
        forecast = s.forecast(_state_multi({}, last_price=60.0, delivery_date=date(2024, 12, 15)))
        # Falls back to overall avg (50), current 60 > 50 -> short
        assert forecast < 60.0

    def test_reset_is_noop(self) -> None:
        s = MonthlyMeanReversionStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 2. LoadGenerationGapStrategy
# ---------------------------------------------------------------------------


class TestLoadGenerationGapStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(LoadGenerationGapStrategy, BacktestStrategy)

    def test_fit_computes_median_gap(self) -> None:
        s = LoadGenerationGapStrategy()
        df = pd.DataFrame(
            {
                "load_actual_mw_mean": [60000, 65000, 70000, 75000, 80000],
                "total_fossil_mw": [30000, 30000, 30000, 30000, 30000],
                "gen_wind_onshore_mw_mean": [10000, 10000, 10000, 10000, 10000],
                "gen_wind_offshore_mw_mean": [5000, 5000, 5000, 5000, 5000],
                "gen_solar_mw_mean": [5000, 5000, 5000, 5000, 5000],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # gap = load - (fossil + wind_on + wind_off + solar)
        # = [10000, 15000, 20000, 25000, 30000], median = 20000
        assert s._median_gap == pytest.approx(20000.0)

    def test_raises_before_fit(self) -> None:
        s = LoadGenerationGapStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({"load_actual_mw_mean": 70000.0}))

    def test_high_gap_goes_long(self) -> None:
        s = LoadGenerationGapStrategy()
        df = pd.DataFrame(
            {
                "load_actual_mw_mean": [60000, 65000, 70000, 75000, 80000],
                "total_fossil_mw": [30000] * 5,
                "gen_wind_onshore_mw_mean": [10000] * 5,
                "gen_wind_offshore_mw_mean": [5000] * 5,
                "gen_solar_mw_mean": [5000] * 5,
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)  # median gap = 20000
        # Gap = 100000 - 50000 = 50000 > 20000
        features = {
            "load_actual_mw_mean": 100000.0,
            "total_fossil_mw": 30000.0,
            "gen_wind_onshore_mw_mean": 10000.0,
            "gen_wind_offshore_mw_mean": 5000.0,
            "gen_solar_mw_mean": 5000.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_low_gap_goes_short(self) -> None:
        s = LoadGenerationGapStrategy()
        df = pd.DataFrame(
            {
                "load_actual_mw_mean": [60000, 65000, 70000, 75000, 80000],
                "total_fossil_mw": [30000] * 5,
                "gen_wind_onshore_mw_mean": [10000] * 5,
                "gen_wind_offshore_mw_mean": [5000] * 5,
                "gen_solar_mw_mean": [5000] * 5,
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Gap = 40000 - 50000 = -10000 <= 20000
        features = {
            "load_actual_mw_mean": 40000.0,
            "total_fossil_mw": 30000.0,
            "gen_wind_onshore_mw_mean": 10000.0,
            "gen_wind_offshore_mw_mean": 5000.0,
            "gen_solar_mw_mean": 5000.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_reset_is_noop(self) -> None:
        s = LoadGenerationGapStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 3. RenewableRampStrategy
# ---------------------------------------------------------------------------


class TestRenewableRampStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(RenewableRampStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = RenewableRampStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_ramp_up_goes_short(self) -> None:
        s = RenewableRampStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # Yesterday: 10000 MW total renewable, today: 15000 MW (+50%)
        history = pd.DataFrame(
            {
                "gen_wind_onshore_mw_mean": [5000.0],
                "gen_wind_offshore_mw_mean": [3000.0],
                "gen_solar_mw_mean": [2000.0],
            }
        )
        features = {
            "gen_wind_onshore_mw_mean": 8000.0,
            "gen_wind_offshore_mw_mean": 4000.0,
            "gen_solar_mw_mean": 3000.0,
        }
        forecast = s.forecast(_state_multi(features, history=history))
        assert forecast < 50.0  # short (surplus)

    def test_ramp_down_goes_long(self) -> None:
        s = RenewableRampStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # Yesterday: 15000 MW, today: 5000 MW (-67%)
        history = pd.DataFrame(
            {
                "gen_wind_onshore_mw_mean": [8000.0],
                "gen_wind_offshore_mw_mean": [4000.0],
                "gen_solar_mw_mean": [3000.0],
            }
        )
        features = {
            "gen_wind_onshore_mw_mean": 3000.0,
            "gen_wind_offshore_mw_mean": 1000.0,
            "gen_solar_mw_mean": 1000.0,
        }
        forecast = s.forecast(_state_multi(features, history=history))
        assert forecast > 50.0  # long (deficit)

    def test_small_change_neutral(self) -> None:
        s = RenewableRampStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # Yesterday: 10000 MW, today: 10500 MW (+5% < 10% threshold)
        history = pd.DataFrame(
            {
                "gen_wind_onshore_mw_mean": [5000.0],
                "gen_wind_offshore_mw_mean": [3000.0],
                "gen_solar_mw_mean": [2000.0],
            }
        )
        features = {
            "gen_wind_onshore_mw_mean": 5250.0,
            "gen_wind_offshore_mw_mean": 3150.0,
            "gen_solar_mw_mean": 2100.0,
        }
        forecast = s.forecast(_state_multi(features, history=history))
        assert forecast == 50.0  # neutral

    def test_no_history_neutral(self) -> None:
        s = RenewableRampStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        forecast = s.forecast(_state_multi({}))
        assert forecast == 50.0

    def test_reset_is_noop(self) -> None:
        s = RenewableRampStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 4. NuclearGasSubstitutionStrategy
# ---------------------------------------------------------------------------


class TestNuclearGasSubstitutionStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(NuclearGasSubstitutionStrategy, BacktestStrategy)

    def test_fit_computes_medians(self) -> None:
        s = NuclearGasSubstitutionStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000, 6000, 7000, 8000, 9000],
                "gas_price_usd_mean": [10, 20, 30, 40, 50],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        assert s._median_nuclear == pytest.approx(7000.0)
        assert s._median_gas == pytest.approx(30.0)

    def test_raises_before_fit(self) -> None:
        s = NuclearGasSubstitutionStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_low_nuclear_high_gas_goes_long(self) -> None:
        s = NuclearGasSubstitutionStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000, 6000, 7000, 8000, 9000],
                "gas_price_usd_mean": [10, 20, 30, 40, 50],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Low nuclear (3000 < 7000), high gas (60 > 30)
        features = {"gen_nuclear_mw_mean": 3000.0, "gas_price_usd_mean": 60.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_high_nuclear_low_gas_goes_short(self) -> None:
        s = NuclearGasSubstitutionStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000, 6000, 7000, 8000, 9000],
                "gas_price_usd_mean": [10, 20, 30, 40, 50],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # High nuclear (10000 >= 7000), low gas (10 <= 30)
        features = {"gen_nuclear_mw_mean": 10000.0, "gas_price_usd_mean": 10.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_mixed_signals_neutral(self) -> None:
        s = NuclearGasSubstitutionStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000, 6000, 7000, 8000, 9000],
                "gas_price_usd_mean": [10, 20, 30, 40, 50],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Low nuclear, low gas -> mixed -> neutral
        features = {"gen_nuclear_mw_mean": 3000.0, "gas_price_usd_mean": 10.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0

    def test_reset_is_noop(self) -> None:
        s = NuclearGasSubstitutionStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 5. VolatilityBreakoutStrategy
# ---------------------------------------------------------------------------


class TestVolatilityBreakoutStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(VolatilityBreakoutStrategy, BacktestStrategy)

    def test_fit_computes_p75_vol(self) -> None:
        s = VolatilityBreakoutStrategy()
        df = pd.DataFrame(
            {
                "rolling_vol_7d": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                "price_change_eur_mwh": [1.0, -1.0] * 4,
            }
        )
        s.fit(df)
        assert s._p75_vol is not None
        assert s._p75_vol > 0

    def test_raises_before_fit(self) -> None:
        s = VolatilityBreakoutStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_high_vol_positive_change_goes_long(self) -> None:
        s = VolatilityBreakoutStrategy()
        df = pd.DataFrame(
            {
                "rolling_vol_7d": [2.0, 4.0, 6.0, 8.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)  # p75 = 7.0
        features = {"rolling_vol_7d": 20.0, "price_change_eur_mwh": 5.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long (breakout up)

    def test_high_vol_negative_change_goes_short(self) -> None:
        s = VolatilityBreakoutStrategy()
        df = pd.DataFrame(
            {
                "rolling_vol_7d": [2.0, 4.0, 6.0, 8.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)  # p75 = 7.0
        features = {"rolling_vol_7d": 20.0, "price_change_eur_mwh": -5.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short (breakout down)

    def test_low_vol_neutral(self) -> None:
        s = VolatilityBreakoutStrategy()
        df = pd.DataFrame(
            {
                "rolling_vol_7d": [2.0, 4.0, 6.0, 8.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)
        features = {"rolling_vol_7d": 3.0, "price_change_eur_mwh": 5.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0  # neutral

    def test_reset_is_noop(self) -> None:
        s = VolatilityBreakoutStrategy()
        s.reset()
