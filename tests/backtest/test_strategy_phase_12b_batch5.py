"""Tests for Phase 12B Batch 5 strategies (95-99).

Covers:
1. MedianIndependentStrategy
2. SpreadConsensusStrategy
3. SupplyDemandBalanceStrategy
4. ContrarianMomentumStrategy
5. ConvictionWeightedStrategy
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.contrarian_momentum import ContrarianMomentumStrategy
from strategies.conviction_weighted import ConvictionWeightedStrategy
from strategies.median_independent import MedianIndependentStrategy
from strategies.spread_consensus import SpreadConsensusStrategy
from strategies.supply_demand_balance import SupplyDemandBalanceStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_multi(
    features: dict[str, float],
    last_price: float = 50.0,
) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 7, 15),
        last_settlement_price=last_price,
        features=pd.Series(features),
        history=pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# 1. MedianIndependentStrategy
# ---------------------------------------------------------------------------


class TestMedianIndependentStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(MedianIndependentStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = MedianIndependentStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_all_bullish_goes_long(self) -> None:
        s = MedianIndependentStrategy()
        df = pd.DataFrame(
            {
                "price_zscore_20d": [0.0] * 10,
                "load_surprise": [0.0] * 10,
                "price_change_eur_mwh": [1.0] * 10,
            }
        )
        s.fit(df)
        features = {
            "price_zscore_20d": -2.0,  # below median -> long
            "gas_trend_3d": 5.0,  # positive -> long
            "load_surprise": 1000.0,  # above median -> long
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_all_bearish_goes_short(self) -> None:
        s = MedianIndependentStrategy()
        df = pd.DataFrame(
            {
                "price_zscore_20d": [0.0] * 10,
                "load_surprise": [0.0] * 10,
                "price_change_eur_mwh": [1.0] * 10,
            }
        )
        s.fit(df)
        features = {
            "price_zscore_20d": 2.0,  # above median -> short
            "gas_trend_3d": -5.0,  # negative -> short
            "load_surprise": -1000.0,  # below median -> short
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_mixed_uses_median(self) -> None:
        s = MedianIndependentStrategy()
        df = pd.DataFrame(
            {
                "price_zscore_20d": [0.0] * 10,
                "load_surprise": [0.0] * 10,
                "price_change_eur_mwh": [1.0] * 10,
            }
        )
        s.fit(df)
        # 2 long (zscore, load) + 1 short (gas) -> median = long
        features = {
            "price_zscore_20d": -2.0,  # long
            "gas_trend_3d": -5.0,  # short
            "load_surprise": 1000.0,  # long
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_reset_is_noop(self) -> None:
        s = MedianIndependentStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 2. SpreadConsensusStrategy
# ---------------------------------------------------------------------------


class TestSpreadConsensusStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(SpreadConsensusStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = SpreadConsensusStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_fit_computes_median_spreads(self) -> None:
        s = SpreadConsensusStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [60] * 5,
                "price_fr_eur_mwh_mean": [50] * 5,
                "price_nl_eur_mwh_mean": [55] * 5,
                "price_pl_eur_mwh_mean": [45] * 5,
                "price_dk_1_eur_mwh_mean": [52] * 5,
                "price_change_eur_mwh": [1.0] * 5,
            }
        )
        s.fit(df)
        assert "FR" in s._median_spreads
        assert "NL" in s._median_spreads
        assert s._median_spreads["FR"] == pytest.approx(10.0)

    def test_de_expensive_everywhere_goes_short(self) -> None:
        s = SpreadConsensusStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50] * 5,
                "price_fr_eur_mwh_mean": [50] * 5,
                "price_nl_eur_mwh_mean": [50] * 5,
                "price_pl_eur_mwh_mean": [50] * 5,
                "price_dk_1_eur_mwh_mean": [50] * 5,
                "price_change_eur_mwh": [1.0] * 5,
            }
        )
        s.fit(df)  # all median spreads = 0
        # DE expensive vs all neighbours
        features = {
            "price_mean": 80.0,
            "price_fr_eur_mwh_mean": 50.0,
            "price_nl_eur_mwh_mean": 50.0,
            "price_pl_eur_mwh_mean": 50.0,
            "price_dk_1_eur_mwh_mean": 50.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_de_cheap_everywhere_goes_long(self) -> None:
        s = SpreadConsensusStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50] * 5,
                "price_fr_eur_mwh_mean": [50] * 5,
                "price_nl_eur_mwh_mean": [50] * 5,
                "price_pl_eur_mwh_mean": [50] * 5,
                "price_dk_1_eur_mwh_mean": [50] * 5,
                "price_change_eur_mwh": [1.0] * 5,
            }
        )
        s.fit(df)
        # DE cheap vs all neighbours
        features = {
            "price_mean": 30.0,
            "price_fr_eur_mwh_mean": 50.0,
            "price_nl_eur_mwh_mean": 50.0,
            "price_pl_eur_mwh_mean": 50.0,
            "price_dk_1_eur_mwh_mean": 50.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_mixed_signals_neutral(self) -> None:
        s = SpreadConsensusStrategy()
        df = pd.DataFrame(
            {
                "price_mean": [50] * 5,
                "price_fr_eur_mwh_mean": [50] * 5,
                "price_nl_eur_mwh_mean": [50] * 5,
                "price_pl_eur_mwh_mean": [50] * 5,
                "price_dk_1_eur_mwh_mean": [50] * 5,
                "price_change_eur_mwh": [1.0] * 5,
            }
        )
        s.fit(df)
        # Mixed: 2 expensive, 2 cheap -> no consensus
        features = {
            "price_mean": 50.0,
            "price_fr_eur_mwh_mean": 40.0,  # DE expensive
            "price_nl_eur_mwh_mean": 40.0,  # DE expensive
            "price_pl_eur_mwh_mean": 60.0,  # DE cheap
            "price_dk_1_eur_mwh_mean": 60.0,  # DE cheap
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0

    def test_reset_is_noop(self) -> None:
        s = SpreadConsensusStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 3. SupplyDemandBalanceStrategy
# ---------------------------------------------------------------------------


class TestSupplyDemandBalanceStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(SupplyDemandBalanceStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = SupplyDemandBalanceStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_scarcity_goes_long(self) -> None:
        s = SupplyDemandBalanceStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # forecast_load = 80000, total_gen = 60000 -> ratio 1.33 > 1.05
        features = {
            "forecast_load_mw_mean": 80000.0,
            "total_fossil_mw": 30000.0,
            "gen_nuclear_mw_mean": 10000.0,
            "gen_wind_onshore_mw_mean": 10000.0,
            "gen_wind_offshore_mw_mean": 5000.0,
            "gen_solar_mw_mean": 5000.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_surplus_goes_short(self) -> None:
        s = SupplyDemandBalanceStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # forecast_load = 40000, total_gen = 60000 -> ratio 0.67 < 0.95
        features = {
            "forecast_load_mw_mean": 40000.0,
            "total_fossil_mw": 30000.0,
            "gen_nuclear_mw_mean": 10000.0,
            "gen_wind_onshore_mw_mean": 10000.0,
            "gen_wind_offshore_mw_mean": 5000.0,
            "gen_solar_mw_mean": 5000.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_balanced_neutral(self) -> None:
        s = SupplyDemandBalanceStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        # forecast_load = 60000, total_gen = 60000 -> ratio 1.0 (neutral)
        features = {
            "forecast_load_mw_mean": 60000.0,
            "total_fossil_mw": 30000.0,
            "gen_nuclear_mw_mean": 10000.0,
            "gen_wind_onshore_mw_mean": 10000.0,
            "gen_wind_offshore_mw_mean": 5000.0,
            "gen_solar_mw_mean": 5000.0,
        }
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0

    def test_zero_gen_neutral(self) -> None:
        s = SupplyDemandBalanceStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0]})
        s.fit(df)
        features = {"forecast_load_mw_mean": 60000.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0

    def test_reset_is_noop(self) -> None:
        s = SupplyDemandBalanceStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 4. ContrarianMomentumStrategy
# ---------------------------------------------------------------------------


class TestContrarianMomentumStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(ContrarianMomentumStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = ContrarianMomentumStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_strong_trend_follows(self) -> None:
        s = ContrarianMomentumStrategy()
        df = pd.DataFrame(
            {
                "gas_trend_3d": [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
                "price_change_eur_mwh": [1.0, -1.0] * 4,
            }
        )
        s.fit(df)  # p75 of |trend| = 2.5
        # Strong positive trend (5 > 2.5) -> follow -> long
        features = {"gas_trend_3d": 5.0, "price_change_eur_mwh": -2.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_strong_negative_trend_follows_short(self) -> None:
        s = ContrarianMomentumStrategy()
        df = pd.DataFrame(
            {
                "gas_trend_3d": [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
                "price_change_eur_mwh": [1.0, -1.0] * 4,
            }
        )
        s.fit(df)
        # Strong negative trend (-5) -> follow -> short
        features = {"gas_trend_3d": -5.0, "price_change_eur_mwh": 2.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_weak_trend_goes_contrarian(self) -> None:
        s = ContrarianMomentumStrategy()
        df = pd.DataFrame(
            {
                "gas_trend_3d": [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
                "price_change_eur_mwh": [1.0, -1.0] * 4,
            }
        )
        s.fit(df)  # p75 of |trend| = 2.5
        # Weak trend (0.1), last change positive -> contrarian -> short
        features = {"gas_trend_3d": 0.1, "price_change_eur_mwh": 3.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_weak_trend_zero_change_neutral(self) -> None:
        s = ContrarianMomentumStrategy()
        df = pd.DataFrame(
            {
                "gas_trend_3d": [1.0, -1.0, 2.0, -2.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0],
            }
        )
        s.fit(df)
        features = {"gas_trend_3d": 0.1, "price_change_eur_mwh": 0.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0

    def test_reset_is_noop(self) -> None:
        s = ContrarianMomentumStrategy()
        s.reset()


# ---------------------------------------------------------------------------
# 5. ConvictionWeightedStrategy
# ---------------------------------------------------------------------------


class TestConvictionWeightedStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(ConvictionWeightedStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = ConvictionWeightedStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_high_zscore_goes_short(self) -> None:
        s = ConvictionWeightedStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        features = {"price_zscore_20d": 1.5}  # high zscore -> short
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0

    def test_low_zscore_goes_long(self) -> None:
        s = ConvictionWeightedStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        features = {"price_zscore_20d": -1.5}  # low zscore -> long
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0

    def test_low_conviction_neutral(self) -> None:
        s = ConvictionWeightedStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        features = {"price_zscore_20d": 0.2}  # |0.2| < 0.5 -> neutral
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0

    def test_extreme_zscore_capped(self) -> None:
        s = ConvictionWeightedStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [2.0, -2.0, 3.0]})
        s.fit(df)
        mac = s._mean_abs_change
        # zscore = 5.0, but conviction capped at 2.0
        features = {"price_zscore_20d": 5.0}
        forecast = s.forecast(_state_multi(features))
        expected = 50.0 - 2.0 * mac  # capped at 2x
        assert forecast == pytest.approx(expected)

    def test_reset_is_noop(self) -> None:
        s = ConvictionWeightedStrategy()
        s.reset()
