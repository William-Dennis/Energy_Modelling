"""Tests for Phase C derived-feature threshold strategies.

Covers all 15 strategies (5+ tests each):
1.  NetDemandStrategy
2.  PriceZScoreReversionStrategy
3.  GasTrendStrategy
4.  CarbonTrendStrategy
5.  FuelIndexTrendStrategy
6.  DEFRSpreadStrategy
7.  DENLSpreadStrategy
8.  MultiSpreadStrategy
9.  NLFlowSignalStrategy
10. FRFlowSignalStrategy
11. PriceMinReversionStrategy
12. WindForecastErrorStrategy
13. LoadSurpriseStrategy
14. RenewablesPenetrationStrategy
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.carbon_trend import CarbonTrendStrategy
from strategies.de_fr_spread import DEFRSpreadStrategy
from strategies.de_nl_spread import DENLSpreadStrategy
from strategies.fr_flow_signal import FRFlowSignalStrategy
from strategies.fuel_index_trend import FuelIndexTrendStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.load_surprise import LoadSurpriseStrategy
from strategies.multi_spread import MultiSpreadStrategy
from strategies.net_demand import NetDemandStrategy
from strategies.nl_flow_signal import NLFlowSignalStrategy
from strategies.price_min_reversion import PriceMinReversionStrategy
from strategies.price_zscore_reversion import PriceZScoreReversionStrategy
from strategies.renewables_penetration import RenewablesPenetrationStrategy
from strategies.wind_forecast_error import WindForecastErrorStrategy


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
# 1. NetDemandStrategy
# ---------------------------------------------------------------------------


class TestNetDemandStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(NetDemandStrategy, BacktestStrategy)

    def test_fit_sets_threshold(self) -> None:
        s = NetDemandStrategy()
        s.fit(pd.DataFrame({"net_demand_mw": [1e4, 2e4, 3e4, 4e4, 5e4]}))
        assert s._threshold == 3e4

    def test_above_median_long(self) -> None:
        s = NetDemandStrategy()
        s.fit(pd.DataFrame({"net_demand_mw": [1e4, 2e4, 3e4, 4e4, 5e4]}))
        assert s.act(_state("net_demand_mw", 5e4)) == 1

    def test_below_median_short(self) -> None:
        s = NetDemandStrategy()
        s.fit(pd.DataFrame({"net_demand_mw": [1e4, 2e4, 3e4, 4e4, 5e4]}))
        assert s.act(_state("net_demand_mw", 1e4)) == -1

    def test_at_median_long(self) -> None:
        s = NetDemandStrategy()
        s.fit(pd.DataFrame({"net_demand_mw": [1e4, 2e4, 3e4, 4e4, 5e4]}))
        assert s.act(_state("net_demand_mw", 3e4)) == 1

    def test_reset_preserves_threshold(self) -> None:
        s = NetDemandStrategy()
        s.fit(pd.DataFrame({"net_demand_mw": [1e4, 2e4, 3e4]}))
        s.reset()
        assert s._threshold is not None

    def test_raises_before_fit(self) -> None:
        s = NetDemandStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("net_demand_mw", 3e4))


# ---------------------------------------------------------------------------
# 2. PriceZScoreReversionStrategy
# ---------------------------------------------------------------------------


class TestPriceZScoreReversionStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(PriceZScoreReversionStrategy, BacktestStrategy)

    def test_high_zscore_short(self) -> None:
        s = PriceZScoreReversionStrategy()
        s.fit(pd.DataFrame({"price_zscore_20d": [0.0] * 5}))
        assert s.act(_state("price_zscore_20d", 2.0)) == -1

    def test_low_zscore_long(self) -> None:
        s = PriceZScoreReversionStrategy()
        s.fit(pd.DataFrame({"price_zscore_20d": [0.0] * 5}))
        assert s.act(_state("price_zscore_20d", -2.0)) == 1

    def test_moderate_zscore_skip(self) -> None:
        s = PriceZScoreReversionStrategy()
        s.fit(pd.DataFrame({"price_zscore_20d": [0.0] * 5}))
        assert s.act(_state("price_zscore_20d", 0.5)) is None

    def test_exactly_plus1_skip(self) -> None:
        s = PriceZScoreReversionStrategy()
        s.fit(pd.DataFrame({"price_zscore_20d": [0.0] * 5}))
        assert s.act(_state("price_zscore_20d", 1.0)) is None

    def test_exactly_minus1_skip(self) -> None:
        s = PriceZScoreReversionStrategy()
        s.fit(pd.DataFrame({"price_zscore_20d": [0.0] * 5}))
        assert s.act(_state("price_zscore_20d", -1.0)) is None

    def test_raises_before_fit(self) -> None:
        s = PriceZScoreReversionStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("price_zscore_20d", 2.0))

    def test_reset_preserves_fitted(self) -> None:
        s = PriceZScoreReversionStrategy()
        s.fit(pd.DataFrame({"price_zscore_20d": [0.0] * 5}))
        s.reset()
        assert s.act(_state("price_zscore_20d", 2.0)) == -1


# ---------------------------------------------------------------------------
# 3. GasTrendStrategy
# ---------------------------------------------------------------------------


class TestGasTrendStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(GasTrendStrategy, BacktestStrategy)

    def test_positive_trend_long(self) -> None:
        s = GasTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [1.0]}))
        assert s.act(_state("gas_trend_3d", 0.5)) == 1

    def test_negative_trend_short(self) -> None:
        s = GasTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [0.0]}))
        assert s.act(_state("gas_trend_3d", -1.0)) == -1

    def test_zero_trend_short(self) -> None:
        s = GasTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [0.0]}))
        assert s.act(_state("gas_trend_3d", 0.0)) == -1

    def test_reset_preserves_state(self) -> None:
        s = GasTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [0.0]}))
        s.reset()
        assert s._fitted

    def test_raises_before_fit(self) -> None:
        s = GasTrendStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("gas_trend_3d", 1.0))


# ---------------------------------------------------------------------------
# 4. CarbonTrendStrategy
# ---------------------------------------------------------------------------


class TestCarbonTrendStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(CarbonTrendStrategy, BacktestStrategy)

    def test_positive_trend_long(self) -> None:
        s = CarbonTrendStrategy()
        s.fit(pd.DataFrame({"carbon_trend_3d": [0.0]}))
        assert s.act(_state("carbon_trend_3d", 0.3)) == 1

    def test_negative_trend_short(self) -> None:
        s = CarbonTrendStrategy()
        s.fit(pd.DataFrame({"carbon_trend_3d": [0.0]}))
        assert s.act(_state("carbon_trend_3d", -0.3)) == -1

    def test_zero_trend_short(self) -> None:
        s = CarbonTrendStrategy()
        s.fit(pd.DataFrame({"carbon_trend_3d": [0.0]}))
        assert s.act(_state("carbon_trend_3d", 0.0)) == -1

    def test_reset_preserves_fitted(self) -> None:
        s = CarbonTrendStrategy()
        s.fit(pd.DataFrame({"carbon_trend_3d": [0.0]}))
        s.reset()
        assert s._fitted

    def test_raises_before_fit(self) -> None:
        s = CarbonTrendStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("carbon_trend_3d", 1.0))


# ---------------------------------------------------------------------------
# 5. FuelIndexTrendStrategy
# ---------------------------------------------------------------------------


class TestFuelIndexTrendStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(FuelIndexTrendStrategy, BacktestStrategy)

    def test_both_positive_long(self) -> None:
        s = FuelIndexTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [0.0], "carbon_trend_3d": [0.0]}))
        assert s.act(_state_multi({"gas_trend_3d": 1.0, "carbon_trend_3d": 0.5})) == 1

    def test_both_negative_short(self) -> None:
        s = FuelIndexTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [0.0], "carbon_trend_3d": [0.0]}))
        assert s.act(_state_multi({"gas_trend_3d": -1.0, "carbon_trend_3d": -0.5})) == -1

    def test_mixed_sum_negative_short(self) -> None:
        s = FuelIndexTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [0.0], "carbon_trend_3d": [0.0]}))
        # sum = -0.5 → short
        assert s.act(_state_multi({"gas_trend_3d": 1.0, "carbon_trend_3d": -1.5})) == -1

    def test_zero_sum_short(self) -> None:
        s = FuelIndexTrendStrategy()
        s.fit(pd.DataFrame({"gas_trend_3d": [0.0], "carbon_trend_3d": [0.0]}))
        assert s.act(_state_multi({"gas_trend_3d": 0.0, "carbon_trend_3d": 0.0})) == -1

    def test_raises_before_fit(self) -> None:
        s = FuelIndexTrendStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state_multi({"gas_trend_3d": 1.0, "carbon_trend_3d": 1.0}))


# ---------------------------------------------------------------------------
# 6. DEFRSpreadStrategy
# ---------------------------------------------------------------------------


class TestDEFRSpreadStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(DEFRSpreadStrategy, BacktestStrategy)

    def test_negative_spread_long(self) -> None:
        # DE cheaper than FR → spread < 0 → long
        s = DEFRSpreadStrategy()
        s.fit(pd.DataFrame({"de_fr_spread": [0.0]}))
        assert s.act(_state("de_fr_spread", -5.0)) == 1

    def test_positive_spread_short(self) -> None:
        # DE dearer than FR → spread > 0 → short
        s = DEFRSpreadStrategy()
        s.fit(pd.DataFrame({"de_fr_spread": [0.0]}))
        assert s.act(_state("de_fr_spread", 5.0)) == -1

    def test_zero_spread_long(self) -> None:
        # spread == 0 → long (≤ 0)
        s = DEFRSpreadStrategy()
        s.fit(pd.DataFrame({"de_fr_spread": [0.0]}))
        assert s.act(_state("de_fr_spread", 0.0)) == 1

    def test_reset_preserves_fitted(self) -> None:
        s = DEFRSpreadStrategy()
        s.fit(pd.DataFrame({"de_fr_spread": [0.0]}))
        s.reset()
        assert s._fitted

    def test_raises_before_fit(self) -> None:
        s = DEFRSpreadStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("de_fr_spread", -1.0))


# ---------------------------------------------------------------------------
# 7. DENLSpreadStrategy
# ---------------------------------------------------------------------------


class TestDENLSpreadStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(DENLSpreadStrategy, BacktestStrategy)

    def test_negative_spread_long(self) -> None:
        s = DENLSpreadStrategy()
        s.fit(pd.DataFrame({"de_nl_spread": [0.0]}))
        assert s.act(_state("de_nl_spread", -3.0)) == 1

    def test_positive_spread_short(self) -> None:
        s = DENLSpreadStrategy()
        s.fit(pd.DataFrame({"de_nl_spread": [0.0]}))
        assert s.act(_state("de_nl_spread", 3.0)) == -1

    def test_zero_spread_long(self) -> None:
        s = DENLSpreadStrategy()
        s.fit(pd.DataFrame({"de_nl_spread": [0.0]}))
        assert s.act(_state("de_nl_spread", 0.0)) == 1

    def test_reset_preserves_fitted(self) -> None:
        s = DENLSpreadStrategy()
        s.fit(pd.DataFrame({"de_nl_spread": [0.0]}))
        s.reset()
        assert s._fitted

    def test_raises_before_fit(self) -> None:
        s = DENLSpreadStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("de_nl_spread", -1.0))


# ---------------------------------------------------------------------------
# 8. MultiSpreadStrategy
# ---------------------------------------------------------------------------


class TestMultiSpreadStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(MultiSpreadStrategy, BacktestStrategy)

    def test_negative_spread_long(self) -> None:
        s = MultiSpreadStrategy()
        s.fit(pd.DataFrame({"de_avg_neighbour_spread": [0.0]}))
        assert s.act(_state("de_avg_neighbour_spread", -4.0)) == 1

    def test_positive_spread_short(self) -> None:
        s = MultiSpreadStrategy()
        s.fit(pd.DataFrame({"de_avg_neighbour_spread": [0.0]}))
        assert s.act(_state("de_avg_neighbour_spread", 4.0)) == -1

    def test_zero_spread_long(self) -> None:
        s = MultiSpreadStrategy()
        s.fit(pd.DataFrame({"de_avg_neighbour_spread": [0.0]}))
        assert s.act(_state("de_avg_neighbour_spread", 0.0)) == 1

    def test_reset_preserves_fitted(self) -> None:
        s = MultiSpreadStrategy()
        s.fit(pd.DataFrame({"de_avg_neighbour_spread": [0.0]}))
        s.reset()
        assert s._fitted

    def test_raises_before_fit(self) -> None:
        s = MultiSpreadStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("de_avg_neighbour_spread", -2.0))


# ---------------------------------------------------------------------------
# 9. NLFlowSignalStrategy
# ---------------------------------------------------------------------------


class TestNLFlowSignalStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(NLFlowSignalStrategy, BacktestStrategy)

    def test_fit_sets_threshold(self) -> None:
        s = NLFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_nl_net_import_mw_mean": [-500.0, 0.0, 500.0]}))
        assert s._threshold == 0.0

    def test_low_flow_long(self) -> None:
        # heavy NL export (negative) → flow < median → long
        s = NLFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_nl_net_import_mw_mean": [-500.0, 0.0, 500.0]}))
        assert s.act(_state("flow_nl_net_import_mw_mean", -1000.0)) == 1

    def test_high_flow_short(self) -> None:
        s = NLFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_nl_net_import_mw_mean": [-500.0, 0.0, 500.0]}))
        assert s.act(_state("flow_nl_net_import_mw_mean", 1000.0)) == -1

    def test_at_median_short(self) -> None:
        s = NLFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_nl_net_import_mw_mean": [-500.0, 0.0, 500.0]}))
        assert s.act(_state("flow_nl_net_import_mw_mean", 0.0)) == -1

    def test_raises_before_fit(self) -> None:
        s = NLFlowSignalStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("flow_nl_net_import_mw_mean", 100.0))


# ---------------------------------------------------------------------------
# 10. FRFlowSignalStrategy
# ---------------------------------------------------------------------------


class TestFRFlowSignalStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(FRFlowSignalStrategy, BacktestStrategy)

    def test_low_flow_long(self) -> None:
        s = FRFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_fr_net_import_mw_mean": [-500.0, 0.0, 500.0]}))
        assert s.act(_state("flow_fr_net_import_mw_mean", -1000.0)) == 1

    def test_high_flow_short(self) -> None:
        s = FRFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_fr_net_import_mw_mean": [-500.0, 0.0, 500.0]}))
        assert s.act(_state("flow_fr_net_import_mw_mean", 1000.0)) == -1

    def test_at_median_short(self) -> None:
        s = FRFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_fr_net_import_mw_mean": [-500.0, 0.0, 500.0]}))
        assert s.act(_state("flow_fr_net_import_mw_mean", 0.0)) == -1

    def test_reset_preserves_threshold(self) -> None:
        s = FRFlowSignalStrategy()
        s.fit(pd.DataFrame({"flow_fr_net_import_mw_mean": [100.0, 200.0, 300.0]}))
        s.reset()
        assert s._threshold is not None

    def test_raises_before_fit(self) -> None:
        s = FRFlowSignalStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("flow_fr_net_import_mw_mean", 100.0))


# ---------------------------------------------------------------------------
# 11. PriceMinReversionStrategy
# ---------------------------------------------------------------------------


class TestPriceMinReversionStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(PriceMinReversionStrategy, BacktestStrategy)

    def test_fit_sets_threshold(self) -> None:
        s = PriceMinReversionStrategy()
        s.fit(pd.DataFrame({"price_min": [20.0, 30.0, 40.0, 50.0, 60.0]}))
        assert s._threshold == 40.0

    def test_low_min_long(self) -> None:
        s = PriceMinReversionStrategy()
        s.fit(pd.DataFrame({"price_min": [20.0, 30.0, 40.0, 50.0, 60.0]}))
        assert s.act(_state("price_min", 10.0)) == 1

    def test_high_min_short(self) -> None:
        s = PriceMinReversionStrategy()
        s.fit(pd.DataFrame({"price_min": [20.0, 30.0, 40.0, 50.0, 60.0]}))
        assert s.act(_state("price_min", 70.0)) == -1

    def test_at_median_short(self) -> None:
        s = PriceMinReversionStrategy()
        s.fit(pd.DataFrame({"price_min": [20.0, 30.0, 40.0, 50.0, 60.0]}))
        assert s.act(_state("price_min", 40.0)) == -1

    def test_raises_before_fit(self) -> None:
        s = PriceMinReversionStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("price_min", 30.0))


# ---------------------------------------------------------------------------
# 12. WindForecastErrorStrategy
# ---------------------------------------------------------------------------


class TestWindForecastErrorStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(WindForecastErrorStrategy, BacktestStrategy)

    def test_positive_error_short(self) -> None:
        # today's wind forecast > yesterday's actual → more supply → short
        s = WindForecastErrorStrategy()
        s.fit(pd.DataFrame({"wind_forecast_error": [0.0]}))
        assert s.act(_state("wind_forecast_error", 500.0)) == -1

    def test_negative_error_long(self) -> None:
        s = WindForecastErrorStrategy()
        s.fit(pd.DataFrame({"wind_forecast_error": [0.0]}))
        assert s.act(_state("wind_forecast_error", -500.0)) == 1

    def test_zero_error_long(self) -> None:
        s = WindForecastErrorStrategy()
        s.fit(pd.DataFrame({"wind_forecast_error": [0.0]}))
        assert s.act(_state("wind_forecast_error", 0.0)) == 1

    def test_reset_preserves_fitted(self) -> None:
        s = WindForecastErrorStrategy()
        s.fit(pd.DataFrame({"wind_forecast_error": [0.0]}))
        s.reset()
        assert s._fitted

    def test_raises_before_fit(self) -> None:
        s = WindForecastErrorStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("wind_forecast_error", 100.0))


# ---------------------------------------------------------------------------
# 13. LoadSurpriseStrategy
# ---------------------------------------------------------------------------


class TestLoadSurpriseStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(LoadSurpriseStrategy, BacktestStrategy)

    def test_positive_surprise_long(self) -> None:
        s = LoadSurpriseStrategy()
        s.fit(pd.DataFrame({"load_surprise": [0.0]}))
        assert s.act(_state("load_surprise", 1000.0)) == 1

    def test_negative_surprise_short(self) -> None:
        s = LoadSurpriseStrategy()
        s.fit(pd.DataFrame({"load_surprise": [0.0]}))
        assert s.act(_state("load_surprise", -1000.0)) == -1

    def test_zero_surprise_short(self) -> None:
        s = LoadSurpriseStrategy()
        s.fit(pd.DataFrame({"load_surprise": [0.0]}))
        assert s.act(_state("load_surprise", 0.0)) == -1

    def test_reset_preserves_fitted(self) -> None:
        s = LoadSurpriseStrategy()
        s.fit(pd.DataFrame({"load_surprise": [0.0]}))
        s.reset()
        assert s._fitted

    def test_raises_before_fit(self) -> None:
        s = LoadSurpriseStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("load_surprise", 500.0))


# ---------------------------------------------------------------------------
# 14. RenewablesPenetrationStrategy
# ---------------------------------------------------------------------------


class TestRenewablesPenetrationStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(RenewablesPenetrationStrategy, BacktestStrategy)

    def test_fit_sets_threshold(self) -> None:
        s = RenewablesPenetrationStrategy()
        s.fit(pd.DataFrame({"renewable_penetration_pct": [0.1, 0.2, 0.3, 0.4, 0.5]}))
        assert s._threshold == 0.3

    def test_high_penetration_short(self) -> None:
        s = RenewablesPenetrationStrategy()
        s.fit(pd.DataFrame({"renewable_penetration_pct": [0.1, 0.2, 0.3, 0.4, 0.5]}))
        assert s.act(_state("renewable_penetration_pct", 0.8)) == -1

    def test_low_penetration_long(self) -> None:
        s = RenewablesPenetrationStrategy()
        s.fit(pd.DataFrame({"renewable_penetration_pct": [0.1, 0.2, 0.3, 0.4, 0.5]}))
        assert s.act(_state("renewable_penetration_pct", 0.05)) == 1

    def test_at_median_short(self) -> None:
        s = RenewablesPenetrationStrategy()
        s.fit(pd.DataFrame({"renewable_penetration_pct": [0.1, 0.2, 0.3, 0.4, 0.5]}))
        assert s.act(_state("renewable_penetration_pct", 0.3)) == -1

    def test_raises_before_fit(self) -> None:
        s = RenewablesPenetrationStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_state("renewable_penetration_pct", 0.4))
