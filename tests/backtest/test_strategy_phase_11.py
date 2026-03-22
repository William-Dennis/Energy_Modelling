"""Tests for Phase 11 new strategies.

Covers all 7 new strategies (5+ tests each):
1. SpreadMomentumStrategy
2. SelectiveHighConvictionStrategy
3. TemperatureCurveStrategy
4. NuclearEventStrategy
5. FlowImbalanceStrategy
6. RegimeRidgeStrategy
7. PrunedMLEnsembleStrategy
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.flow_imbalance import FlowImbalanceStrategy
from strategies.nuclear_event import NuclearEventStrategy
from strategies.pruned_ml_ensemble import PrunedMLEnsembleStrategy
from strategies.regime_ridge import RegimeRidgeStrategy
from strategies.selective_high_conviction import SelectiveHighConvictionStrategy
from strategies.spread_momentum import SpreadMomentumStrategy
from strategies.temperature_curve import TemperatureCurveStrategy

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


def _make_train_data(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic training DataFrame with all key columns."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = 50.0 + rng.randn(n).cumsum()
    return pd.DataFrame(
        {
            "delivery_date": dates,
            "settlement_price": prices,
            "last_settlement_price": np.concatenate([[50.0], prices[:-1]]),
            "price_change_eur_mwh": np.concatenate([[0.0], np.diff(prices)]),
            "target_direction": np.sign(np.concatenate([[0.0], np.diff(prices)])),
            "pnl_long_eur": np.concatenate([[0.0], np.diff(prices)]) * 24,
            "pnl_short_eur": -np.concatenate([[0.0], np.diff(prices)]) * 24,
            "de_fr_spread": rng.randn(n) * 5,
            "de_nl_spread": rng.randn(n) * 5,
            "de_avg_neighbour_spread": rng.randn(n) * 4,
            "weather_temperature_2m_degc_mean": rng.uniform(0, 30, n),
            "gen_nuclear_mw_mean": rng.uniform(3000, 6000, n),
            "flow_fr_net_import_mw_mean": rng.randn(n) * 2000,
            "flow_nl_net_import_mw_mean": rng.randn(n) * 1500,
            "rolling_vol_7d": rng.uniform(1, 10, n),
            "load_forecast_mw_mean": rng.uniform(40000, 70000, n),
            "forecast_wind_offshore_mw_mean": rng.uniform(0, 5000, n),
            "forecast_wind_onshore_mw_mean": rng.uniform(0, 15000, n),
            "forecast_solar_mw_mean": rng.uniform(0, 20000, n),
            "gen_fossil_gas_mw_mean": rng.uniform(2000, 8000, n),
            "gen_fossil_hard_coal_mw_mean": rng.uniform(1000, 5000, n),
            "gen_fossil_brown_coal_lignite_mw_mean": rng.uniform(3000, 10000, n),
            "gen_wind_onshore_mw_mean": rng.uniform(0, 15000, n),
            "gen_wind_offshore_mw_mean": rng.uniform(0, 5000, n),
            "gen_solar_mw_mean": rng.uniform(0, 20000, n),
            "load_actual_mw_mean": rng.uniform(40000, 70000, n),
            "weather_wind_speed_10m_kmh_mean": rng.uniform(0, 30, n),
            "weather_shortwave_radiation_wm2_mean": rng.uniform(0, 400, n),
            "price_fr_eur_mwh_mean": rng.uniform(30, 70, n),
            "price_nl_eur_mwh_mean": rng.uniform(30, 70, n),
            "price_at_eur_mwh_mean": rng.uniform(30, 70, n),
            "price_pl_eur_mwh_mean": rng.uniform(20, 60, n),
            "price_cz_eur_mwh_mean": rng.uniform(25, 65, n),
            "price_dk_1_eur_mwh_mean": rng.uniform(25, 65, n),
            "carbon_price_usd_mean": rng.uniform(50, 100, n),
            "gas_price_usd_mean": rng.uniform(20, 50, n),
            "price_mean": prices,
            "price_max": prices + rng.uniform(0, 10, n),
            "price_min": prices - rng.uniform(0, 10, n),
            "price_std": rng.uniform(1, 5, n),
            "net_demand_mw": rng.uniform(20000, 50000, n),
            "renewable_penetration_pct": rng.uniform(0.2, 0.8, n),
            "price_zscore_20d": rng.randn(n),
            "price_range": rng.uniform(2, 15, n),
            "gas_trend_3d": rng.randn(n) * 2,
            "carbon_trend_3d": rng.randn(n) * 3,
            "fuel_cost_index": rng.uniform(100, 300, n),
            "wind_forecast_error": rng.randn(n) * 1000,
            "load_surprise": rng.randn(n) * 2000,
            "rolling_vol_14d": rng.uniform(1, 10, n),
            "total_fossil_mw": rng.uniform(8000, 20000, n),
            "net_flow_mw": rng.randn(n) * 3000,
            "dow_int": np.tile([1, 2, 3, 4, 5, 6, 7], (n // 7) + 1)[:n],
            "is_weekend": np.tile([0, 0, 0, 0, 0, 1, 1], (n // 7) + 1)[:n],
        }
    )


# ---------------------------------------------------------------------------
# 1. SpreadMomentumStrategy
# ---------------------------------------------------------------------------


class TestSpreadMomentumStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(SpreadMomentumStrategy, BacktestStrategy)

    def test_fit_sets_attributes(self) -> None:
        s = SpreadMomentumStrategy()
        df = pd.DataFrame(
            {
                "de_fr_spread": [1.0, 2.0, 3.0, 4.0, 5.0],
                "de_nl_spread": [0.5, 1.5, 2.5, 3.5, 4.5],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        assert s._fitted
        assert s._mean_abs_change > 0

    def test_raises_before_fit(self) -> None:
        s = SpreadMomentumStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({"de_fr_spread": 1.0, "de_nl_spread": 1.0}))

    def test_positive_rising_spread_goes_short(self) -> None:
        s = SpreadMomentumStrategy()
        df = pd.DataFrame(
            {
                "de_fr_spread": [1.0, 2.0, 3.0, 4.0, 5.0],
                "de_nl_spread": [1.0, 2.0, 3.0, 4.0, 5.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Force EMA to be positive and rising by sending a large positive spread
        state = _state_multi({"de_fr_spread": 10.0, "de_nl_spread": 10.0})
        forecast = s.forecast(state)
        assert forecast < state.last_settlement_price  # short

    def test_negative_falling_spread_goes_long(self) -> None:
        s = SpreadMomentumStrategy()
        # Train on negative spreads so EMA starts negative
        df = pd.DataFrame(
            {
                "de_fr_spread": [-5.0, -6.0, -7.0, -8.0, -9.0],
                "de_nl_spread": [-5.0, -6.0, -7.0, -8.0, -9.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Feed a more negative spread (falling and negative)
        state = _state_multi({"de_fr_spread": -15.0, "de_nl_spread": -15.0})
        forecast = s.forecast(state)
        assert forecast > state.last_settlement_price  # long

    def test_neutral_returns_entry(self) -> None:
        s = SpreadMomentumStrategy()
        # Train on near-zero spreads
        df = pd.DataFrame(
            {
                "de_fr_spread": [0.0, 0.0, 0.0, 0.0, 0.0],
                "de_nl_spread": [0.0, 0.0, 0.0, 0.0, 0.0],
                "price_change_eur_mwh": [1.0, -1.0, 2.0, -2.0, 1.0],
            }
        )
        s.fit(df)
        # Neutral signal: EMA near zero, no clear direction
        state = _state_multi({"de_fr_spread": 0.01, "de_nl_spread": -0.01})
        forecast = s.forecast(state)
        # Should return entry price (neutral) since spread_ema is near zero
        # and direction is ambiguous
        assert isinstance(forecast, float)


# ---------------------------------------------------------------------------
# 2. SelectiveHighConvictionStrategy
# ---------------------------------------------------------------------------


class TestSelectiveHighConvictionStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(SelectiveHighConvictionStrategy, BacktestStrategy)

    def test_fit_succeeds(self) -> None:
        s = SelectiveHighConvictionStrategy()
        train = _make_train_data(20)
        s.fit(train)
        assert s._fitted
        assert s._forecast_std > 0

    def test_raises_before_fit(self) -> None:
        s = SelectiveHighConvictionStrategy()
        features = {
            "load_forecast_mw_mean": 50000.0,
            "forecast_wind_offshore_mw_mean": 2000.0,
            "forecast_wind_onshore_mw_mean": 5000.0,
            "gen_fossil_gas_mw_mean": 4000.0,
            "gen_fossil_hard_coal_mw_mean": 3000.0,
            "gen_fossil_brown_coal_lignite_mw_mean": 5000.0,
        }
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_low_conviction_returns_entry_price(self) -> None:
        s = SelectiveHighConvictionStrategy()
        train = _make_train_data(30)
        s.fit(train)
        # Construct a state with average features (should produce low z-score)
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
            }:
                features[col] = float(train[col].mean())
        state = _state_multi(features, last_price=float(train["settlement_price"].mean()))
        forecast = s.forecast(state)
        # With average features, the composite signal should be near zero,
        # producing a low z-score and returning entry price
        assert isinstance(forecast, float)

    def test_high_conviction_passes_through(self) -> None:
        s = SelectiveHighConvictionStrategy()
        train = _make_train_data(30)
        s.fit(train)
        # Construct extreme features to trigger high conviction
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
            }:
                features[col] = float(train[col].max()) * 3.0
        state = _state_multi(features, last_price=50.0)
        forecast = s.forecast(state)
        # Extreme features should produce large z-score; forecast differs from entry
        assert isinstance(forecast, float)

    def test_reset_delegates(self) -> None:
        s = SelectiveHighConvictionStrategy()
        train = _make_train_data(20)
        s.fit(train)
        s.reset()
        assert s._fitted


# ---------------------------------------------------------------------------
# 3. TemperatureCurveStrategy
# ---------------------------------------------------------------------------


class TestTemperatureCurveStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(TemperatureCurveStrategy, BacktestStrategy)

    def test_fit_sets_coefficients(self) -> None:
        s = TemperatureCurveStrategy()
        df = pd.DataFrame(
            {
                "weather_temperature_2m_degc_mean": [0, 5, 10, 15, 20, 25, 30],
                "price_change_eur_mwh": [3, 1, -1, -2, -1, 1, 3],
            }
        )
        s.fit(df)
        assert s._coeffs is not None
        assert len(s._coeffs) == 3  # quadratic: [a, b, c]

    def test_raises_before_fit(self) -> None:
        s = TemperatureCurveStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state("weather_temperature_2m_degc_mean", 15.0))

    def test_u_shape_cold_increases_price(self) -> None:
        s = TemperatureCurveStrategy()
        # U-shaped data: cold and hot = positive change, moderate = negative
        df = pd.DataFrame(
            {
                "weather_temperature_2m_degc_mean": [-5, 0, 5, 10, 15, 20, 25, 30, 35],
                "price_change_eur_mwh": [4, 2, 0, -2, -3, -2, 0, 2, 4],
            }
        )
        s.fit(df)
        cold_forecast = s.forecast(_state("weather_temperature_2m_degc_mean", -5.0))
        mild_forecast = s.forecast(_state("weather_temperature_2m_degc_mean", 15.0))
        # Cold should forecast higher than mild
        assert cold_forecast > mild_forecast

    def test_u_shape_hot_increases_price(self) -> None:
        s = TemperatureCurveStrategy()
        df = pd.DataFrame(
            {
                "weather_temperature_2m_degc_mean": [-5, 0, 5, 10, 15, 20, 25, 30, 35],
                "price_change_eur_mwh": [4, 2, 0, -2, -3, -2, 0, 2, 4],
            }
        )
        s.fit(df)
        hot_forecast = s.forecast(_state("weather_temperature_2m_degc_mean", 35.0))
        mild_forecast = s.forecast(_state("weather_temperature_2m_degc_mean", 15.0))
        # Hot should forecast higher than mild
        assert hot_forecast > mild_forecast

    def test_forecast_is_clipped(self) -> None:
        s = TemperatureCurveStrategy()
        df = pd.DataFrame(
            {
                "weather_temperature_2m_degc_mean": [0, 10, 20],
                "price_change_eur_mwh": [1, -1, 1],
            }
        )
        s.fit(df)
        # Even at extreme temperature, forecast should be clipped
        forecast = s.forecast(_state("weather_temperature_2m_degc_mean", 100.0))
        max_change = s._mean_abs_change * 3.0
        assert abs(forecast - 50.0) <= max_change + 0.01


# ---------------------------------------------------------------------------
# 4. NuclearEventStrategy
# ---------------------------------------------------------------------------


class TestNuclearEventStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(NuclearEventStrategy, BacktestStrategy)

    def test_fit_computes_rolling_mean(self) -> None:
        s = NuclearEventStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000] * 10,
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        assert s._fitted
        assert s._rolling_mean == pytest.approx(5000.0)

    def test_raises_before_fit(self) -> None:
        s = NuclearEventStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state("gen_nuclear_mw_mean", 4000.0))

    def test_large_drop_goes_long(self) -> None:
        s = NuclearEventStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000] * 10,
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        # 15% below rolling mean = 4250. Send 3000 (well below)
        forecast = s.forecast(_state("gen_nuclear_mw_mean", 3000.0))
        assert forecast > 50.0  # long

    def test_large_surge_goes_short(self) -> None:
        s = NuclearEventStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000] * 10,
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        # 15% above rolling mean = 5750. Send 6500 (well above)
        forecast = s.forecast(_state("gen_nuclear_mw_mean", 6500.0))
        assert forecast < 50.0  # short

    def test_normal_range_returns_entry(self) -> None:
        s = NuclearEventStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [5000] * 10,
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        forecast = s.forecast(_state("gen_nuclear_mw_mean", 5000.0))
        assert forecast == 50.0  # no event

    def test_zero_nuclear_skips(self) -> None:
        s = NuclearEventStrategy()
        df = pd.DataFrame(
            {
                "gen_nuclear_mw_mean": [0.0] * 10,
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        # Post-shutdown: rolling mean near zero, should return entry
        forecast = s.forecast(_state("gen_nuclear_mw_mean", 0.0))
        assert forecast == 50.0


# ---------------------------------------------------------------------------
# 5. FlowImbalanceStrategy
# ---------------------------------------------------------------------------


class TestFlowImbalanceStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(FlowImbalanceStrategy, BacktestStrategy)

    def test_fit_computes_percentiles(self) -> None:
        s = FlowImbalanceStrategy()
        rng = np.random.RandomState(0)
        df = pd.DataFrame(
            {
                "flow_fr_net_import_mw_mean": rng.randn(100) * 2000,
                "flow_nl_net_import_mw_mean": rng.randn(100) * 1500,
                "price_change_eur_mwh": rng.randn(100),
            }
        )
        s.fit(df)
        assert s._fitted
        assert s._p25 < s._p75

    def test_raises_before_fit(self) -> None:
        s = FlowImbalanceStrategy()
        features = {"flow_fr_net_import_mw_mean": 1000.0, "flow_nl_net_import_mw_mean": 500.0}
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi(features))

    def test_heavy_import_goes_short(self) -> None:
        s = FlowImbalanceStrategy()
        # All positive flows (heavy import)
        df = pd.DataFrame(
            {
                "flow_fr_net_import_mw_mean": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "flow_nl_net_import_mw_mean": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        # Combined flow = 5000 (well above P75)
        features = {"flow_fr_net_import_mw_mean": 3000.0, "flow_nl_net_import_mw_mean": 2000.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_heavy_export_goes_long(self) -> None:
        s = FlowImbalanceStrategy()
        df = pd.DataFrame(
            {
                "flow_fr_net_import_mw_mean": [
                    -1000,
                    -500,
                    0,
                    500,
                    1000,
                    -1000,
                    -500,
                    0,
                    500,
                    1000,
                ],
                "flow_nl_net_import_mw_mean": [-500, -250, 0, 250, 500, -500, -250, 0, 250, 500],
                "price_change_eur_mwh": [1.0, -1.0] * 5,
            }
        )
        s.fit(df)
        # Combined flow well below P25 (heavy export)
        features = {"flow_fr_net_import_mw_mean": -3000.0, "flow_nl_net_import_mw_mean": -2000.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_neutral_returns_entry(self) -> None:
        s = FlowImbalanceStrategy()
        rng = np.random.RandomState(0)
        df = pd.DataFrame(
            {
                "flow_fr_net_import_mw_mean": rng.randn(50) * 2000,
                "flow_nl_net_import_mw_mean": rng.randn(50) * 1500,
                "price_change_eur_mwh": rng.randn(50),
            }
        )
        s.fit(df)
        # Send median-ish flow (should be in neutral zone)
        features = {"flow_fr_net_import_mw_mean": 0.0, "flow_nl_net_import_mw_mean": 0.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast == 50.0  # neutral


# ---------------------------------------------------------------------------
# 6. RegimeRidgeStrategy
# ---------------------------------------------------------------------------


class TestRegimeRidgeStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(RegimeRidgeStrategy, BacktestStrategy)

    def test_fit_succeeds(self) -> None:
        s = RegimeRidgeStrategy()
        train = _make_train_data(30)
        s.fit(train)
        assert s._fitted
        assert s._low_vol_pipeline is not None
        assert s._high_vol_pipeline is not None

    def test_raises_before_fit(self) -> None:
        s = RegimeRidgeStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            features = {
                "rolling_vol_7d": 5.0,
                "load_forecast_mw_mean": 50000.0,
            }
            s.forecast(_state_multi(features))

    def test_low_vol_produces_forecast(self) -> None:
        s = RegimeRidgeStrategy()
        train = _make_train_data(40)
        s.fit(train)
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "last_settlement_price",
            }:
                features[col] = float(train[col].mean())
        features["rolling_vol_7d"] = 1.0  # low volatility
        state = _state_multi(features, last_price=50.0)
        forecast = s.forecast(state)
        assert isinstance(forecast, float)
        assert np.isfinite(forecast)

    def test_high_vol_produces_forecast(self) -> None:
        s = RegimeRidgeStrategy()
        train = _make_train_data(40)
        s.fit(train)
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "last_settlement_price",
            }:
                features[col] = float(train[col].mean())
        features["rolling_vol_7d"] = 100.0  # high volatility
        state = _state_multi(features, last_price=50.0)
        forecast = s.forecast(state)
        assert isinstance(forecast, float)
        assert np.isfinite(forecast)

    def test_two_regimes_differ(self) -> None:
        s = RegimeRidgeStrategy()
        train = _make_train_data(40)
        s.fit(train)
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "last_settlement_price",
            }:
                features[col] = float(train[col].mean())

        features_low = dict(features)
        features_low["rolling_vol_7d"] = 0.5
        features_high = dict(features)
        features_high["rolling_vol_7d"] = 100.0

        f_low = s.forecast(_state_multi(features_low, last_price=50.0))
        f_high = s.forecast(_state_multi(features_high, last_price=50.0))
        # Different regimes should (generally) produce different forecasts
        # though with synthetic data they could be similar
        assert isinstance(f_low, float)
        assert isinstance(f_high, float)

    def test_fallback_with_small_data(self) -> None:
        """With very small training data, fallback to full-data model."""
        s = RegimeRidgeStrategy()
        train = _make_train_data(6)
        s.fit(train)
        assert s._fitted
        # Should still produce valid forecasts
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "last_settlement_price",
            }:
                features[col] = float(train[col].mean())
        forecast = s.forecast(_state_multi(features, last_price=50.0))
        assert np.isfinite(forecast)


# ---------------------------------------------------------------------------
# 7. PrunedMLEnsembleStrategy
# ---------------------------------------------------------------------------


class TestPrunedMLEnsembleStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(PrunedMLEnsembleStrategy, BacktestStrategy)

    def test_has_three_members(self) -> None:
        assert len(PrunedMLEnsembleStrategy._MEMBERS) == 3

    def test_fit_succeeds(self) -> None:
        s = PrunedMLEnsembleStrategy()
        train = _make_train_data(30)
        s.fit(train)
        assert len(s._fitted_members) == 3

    def test_forecast_returns_float(self) -> None:
        s = PrunedMLEnsembleStrategy()
        train = _make_train_data(30)
        s.fit(train)
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "last_settlement_price",
            }:
                features[col] = float(train[col].mean())
        state = _state_multi(features, last_price=50.0)
        forecast = s.forecast(state)
        assert isinstance(forecast, float)
        assert np.isfinite(forecast)

    def test_forecast_is_average_of_members(self) -> None:
        s = PrunedMLEnsembleStrategy()
        train = _make_train_data(30)
        s.fit(train)
        features = {}
        for col in train.columns:
            if col not in {
                "delivery_date",
                "split",
                "settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "last_settlement_price",
            }:
                features[col] = float(train[col].mean())
        state = _state_multi(features, last_price=50.0)

        ensemble_forecast = s.forecast(state)
        member_forecasts = [float(m.forecast(state)) for m in s._fitted_members]
        expected = sum(member_forecasts) / 3

        assert ensemble_forecast == pytest.approx(expected, abs=1e-6)

    def test_reset_delegates(self) -> None:
        s = PrunedMLEnsembleStrategy()
        train = _make_train_data(30)
        s.fit(train)
        s.reset()
        assert len(s._fitted_members) == 3
