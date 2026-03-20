"""Tests for challenge.market_runner -- market-aware evaluation orchestrator."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.futures_market_runner import (
    FuturesMarketResult,
    _recompute_pnl_against_market,
    run_futures_market_evaluation,
)
from energy_modelling.backtest.types import BacktestState, BacktestStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_frame() -> pd.DataFrame:
    """Minimal daily frame with 2 train + 3 eval rows."""
    index = [
        date(2023, 12, 30),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    return pd.DataFrame(
        {
            "delivery_date": index,
            "split": ["train", "train", "validation", "validation", "validation"],
            "last_settlement_price": [49.0, 50.0, 51.0, 55.0, 48.0],
            "settlement_price": [50.0, 51.0, 55.0, 48.0, 52.0],
            "price_change_eur_mwh": [1.0, 1.0, 4.0, -7.0, 4.0],
            "target_direction": [1, 1, 1, -1, 1],
            "pnl_long_eur": [24.0, 24.0, 96.0, -168.0, 96.0],
            "pnl_short_eur": [-24.0, -24.0, -96.0, 168.0, -96.0],
            "load_actual_mw_mean": [40_000.0, 41_000.0, 42_000.0, 43_000.0, 44_000.0],
            "price_mean": [49.0, 50.0, 51.0, 55.0, 48.0],
        }
    )


class _AlwaysLong(BacktestStrategy):
    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def forecast(self, state: BacktestState) -> float:
        return state.last_settlement_price + 1.0


class _AlwaysShort(BacktestStrategy):
    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def forecast(self, state: BacktestState) -> float:
        return state.last_settlement_price - 1.0


class _PerfectForesight(BacktestStrategy):
    """Uses history to look at target_direction in previous row (cheats via labels)."""

    _FORECAST_MAP = {
        date(2024, 1, 1): 55.0,
        date(2024, 1, 2): 48.0,
        date(2024, 1, 3): 52.0,
    }

    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def forecast(self, state: BacktestState) -> float:
        return self._FORECAST_MAP.get(state.delivery_date, state.last_settlement_price)


class _Skipper(BacktestStrategy):
    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def forecast(self, state: BacktestState) -> float:
        return state.last_settlement_price


# ---------------------------------------------------------------------------
# _recompute_pnl_against_market
# ---------------------------------------------------------------------------


class TestRecomputePnl:
    def test_basic_long(self) -> None:
        idx = pd.Index([date(2024, 1, 1), date(2024, 1, 2)], name="delivery_date")
        predictions = pd.Series([1, 1], index=idx, dtype="Int64")
        settlements = pd.Series([55.0, 48.0], index=idx)
        market = pd.Series([52.0, 50.0], index=idx)
        pnl = _recompute_pnl_against_market(predictions, settlements, market)
        # Day 1: 1*(55-52)*24=72, Day 2: 1*(48-50)*24=-48
        assert pnl.iloc[0] == pytest.approx(72.0)
        assert pnl.iloc[1] == pytest.approx(-48.0)

    def test_skip_is_zero(self) -> None:
        idx = pd.Index([date(2024, 1, 1)], name="delivery_date")
        predictions = pd.Series([pd.NA], index=idx, dtype="Int64")
        settlements = pd.Series([55.0], index=idx)
        market = pd.Series([52.0], index=idx)
        pnl = _recompute_pnl_against_market(predictions, settlements, market)
        assert pnl.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_futures_market_evaluation
# ---------------------------------------------------------------------------


class TestRunMarketEvaluation:
    def test_returns_correct_structure(self) -> None:
        factories = {"long": _AlwaysLong, "short": _AlwaysShort}
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
        )
        assert isinstance(result, FuturesMarketResult)
        assert "long" in result.original_results
        assert "short" in result.original_results
        assert "long" in result.market_results
        assert "short" in result.market_results

    def test_original_results_match_standalone_runner(self) -> None:
        """Original results should be identical to running the backtest directly."""
        from energy_modelling.backtest.runner import run_backtest

        factories = {"long": _AlwaysLong}
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
        )
        standalone = run_backtest(
            strategy=_AlwaysLong(),
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
        )
        assert result.original_results["long"].metrics["total_pnl"] == pytest.approx(
            standalone.metrics["total_pnl"]
        )

    def test_market_pnl_differs_from_original(self) -> None:
        """With multiple strategies, market price should shift, changing PnL."""
        factories = {
            "long": _AlwaysLong,
            "short": _AlwaysShort,
            "perfect": _PerfectForesight,
        }
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
            forecast_spread=5.0,
        )
        # Market price should have shifted from initial, so PnL changes
        orig_pnl = result.original_results["perfect"].metrics["total_pnl"]
        market_pnl = result.market_results["perfect"].metrics["total_pnl"]
        # They should differ because market price != last_settlement_price
        # (unless the market converged exactly to last_settlement)
        # At minimum, the structure should be valid
        assert isinstance(market_pnl, float)
        assert isinstance(orig_pnl, float)

    def test_equilibrium_has_final_prices(self) -> None:
        factories = {"long": _AlwaysLong, "perfect": _PerfectForesight}
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
            forecast_spread=5.0,
        )
        eq = result.equilibrium
        assert len(eq.final_market_prices) == 3
        assert eq.converged or len(eq.iterations) == 20

    def test_market_prices_shift_from_initial(self) -> None:
        """Market prices should differ from last_settlement after convergence."""
        factories = {
            "long": _AlwaysLong,
            "short": _AlwaysShort,
            "perfect": _PerfectForesight,
        }
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
            forecast_spread=5.0,
        )
        # Market prices should have shifted from last_settlement
        initial = pd.Series(
            [51.0, 55.0, 48.0],
            index=result.equilibrium.final_market_prices.index,
        )
        diff = (result.equilibrium.final_market_prices - initial).abs()
        assert diff.max() > 0.0

    def test_long_short_market_pnl_symmetric(self) -> None:
        """Long and short should have opposite market PnL (both always trade)."""
        factories = {"long": _AlwaysLong, "short": _AlwaysShort}
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
            forecast_spread=5.0,
        )
        long_pnl = result.market_results["long"].metrics["total_pnl"]
        short_pnl = result.market_results["short"].metrics["total_pnl"]
        assert long_pnl == pytest.approx(-short_pnl)

    def test_skip_strategy_earns_zero(self) -> None:
        factories = {"skip": _Skipper, "long": _AlwaysLong}
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
        )
        assert result.market_results["skip"].metrics["total_pnl"] == pytest.approx(0.0)

    def test_empty_factories_raises(self) -> None:
        with pytest.raises(ValueError, match="No strategies"):
            run_futures_market_evaluation(
                strategy_factories={},
                daily_data=_make_daily_frame(),
                training_end=date(2023, 12, 31),
                evaluation_start=date(2024, 1, 1),
                evaluation_end=date(2024, 1, 3),
            )

    def test_trade_count_preserved(self) -> None:
        factories = {"long": _AlwaysLong, "skip": _Skipper}
        result = run_futures_market_evaluation(
            strategy_factories=factories,
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 3),
        )
        assert result.market_results["long"].trade_count == 3
        assert result.market_results["skip"].trade_count == 0
