"""Regenerate market_2024.pkl and market_2025.pkl with Phase 9 winner config.

Uses pre-collected forecasts from data/results/phase8/ so no strategy
re-fitting is needed.  Runs in seconds.

Winner config (Phase 9, R2_G6_T5.0_K15):
    weight_mode="softmax", softmax_temp=5.0, running_avg_k=15,
    monotone_window=5, convergence_threshold=0.01, max_iterations=500

Usage:
    uv run python scripts/regenerate_market_pkls.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from energy_modelling.backtest.futures_market_engine import run_futures_market
from energy_modelling.backtest.futures_market_runner import FuturesMarketResult
from energy_modelling.backtest.io import save_market_results
from energy_modelling.backtest.runner import BacktestResult, run_backtest
from energy_modelling.backtest.scoring import compute_backtest_metrics

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"

# Phase 9 winner config
WINNER_CFG = dict(
    weight_mode="softmax",
    softmax_temp=5.0,
    running_avg_k=15,
    monotone_window=5,
    convergence_threshold=0.01,
    max_iterations=500,
    convergence_window=1,
)


def _recompute_pnl(predictions, settlement_prices, market_prices):
    """Recompute daily PnL using market prices as entry price."""
    import pandas as pd

    direction = predictions.reindex(market_prices.index).astype("Float64").fillna(0.0)
    price_change = settlement_prices.reindex(market_prices.index) - market_prices
    return (direction * price_change * 24.0).rename("pnl")


def _rebuild_market_result(forecast_data: dict) -> FuturesMarketResult:
    """Run market engine on pre-collected forecasts and rebuild FuturesMarketResult."""
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from datetime import date as date_type

    from energy_modelling.backtest.feature_engineering import add_derived_features
    from energy_modelling.backtest.runner import _normalise_daily_data, run_backtest
    from energy_modelling.backtest.scoring import compute_backtest_metrics

    strategy_forecasts = forecast_data["strategy_forecasts"]
    strategy_predictions = forecast_data["strategy_predictions"]
    real_prices = forecast_data["real_prices"]
    initial_prices = forecast_data["initial_prices"]
    year = forecast_data["year"]
    evaluation_start = forecast_data["evaluation_start"]
    evaluation_end = forecast_data["evaluation_end"]

    print(f"  Running engine for {year} with winner config...")
    t0 = time.perf_counter()
    equilibrium = run_futures_market(
        initial_market_prices=initial_prices,
        real_prices=real_prices,
        strategy_forecasts=strategy_forecasts,
        **WINNER_CFG,
    )
    elapsed = time.perf_counter() - t0
    iters = len(equilibrium.iterations)
    print(
        f"  {year}: {iters} iterations, converged={equilibrium.converged}, "
        f"delta={equilibrium.convergence_delta:.4f}  ({elapsed:.2f}s)"
    )

    # Rebuild original BacktestResult objects from stored predictions
    # We need proper BacktestResult objects; reconstruct from stored predictions + real prices
    original_results: dict[str, BacktestResult] = {}
    market_results: dict[str, BacktestResult] = {}

    for name, pred_dict in strategy_predictions.items():
        import pandas as pd

        predictions = pd.Series(pred_dict).sort_index()
        predictions.index = pd.Index(
            [d if isinstance(d, date_type) else d.date() for d in predictions.index]
        )

        # original PnL: direction * (settlement - last_settlement)
        # We only have settlement_prices (real_prices) and initial_prices (last_settlement)
        direction = predictions.reindex(real_prices.index).astype("Float64").fillna(0.0)
        orig_pnl = (direction * (real_prices - initial_prices) * 24.0).rename("pnl")
        orig_cumulative = orig_pnl.cumsum()
        trade_count = int(predictions.notna().sum())
        orig_metrics = compute_backtest_metrics(orig_pnl, trade_count)

        original_results[name] = BacktestResult(
            predictions=predictions,
            daily_pnl=orig_pnl,
            cumulative_pnl=orig_cumulative,
            trade_count=trade_count,
            days_evaluated=len(orig_pnl),
            metrics=orig_metrics,
        )

        # market PnL: direction * (settlement - market_price)
        market_pnl = _recompute_pnl(predictions, real_prices, equilibrium.final_market_prices)
        market_cumulative = market_pnl.cumsum()
        market_metrics = compute_backtest_metrics(market_pnl, trade_count)

        market_results[name] = BacktestResult(
            predictions=predictions,
            daily_pnl=market_pnl,
            cumulative_pnl=market_cumulative,
            trade_count=trade_count,
            days_evaluated=len(market_pnl),
            metrics=market_metrics,
        )

    return FuturesMarketResult(
        equilibrium=equilibrium,
        market_results=market_results,
        original_results=original_results,
        strategy_forecasts=strategy_forecasts,
    )


def main() -> None:
    print("=" * 60)
    print("Regenerating market PKLs with Phase 9 winner config")
    print(f"  weight_mode=softmax, T=5.0, K=15, monotone_window=5")
    print("=" * 60)

    for year, pkl_name, forecast_pkl in [
        (2024, "market_2024.pkl", "forecasts_2024.pkl"),
        (2025, "market_2025.pkl", "forecasts_2025.pkl"),
    ]:
        forecast_path = PHASE8_DIR / forecast_pkl
        if not forecast_path.exists():
            print(f"  SKIP {year}: {forecast_path} not found")
            continue

        print(f"\n--- {year} ---")
        with open(forecast_path, "rb") as f:
            forecast_data = pickle.load(f)

        result = _rebuild_market_result(forecast_data)

        out_path = RESULTS_DIR / pkl_name
        save_market_results(result, out_path)
        print(f"  Saved -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
