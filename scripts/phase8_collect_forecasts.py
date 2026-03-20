"""Phase 8: Collect and persist strategy forecasts for experiment reuse.

Fits all strategies on 2024 and 2025 evaluation windows, collects their
price forecasts, and saves them as compact pickle files in data/results/phase8/.

These forecast files are the input to phase8_experiments.py, which sweeps
all oscillation-remedy configurations without re-fitting strategies.

Usage:
    uv run python scripts/phase8_collect_forecasts.py
    uv run python scripts/phase8_collect_forecasts.py --max-workers 4
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE8_DIR.mkdir(parents=True, exist_ok=True)


def _run_single(
    name: str,
    factory,
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
) -> tuple[str, dict, dict]:
    """Worker: fit strategy, collect both forecasts and predictions."""
    from energy_modelling.backtest.feature_engineering import add_derived_features
    from energy_modelling.backtest.futures_market_runner import (
        _STATE_EXCLUDE_COLUMNS,
        _collect_forecasts,
    )
    from energy_modelling.backtest.runner import _normalise_daily_data, run_backtest
    from energy_modelling.backtest.types import BacktestState

    strategy = factory()
    result = run_backtest(
        strategy=strategy,
        daily_data=daily_data,
        training_end=training_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )

    data = _normalise_daily_data(daily_data)
    data = add_derived_features(data)
    eval_mask = (data.index >= evaluation_start) & (data.index <= evaluation_end)
    eval_data = data.loc[eval_mask]

    forecasts = _collect_forecasts(strategy, eval_data, data)
    return name, forecasts, result.predictions.to_dict()


def collect_forecasts(
    year: int,
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
    max_workers: int | None = None,
) -> dict:
    """Collect forecasts from all strategies for a given eval window."""
    from energy_modelling.dashboard._backtest import STRATEGY_FACTORIES

    strategy_factories = STRATEGY_FACTORIES
    n = len(strategy_factories)
    logger.info("Collecting forecasts for %d (%d strategies)...", year, n)

    forecasts: dict[str, dict] = {}
    predictions: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single,
                name,
                factory,
                daily_data,
                training_end,
                evaluation_start,
                evaluation_end,
            ): name
            for name, factory in strategy_factories.items()
        }
        done = 0
        for future in futures:
            name, f, p = future.result()
            forecasts[name] = f
            predictions[name] = p
            done += 1
            if done % 10 == 0 or done == n:
                logger.info("  %d/%d done", done, n)

    # Also collect real prices and initial prices
    data_norm = _normalise_eval_data(daily_data)
    eval_mask = (data_norm.index >= evaluation_start) & (data_norm.index <= evaluation_end)
    eval_rows = data_norm.loc[eval_mask]

    real_prices = eval_rows["settlement_price"].astype(float)
    initial_prices = eval_rows["last_settlement_price"].astype(float)

    return {
        "year": year,
        "strategy_forecasts": forecasts,
        "strategy_predictions": predictions,
        "real_prices": real_prices,
        "initial_prices": initial_prices,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "training_end": training_end,
    }


def _normalise_eval_data(daily_data: pd.DataFrame) -> pd.DataFrame:
    from energy_modelling.backtest.feature_engineering import add_derived_features

    data = daily_data.copy()
    if "delivery_date" in data.columns:
        data["delivery_date"] = pd.to_datetime(data["delivery_date"]).dt.date
        data = data.set_index("delivery_date", drop=False)
    else:
        data.index = pd.Index(pd.to_datetime(data.index).date, name="delivery_date")
    data = add_derived_features(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 8 strategy forecasts.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    from energy_modelling.dashboard._backtest import (
        _resolve_path,
        combine_public_hidden,
        load_daily,
    )

    pub_path = _resolve_path("data/backtest/daily_public.csv")
    hid_path = _resolve_path("data/backtest/daily_hidden_test_full.csv")

    if not pub_path.exists():
        logger.error("Missing %s", pub_path)
        sys.exit(1)

    daily = load_daily(pub_path)

    # --- 2024 ---
    out_24 = PHASE8_DIR / "forecasts_2024.pkl"
    if out_24.exists():
        logger.info("forecasts_2024.pkl already exists — skipping (delete to rerun)")
    else:
        t0 = time.perf_counter()
        data_24 = collect_forecasts(
            year=2024,
            daily_data=daily,
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 12, 31),
            max_workers=args.max_workers,
        )
        with open(out_24, "wb") as f:
            pickle.dump(data_24, f)
        logger.info("Saved %s (%.1fs)", out_24, time.perf_counter() - t0)

    # --- 2025 ---
    if not hid_path.exists():
        logger.info("No hidden data — skipping 2025.")
        return

    out_25 = PHASE8_DIR / "forecasts_2025.pkl"
    if out_25.exists():
        logger.info("forecasts_2025.pkl already exists — skipping (delete to rerun)")
    else:
        t0 = time.perf_counter()
        combined = combine_public_hidden(daily, load_daily(hid_path))
        data_25 = collect_forecasts(
            year=2025,
            daily_data=combined,
            training_end=date(2024, 12, 31),
            evaluation_start=date(2025, 1, 1),
            evaluation_end=date(2025, 12, 31),
            max_workers=args.max_workers,
        )
        with open(out_25, "wb") as f:
            pickle.dump(data_25, f)
        logger.info("Saved %s (%.1fs)", out_25, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
