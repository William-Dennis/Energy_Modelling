"""Market-aware evaluation orchestrator.

Collects predictions from all strategies via the existing
:func:`~energy_modelling.backtest.runner.run_backtest`, then feeds
them into the synthetic futures market to produce market-adjusted PnL.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date

import pandas as pd

from energy_modelling.backtest.feature_engineering import add_derived_features
from energy_modelling.backtest.futures_market_engine import (
    FuturesMarketEquilibrium,
    run_futures_market,
)
from energy_modelling.backtest.runner import BacktestResult, run_backtest
from energy_modelling.backtest.scoring import compute_backtest_metrics
from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_STATE_EXCLUDE_COLUMNS = {
    "delivery_date",
    "split",
    "settlement_price",
    "price_change_eur_mwh",
    "target_direction",
    "pnl_long_eur",
    "pnl_short_eur",
}


@dataclass(frozen=True)
class FuturesMarketResult:
    """Full output of a market-aware evaluation run.

    Parameters
    ----------
    equilibrium:
        Convergence details from the synthetic futures market.
    market_results:
        Per-strategy backtest results with PnL recomputed against the
        converged market price.
    original_results:
        Per-strategy backtest results under the original
        ``last_settlement_price`` entry price (for comparison).
    strategy_forecasts:
        Raw forecasts ``{strategy_name: {date: forecast_price}}``.
        Persisted so that the market engine can be re-run with different
        configurations without re-fitting strategies.
    """

    equilibrium: FuturesMarketEquilibrium
    market_results: dict[str, BacktestResult]
    original_results: dict[str, BacktestResult]
    strategy_forecasts: dict[str, dict] = None  # type: ignore[assignment]


def _recompute_pnl_against_market(
    predictions: pd.Series,
    settlement_prices: pd.Series,
    market_prices: pd.Series,
) -> pd.Series:
    """Recompute daily PnL using market prices as the entry price.

    PnL_t = direction_t * (settlement_t - market_price_t) * 24
    """

    direction = predictions.reindex(market_prices.index).astype("Float64").fillna(0.0)
    price_change = settlement_prices.reindex(market_prices.index) - market_prices
    return (direction * price_change * 24.0).rename("pnl")


def _collect_forecasts(
    strategy: BacktestStrategy,
    eval_data: pd.DataFrame,
    full_data: pd.DataFrame,
) -> dict:
    """Call ``strategy.forecast()`` for each evaluation date.

    Returns a dict mapping ``date -> float`` for every evaluation date.
    All strategies are required to produce a forecast.
    """

    forecasts: dict = {}
    for delivery_date, row in eval_data.iterrows():
        features = row.drop(labels=list(_STATE_EXCLUDE_COLUMNS), errors="ignore").copy()
        state = BacktestState(
            delivery_date=delivery_date,
            last_settlement_price=float(row["last_settlement_price"]),
            features=features,
            history=full_data.loc[full_data.index < delivery_date].copy(),
        )
        forecasts[delivery_date] = float(strategy.forecast(state))
    return forecasts


def _run_single_strategy(
    name: str,
    factory: Callable[[], BacktestStrategy],
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
) -> tuple[str, BacktestResult, dict]:
    """Worker: fit strategy, collect predictions and forecasts.

    Designed to run in a subprocess via :class:`ProcessPoolExecutor`.
    Combines Phase 1 (run_backtest) and Phase 2b (_collect_forecasts) so the
    fitted strategy object is used in the same process and never pickled.

    Returns
    -------
    tuple of (name, BacktestResult, forecasts_dict)
    """
    from energy_modelling.backtest.feature_engineering import add_derived_features  # noqa: PLC0415
    from energy_modelling.backtest.runner import _normalise_daily_data  # noqa: PLC0415

    strategy = factory()
    result = run_backtest(
        strategy=strategy,
        daily_data=daily_data,
        training_end=training_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )

    # Prepare eval data for forecast collection (mirrors run_futures_market_evaluation)
    data = _normalise_daily_data(daily_data)
    data = add_derived_features(data)
    eval_mask = (data.index >= evaluation_start) & (data.index <= evaluation_end)
    eval_data = data.loc[eval_mask]

    forecasts = _collect_forecasts(strategy, eval_data, data)
    return name, result, forecasts


def run_futures_market_evaluation(
    strategy_factories: dict[str, Callable[[], BacktestStrategy]],
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
    max_iterations: int = 500,
    convergence_threshold: float = 0.01,
    convergence_window: int = 1,
    initial_market_prices: pd.Series | None = None,
    max_workers: int | None = None,
    running_avg_k: int | None = 30,
) -> FuturesMarketResult:
    """Run all strategies, then evaluate them under the synthetic market.

    1. Run each strategy through ``run_backtest`` to obtain
       direction predictions and original PnL.
    2. Collect forecasts from each strategy.
    3. Feed all forecasts into ``run_futures_market`` to find
       the equilibrium market price.
    4. Recompute each strategy's PnL against the market price.

    Parameters
    ----------
    max_workers:
        Number of worker processes for parallelism. ``None`` uses
        :class:`ProcessPoolExecutor` defaults (one per CPU).
        Set to 1 to disable parallelism (serial execution).
    convergence_window:
        Number of consecutive iterations that must all have
        ``delta < convergence_threshold`` before convergence is declared.
        Default 1 (Phase 8c winner: K=30 running-average absorbs oscillation
        so a single small step reliably signals true convergence).
    running_avg_k:
        Running-average window applied across iterations (Phase 8 E1/8c).
        Default 30 — converges on both 2024 and 2025 with lower MAE than K=5.
    """

    # Phase 1 + 2b: Fit strategies and collect forecasts (parallel)
    original_results: dict[str, BacktestResult] = {}
    strategy_forecasts: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_strategy,
                name,
                factory,
                daily_data,
                training_end,
                evaluation_start,
                evaluation_end,
            ): name
            for name, factory in strategy_factories.items()
        }
        for future in futures:
            name, result, forecasts = future.result()
            original_results[name] = result
            strategy_forecasts[name] = forecasts

    if not original_results:
        msg = "No strategies were evaluated successfully."
        raise ValueError(msg)

    # Phase 2: Extract ground truth from daily data
    # Normalise the daily data index to date objects
    data = daily_data.copy()
    if "delivery_date" in data.columns:
        data["delivery_date"] = pd.to_datetime(data["delivery_date"]).dt.date
        data = data.set_index("delivery_date", drop=False)
    else:
        data.index = pd.Index(pd.to_datetime(data.index).date, name="delivery_date")

    # Ensure derived features are present for _collect_forecasts().
    # add_derived_features() is idempotent.
    data = add_derived_features(data)

    eval_mask = (data.index >= evaluation_start) & (data.index <= evaluation_end)
    eval_data = data.loc[eval_mask]

    settlement_prices = eval_data["settlement_price"].astype(float)
    if initial_market_prices is None:
        initial_market_prices = eval_data["last_settlement_price"].astype(float)

    # Phase 3: Run market convergence
    equilibrium = run_futures_market(
        initial_market_prices=initial_market_prices,
        real_prices=settlement_prices,
        strategy_forecasts=strategy_forecasts,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        convergence_window=convergence_window,
        running_avg_k=running_avg_k,
    )

    # Phase 4: Recompute PnL for each strategy under market prices
    market_results: dict[str, BacktestResult] = {}

    for name, orig in original_results.items():
        market_pnl = _recompute_pnl_against_market(
            predictions=orig.predictions,
            settlement_prices=settlement_prices,
            market_prices=equilibrium.final_market_prices,
        )
        cumulative = market_pnl.cumsum()
        trade_count = int(orig.predictions.notna().sum())
        metrics = compute_backtest_metrics(market_pnl, trade_count)

        market_results[name] = BacktestResult(
            predictions=orig.predictions,
            daily_pnl=market_pnl,
            cumulative_pnl=cumulative,
            trade_count=trade_count,
            days_evaluated=len(market_pnl),
            metrics=metrics,
        )

    return FuturesMarketResult(
        equilibrium=equilibrium,
        market_results=market_results,
        original_results=original_results,
        strategy_forecasts=strategy_forecasts,
    )
