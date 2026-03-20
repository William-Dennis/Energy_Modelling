"""Synthetic Futures Market engine for strategy aggregation.

Implements a prediction-market-style mechanism that aggregates multiple
strategy forecasts into a consensus market price.  Strategies are weighted
by their profitability against the real settlement price, and only profitable
strategies influence the next market price.

The market price replaces ``last_settlement_price`` as the entry price for
PnL computation, rewarding strategies that are correct *and* differentiated
from the consensus.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuturesMarketIteration:
    """Snapshot of one market-simulation iteration.

    Parameters
    ----------
    iteration:
        Zero-based iteration index.
    market_prices:
        The consensus market price per delivery date after this iteration.
    strategy_profits:
        Total profit of each strategy over the evaluation window.
    strategy_weights:
        Normalised weight of each strategy (zero for unprofitable ones).
    active_strategies:
        Names of strategies that contributed (positive profit).
    """

    iteration: int
    market_prices: pd.Series
    strategy_profits: dict[str, float]
    strategy_weights: dict[str, float]
    active_strategies: list[str]


@dataclass(frozen=True)
class FuturesMarketEquilibrium:
    """Result of running the market to convergence.

    Parameters
    ----------
    iterations:
        Full history of every iteration snapshot.
    final_market_prices:
        Converged (or last) market prices per delivery date.
    final_weights:
        Strategy weights at the final iteration.
    converged:
        Whether the price series stabilised within the tolerance.
    convergence_delta:
        Maximum absolute price change between the last two iterations.
    """

    iterations: list[FuturesMarketIteration]
    final_market_prices: pd.Series
    final_weights: dict[str, float]
    converged: bool
    convergence_delta: float


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------


def compute_strategy_profits(
    directions: dict[str, pd.Series],
    market_prices: pd.Series,
    real_prices: pd.Series,
    strategy_forecasts: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Compute total profit for each strategy against the market price.

    When a strategy provides explicit price forecasts via ``strategy_forecasts``,
    the profit is computed as a forecast-alignment bonus:

        r_{i,t} = sign(forecast_{i,t} - market_t) * (real_t - market_t) * 24

    This makes profit adaptive to the *current* market price rather than
    frozen binary directions from the initial backtest.  Strategies whose
    forecasts match the direction of the real price movement are rewarded;
    those that forecast in the wrong direction are penalised.

    Without forecasts, the legacy formula is used:

        r_{i,t} = direction_{i,t} * (real_t - market_t) * 24

    Parameters
    ----------
    directions:
        ``{strategy_name: series_of_directions}`` where values are +1, -1,
        or ``pd.NA``/``None`` for skip days.
    market_prices:
        Current market price per delivery date (same index as directions).
    real_prices:
        Ground-truth settlement price per delivery date.
    strategy_forecasts:
        Optional ``{strategy_name: {date: forecast_price}}`` of explicit
        forecasts.  When present, adaptive profit is used instead of frozen
        directions.

    Returns
    -------
    dict mapping strategy name to total profit (float).
    """

    if strategy_forecasts is None:
        strategy_forecasts = {}

    profits: dict[str, float] = {}
    for name, direction_series in directions.items():
        forecasts = strategy_forecasts.get(name)
        price_change = real_prices.reindex(market_prices.index) - market_prices

        if forecasts:
            # Adaptive: derive direction from forecast vs current market
            import numpy as np  # noqa: PLC0415

            adaptive_dir = pd.Series(
                {
                    t: float(np.sign(float(forecasts[t]) - float(market_prices.loc[t])))
                    if t in forecasts
                    else 0.0
                    for t in market_prices.index
                },
                dtype=float,
            )
            daily_pnl = adaptive_dir * price_change * 24.0
        else:
            d = direction_series.reindex(market_prices.index).astype("Float64")
            daily_pnl = d.fillna(0.0) * price_change * 24.0

        profits[name] = float(daily_pnl.sum())
    return profits


def compute_weights(
    strategy_profits: dict[str, float],
) -> dict[str, float]:
    """Normalise profits into non-negative weights that sum to 1.

    Only strategies with strictly positive total profit receive weight.
    If all strategies are non-positive, returns uniform zero weights.
    """

    raw = {name: max(profit, 0.0) for name, profit in strategy_profits.items()}
    total = sum(raw.values())
    if total == 0.0:
        return {name: 0.0 for name in raw}
    return {name: w / total for name, w in raw.items()}


def compute_market_prices(
    directions: dict[str, pd.Series],
    weights: dict[str, float],
    current_market_prices: pd.Series,
    forecast_spread: float,
    strategy_forecasts: dict[str, dict] | None = None,
) -> pd.Series:
    """Compute new market prices as a weighted average of implied forecasts.

    When a strategy provides an explicit forecast for a day via
    ``strategy_forecasts``, that value is used directly.  Otherwise the
    implied forecast is synthesised as ``market_price ± spread`` from the
    direction (the legacy behaviour).

    Skipped days (None/NA) are excluded from that day's aggregation.
    If no strategy has weight for a given day, the price carries forward
    from ``current_market_prices``.
    """

    if strategy_forecasts is None:
        strategy_forecasts = {}

    index = current_market_prices.index
    new_prices = current_market_prices.copy().astype(float)

    for t in index:
        numerator = 0.0
        denominator = 0.0
        for name, w in weights.items():
            if w <= 0.0:
                continue
            d_series = directions[name]
            if t not in d_series.index:
                continue
            d = d_series.loc[t]
            if pd.isna(d) or d == 0:
                continue

            # Use actual forecast when available, fall back to synthesis
            forecast = strategy_forecasts.get(name, {}).get(t)
            if forecast is not None:
                implied_forecast = float(forecast)
            else:
                implied_forecast = float(current_market_prices.loc[t]) + float(d) * forecast_spread

            numerator += w * implied_forecast
            denominator += w
        if denominator > 0.0:
            new_prices.loc[t] = numerator / denominator

    return new_prices


# ---------------------------------------------------------------------------
# Iteration and convergence
# ---------------------------------------------------------------------------


def run_futures_market_iteration(
    directions: dict[str, pd.Series],
    market_prices: pd.Series,
    real_prices: pd.Series,
    forecast_spread: float,
    iteration: int,
    strategy_forecasts: dict[str, dict] | None = None,
) -> FuturesMarketIteration:
    """Execute one full iteration of the synthetic futures market.

    Steps:
    1. Compute each strategy's profit against the current market price.
    2. Select and weight strategies (only profitable ones).
    3. Compute new market prices from the weighted implied forecasts.
    """

    profits = compute_strategy_profits(directions, market_prices, real_prices, strategy_forecasts)
    weights = compute_weights(profits)
    active = [name for name, w in weights.items() if w > 0.0]
    new_prices = compute_market_prices(
        directions, weights, market_prices, forecast_spread, strategy_forecasts
    )

    return FuturesMarketIteration(
        iteration=iteration,
        market_prices=new_prices,
        strategy_profits=profits,
        strategy_weights=weights,
        active_strategies=active,
    )


def run_futures_market(
    directions: dict[str, pd.Series],
    initial_market_prices: pd.Series,
    real_prices: pd.Series,
    max_iterations: int = 20,
    convergence_threshold: float = 0.01,
    forecast_spread: float | None = None,
    dampening: float = 0.5,
    strategy_forecasts: dict[str, dict] | None = None,
) -> FuturesMarketEquilibrium:
    """Run the synthetic futures market until prices converge.

    Parameters
    ----------
    directions:
        ``{strategy_name: series_of_+1/-1/None}`` per delivery date.
    initial_market_prices:
        Starting market prices, typically ``last_settlement_price``.
    real_prices:
        Ground-truth settlement prices for profit calculation.
    max_iterations:
        Hard cap on iteration count.
    convergence_threshold:
        Maximum absolute EUR/MWh change to declare convergence.
    forecast_spread:
        How far implied forecasts deviate from market price.  If *None*,
        auto-calibrated from the std of ``(real_prices - initial_market_prices)``.
    dampening:
        Blend factor in ``[0, 1]`` for the update rule:
        ``P_new = dampening * P_computed + (1 - dampening) * P_old``.
        Lower values slow convergence but improve stability.
    strategy_forecasts:
        ``{strategy_name: {date: forecast_price}}`` of explicit price
        forecasts.  When a strategy provides a forecast for a given day it
        is used directly in the weighted-average update; otherwise the
        engine falls back to direction ± spread synthesis.
    """

    if forecast_spread is None:
        price_changes = (real_prices - initial_market_prices).dropna()
        forecast_spread = float(price_changes.std()) if len(price_changes) > 1 else 1.0
        forecast_spread = max(forecast_spread, 0.1)  # floor to avoid near-zero

    current_prices = initial_market_prices.copy().astype(float)
    iterations: list[FuturesMarketIteration] = []
    converged = False
    delta = float("inf")

    for k in range(max_iterations):
        result = run_futures_market_iteration(
            directions=directions,
            market_prices=current_prices,
            real_prices=real_prices,
            forecast_spread=forecast_spread,
            iteration=k,
            strategy_forecasts=strategy_forecasts,
        )

        # Dampened update
        new_prices = dampening * result.market_prices + (1.0 - dampening) * current_prices
        delta = float((new_prices - current_prices).abs().max())

        # Store the *dampened* prices so dashboard charts show the actual
        # converging trajectory, not the raw undampened jump.
        dampened_snapshot = FuturesMarketIteration(
            iteration=result.iteration,
            market_prices=new_prices,
            strategy_profits=result.strategy_profits,
            strategy_weights=result.strategy_weights,
            active_strategies=result.active_strategies,
        )
        iterations.append(dampened_snapshot)

        current_prices = new_prices

        if delta < convergence_threshold:
            converged = True
            break

    return FuturesMarketEquilibrium(
        iterations=iterations,
        final_market_prices=current_prices,
        final_weights=iterations[-1].strategy_weights if iterations else {},
        converged=converged,
        convergence_delta=delta,
    )
