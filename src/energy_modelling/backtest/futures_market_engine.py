"""Synthetic Futures Market engine for strategy aggregation.

Implements the prediction-market model from ``docs/energy_market_spec.md``:

1. **Trading decision**: q_{i,t} = sign(forecast_{i,t} - P^m_t)
2. **Profit**: r_{i,t} = q_{i,t} * (P_real_t - P^m_t)
3. **Selection**: w_i = max(Pi_i, 0) / sum(max(Pi_j, 0))
4. **Price update**: P^m_{t}^(k+1) = sum_i w_i * forecast_{i,t}
5. **Iteration**: repeat until convergence.

Every strategy must provide explicit price forecasts.  The market price
is the profit-weighted average of forecasts from profitable strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    market_prices: pd.Series,
    real_prices: pd.Series,
    strategy_forecasts: dict[str, dict],
) -> dict[str, float]:
    """Compute total profit for each strategy (spec Steps 1-2).

    For each strategy *i* on each day *t*:

        q_{i,t} = sign(forecast_{i,t} - market_t)
        r_{i,t} = q_{i,t} * (real_t - market_t)
        Pi_i    = sum_t r_{i,t}

    Parameters
    ----------
    market_prices:
        Current market price per delivery date.
    real_prices:
        Ground-truth settlement price per delivery date.
    strategy_forecasts:
        ``{strategy_name: {date: forecast_price}}``.

    Returns
    -------
    dict mapping strategy name to total profit (float).
    """
    price_change = real_prices.reindex(market_prices.index) - market_prices

    profits: dict[str, float] = {}
    for name, forecasts in strategy_forecasts.items():
        direction = pd.Series(
            {
                t: float(np.sign(float(forecasts[t]) - float(market_prices.loc[t])))
                if t in forecasts
                else 0.0
                for t in market_prices.index
            },
            dtype=float,
        )
        daily_pnl = direction * price_change
        profits[name] = float(daily_pnl.sum())
    return profits


def compute_weights(
    strategy_profits: dict[str, float],
) -> dict[str, float]:
    """Normalise profits into non-negative weights that sum to 1 (spec Step 3).

    Only strategies with strictly positive total profit receive weight.
    If all strategies are non-positive, returns uniform zero weights.
    """
    raw = {name: max(profit, 0.0) for name, profit in strategy_profits.items()}
    total = sum(raw.values())
    if total == 0.0:
        return {name: 0.0 for name in raw}
    return {name: w / total for name, w in raw.items()}


def compute_market_prices(
    weights: dict[str, float],
    strategy_forecasts: dict[str, dict],
    current_market_prices: pd.Series,
) -> pd.Series:
    """Compute new market prices as weighted average of forecasts (spec Step 4).

    P^m_{t}^(k+1) = sum_i w_i^{norm} * forecast_{i,t}

    If no strategy has weight for a given day, the price carries forward
    from ``current_market_prices``.
    """
    index = current_market_prices.index
    new_prices = current_market_prices.copy().astype(float)

    for t in index:
        numerator = 0.0
        denominator = 0.0
        for name, w in weights.items():
            if w <= 0.0:
                continue
            forecast = strategy_forecasts.get(name, {}).get(t)
            if forecast is None:
                continue
            numerator += w * float(forecast)
            denominator += w
        if denominator > 0.0:
            new_prices.loc[t] = numerator / denominator

    return new_prices


# ---------------------------------------------------------------------------
# Iteration and convergence
# ---------------------------------------------------------------------------


def run_futures_market_iteration(
    market_prices: pd.Series,
    real_prices: pd.Series,
    iteration: int,
    strategy_forecasts: dict[str, dict],
) -> FuturesMarketIteration:
    """Execute one full iteration of the synthetic futures market.

    Steps (matching ``energy_market_spec.md``):
    1. Derive trading decisions from forecasts vs current market prices.
    2. Compute each strategy's profit.
    3. Select and weight strategies (only profitable ones).
    4. Compute new market prices from the weighted forecasts.
    """
    profits = compute_strategy_profits(market_prices, real_prices, strategy_forecasts)
    weights = compute_weights(profits)
    active = [name for name, w in weights.items() if w > 0.0]
    new_prices = compute_market_prices(weights, strategy_forecasts, market_prices)

    return FuturesMarketIteration(
        iteration=iteration,
        market_prices=new_prices,
        strategy_profits=profits,
        strategy_weights=weights,
        active_strategies=active,
    )


def run_futures_market(
    initial_market_prices: pd.Series,
    real_prices: pd.Series,
    strategy_forecasts: dict[str, dict],
    max_iterations: int = 20,
    convergence_threshold: float = 0.01,
) -> FuturesMarketEquilibrium:
    """Run the synthetic futures market until prices converge (spec Step 5).

    Parameters
    ----------
    initial_market_prices:
        Starting market prices, typically ``last_settlement_price``.
    real_prices:
        Ground-truth settlement prices for profit calculation.
    strategy_forecasts:
        ``{strategy_name: {date: forecast_price}}``.  Every strategy must
        provide forecasts for all delivery dates.
    max_iterations:
        Hard cap on iteration count.
    convergence_threshold:
        Maximum absolute EUR/MWh change to declare convergence.
    """
    current_prices = initial_market_prices.copy().astype(float)
    iterations: list[FuturesMarketIteration] = []
    converged = False
    delta = float("inf")

    for k in tqdm(range(max_iterations), desc="Market simulation", unit="iter"):
        result = run_futures_market_iteration(
            market_prices=current_prices,
            real_prices=real_prices,
            iteration=k,
            strategy_forecasts=strategy_forecasts,
        )

        delta = float((result.market_prices - current_prices).abs().max())
        iterations.append(result)
        current_prices = result.market_prices

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
