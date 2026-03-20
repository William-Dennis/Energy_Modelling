"""Convergence analysis for the synthetic futures market.

Theoretical and empirical tools for analyzing market convergence
under various strategy configurations (fixed/adaptive perfect foresight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_modelling.backtest.futures_market_engine import (
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
    compute_market_prices,
    compute_strategy_profits,
    compute_weights,
)


@dataclass(frozen=True)
class ConvergenceTrajectory:
    """Summary of a market convergence run.

    Parameters
    ----------
    n_iterations:
        Number of iterations executed.
    deltas:
        Max absolute price change per iteration.
    rmse_per_iteration:
        RMSE(market_price, real_price) per iteration.
    final_rmse:
        RMSE at the last iteration.
    converged:
        Whether the market engine declared convergence.
    """

    n_iterations: int
    deltas: list[float]
    rmse_per_iteration: list[float]
    final_rmse: float
    converged: bool


def fixed_perfect_foresight_directions(
    real_prices: pd.Series,
    initial_market_prices: pd.Series,
) -> pd.Series:
    """Compute static PF directions: sign(P_real - P_initial).

    These directions are computed ONCE and do not change across iterations.
    This is what would happen if a perfect foresight strategy submitted its
    predictions before the market runs.

    Returns +1 where real > initial, -1 where real < initial, +1 where equal
    (convention: tie-break to long).
    """
    diff = real_prices - initial_market_prices
    directions = np.sign(diff)
    # Replace 0 (exact tie) with +1 (convention)
    directions = directions.replace(0, 1)
    return directions.astype(int)


def adaptive_perfect_foresight_directions(
    real_prices: pd.Series,
    current_market_prices: pd.Series,
) -> pd.Series:
    """Compute dynamic PF directions: sign(P_real - P_current_market).

    These directions adapt to the current market price each iteration,
    always pointing toward the real price.
    """
    diff = real_prices - current_market_prices
    directions = np.sign(diff)
    directions = directions.replace(0, 1)
    return directions.astype(int)


def compute_theoretical_steps_to_arrival(
    distance: float,
    dampening: float,
    spread: float,
) -> int:
    """Number of iterations for fixed PF to reach or overshoot P_real.

    With a single PF strategy and dampening alpha, each iteration moves
    the market price by alpha * spread toward (or past) P_real. The number
    of steps to first arrival (or overshoot) is ceil(distance / (alpha * spread)).

    Parameters
    ----------
    distance:
        |P_real - P_initial|.
    dampening:
        Blend factor alpha in [0, 1].
    spread:
        Forecast spread.

    Returns
    -------
    Number of iterations (0 if distance is 0).
    """
    if distance <= 0.0:
        return 0
    step = dampening * spread
    if step <= 0.0:
        return 0  # degenerate case
    return math.ceil(distance / step)


def compute_overshoot_bias(
    distance: float,
    dampening: float,
    spread: float,
) -> float:
    """Compute the overshoot when fixed PF reaches P_real.

    After ceil(distance / step) iterations, the market price is at
    P_initial + steps * step. The overshoot is steps * step - distance.

    Returns
    -------
    Non-negative overshoot value.
    """
    if distance <= 0.0:
        return 0.0
    step = dampening * spread
    if step <= 0.0:
        return 0.0
    steps = math.ceil(distance / step)
    return steps * step - distance


def _compute_iteration_metrics(
    equilibrium: FuturesMarketEquilibrium,
    real_prices: pd.Series,
) -> tuple[list[float], list[float]]:
    """Compute per-iteration deltas and RMSE values."""
    deltas: list[float] = []
    rmses: list[float] = []

    for i, it in enumerate(equilibrium.iterations):
        aligned_real = real_prices.reindex(it.market_prices.index)
        residuals = it.market_prices - aligned_real
        rmses.append(float(np.sqrt((residuals**2).mean())))

        if i == 0:
            deltas.append(float("inf"))
        else:
            prev_prices = equilibrium.iterations[i - 1].market_prices
            deltas.append(float((it.market_prices - prev_prices).abs().max()))

    return deltas, rmses


def compute_convergence_trajectory(
    equilibrium: FuturesMarketEquilibrium,
    real_prices: pd.Series,
) -> ConvergenceTrajectory:
    """Extract convergence metrics from a completed market run.

    Parameters
    ----------
    equilibrium:
        Result of run_futures_market.
    real_prices:
        Ground-truth settlement prices.
    """
    deltas, rmses = _compute_iteration_metrics(equilibrium, real_prices)
    final_rmse = rmses[-1] if rmses else float("inf")

    return ConvergenceTrajectory(
        n_iterations=len(equilibrium.iterations),
        deltas=deltas,
        rmse_per_iteration=rmses,
        final_rmse=final_rmse,
        converged=equilibrium.converged,
    )


def _init_adaptive_market(
    real_prices: pd.Series,
    initial_market_prices: pd.Series,
    forecast_spread: float | None,
) -> tuple[pd.Series, float]:
    """Initialize prices and spread for the adaptive market run."""
    if forecast_spread is None:
        price_changes = (real_prices - initial_market_prices).dropna()
        forecast_spread = float(price_changes.std()) if len(price_changes) > 1 else 1.0
        forecast_spread = max(forecast_spread, 0.1)
    return initial_market_prices.copy().astype(float), forecast_spread


def _run_adaptive_iteration(
    k: int,
    real_prices: pd.Series,
    current_prices: pd.Series,
    forecast_spread: float,
    dampening: float,
    other_directions: dict[str, pd.Series] | None,
) -> tuple[FuturesMarketIteration, pd.Series, float]:
    """Execute one iteration of the adaptive foresight market."""
    pf_directions = adaptive_perfect_foresight_directions(real_prices, current_prices)
    all_directions: dict[str, pd.Series] = {"PerfectForesight": pf_directions}
    if other_directions:
        all_directions.update(other_directions)

    profits = compute_strategy_profits(all_directions, current_prices, real_prices)
    weights = compute_weights(profits)
    active = [name for name, w in weights.items() if w > 0.0]
    new_raw_prices = compute_market_prices(
        all_directions, weights, current_prices, forecast_spread,
    )

    iteration = FuturesMarketIteration(
        iteration=k, market_prices=new_raw_prices,
        strategy_profits=profits, strategy_weights=weights,
        active_strategies=active,
    )

    new_prices = dampening * new_raw_prices + (1.0 - dampening) * current_prices
    delta = float((new_prices - current_prices).abs().max())
    return iteration, new_prices, delta


def run_adaptive_foresight_market(
    real_prices: pd.Series,
    initial_market_prices: pd.Series,
    max_iterations: int = 100,
    convergence_threshold: float = 0.01,
    forecast_spread: float | None = None,
    dampening: float = 0.5,
    other_directions: dict[str, pd.Series] | None = None,
) -> FuturesMarketEquilibrium:
    """Run the market with adaptive PF directions recomputed each iteration.

    Uses the legacy direction ± spread synthesis for implied forecasts.
    """
    current_prices, forecast_spread = _init_adaptive_market(
        real_prices, initial_market_prices, forecast_spread,
    )
    iterations = []
    converged = False
    delta = float("inf")

    for k in range(max_iterations):
        iteration, current_prices, delta = _run_adaptive_iteration(
            k, real_prices, current_prices, forecast_spread,
            dampening, other_directions,
        )
        iterations.append(iteration)
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


# ---------------------------------------------------------------------------
# Forecast-aware convergence analysis
# ---------------------------------------------------------------------------


def _build_pf_forecasts(real_prices: pd.Series) -> dict:
    """Build a PF forecast dict {date: real_price} for all dates."""
    return {t: float(real_prices.loc[t]) for t in real_prices.index}


def _run_forecast_iteration(
    k: int,
    real_prices: pd.Series,
    current_prices: pd.Series,
    dampening: float,
    other_directions: dict[str, pd.Series] | None,
    other_forecasts: dict[str, dict] | None,
) -> tuple[FuturesMarketIteration, pd.Series, float]:
    """Execute one iteration using real-valued forecasts (not direction ± spread).

    PF provides its actual forecast (= real price).  Other strategies may
    provide their own forecasts or fall back to direction ± spread.
    """
    pf_directions = adaptive_perfect_foresight_directions(real_prices, current_prices)
    all_directions: dict[str, pd.Series] = {"PerfectForesight": pf_directions}
    if other_directions:
        all_directions.update(other_directions)

    # Build combined forecast dict — PF always provides real prices
    all_forecasts: dict[str, dict] = {"PerfectForesight": _build_pf_forecasts(real_prices)}
    if other_forecasts:
        all_forecasts.update(other_forecasts)

    profits = compute_strategy_profits(all_directions, current_prices, real_prices)
    weights = compute_weights(profits)
    active = [name for name, w in weights.items() if w > 0.0]

    # forecast_spread=0.0 because all strategies should supply forecasts;
    # the spread is only used as a fallback for strategies without forecasts.
    new_raw_prices = compute_market_prices(
        all_directions, weights, current_prices,
        forecast_spread=0.0,
        strategy_forecasts=all_forecasts,
    )

    iteration = FuturesMarketIteration(
        iteration=k, market_prices=new_raw_prices,
        strategy_profits=profits, strategy_weights=weights,
        active_strategies=active,
    )

    new_prices = dampening * new_raw_prices + (1.0 - dampening) * current_prices
    delta = float((new_prices - current_prices).abs().max())
    return iteration, new_prices, delta


def run_forecast_foresight_market(
    real_prices: pd.Series,
    initial_market_prices: pd.Series,
    max_iterations: int = 100,
    convergence_threshold: float = 0.01,
    dampening: float = 0.5,
    other_directions: dict[str, pd.Series] | None = None,
    other_forecasts: dict[str, dict] | None = None,
) -> FuturesMarketEquilibrium:
    """Run the market with PF providing real-price forecasts each iteration.

    Unlike ``run_adaptive_foresight_market`` which uses direction ± spread,
    this function passes the PF strategy's actual forecast (= real settlement
    price) through ``strategy_forecasts``.  With PF as the sole strategy and
    dampening α, the update rule becomes:

        P_{k+1} = α * P_real + (1 - α) * P_k

    This is a contraction mapping with rate ``(1 - α)`` and guarantees
    geometric convergence to P_real regardless of the initial price gap.
    """
    current_prices = initial_market_prices.copy().astype(float)
    iterations: list = []
    converged = False
    delta = float("inf")

    for k in range(max_iterations):
        iteration, current_prices, delta = _run_forecast_iteration(
            k, real_prices, current_prices, dampening,
            other_directions, other_forecasts,
        )
        iterations.append(iteration)
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
