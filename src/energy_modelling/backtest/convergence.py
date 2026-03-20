"""Convergence analysis for the synthetic futures market.

Theoretical and empirical tools for analyzing market convergence
under the forecast-based market model (``docs/energy_market_spec.md``).

Phase 8e adds iteration-level smoothing functions that extract a stable
consensus price from an oscillating iteration trace without modifying
the engine's update rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_modelling.backtest.futures_market_engine import (
    FuturesMarketEquilibrium,
    run_futures_market,
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


def _build_pf_forecasts(real_prices: pd.Series) -> dict:
    """Build a PF forecast dict {date: real_price} for all dates."""
    return {t: float(real_prices.loc[t]) for t in real_prices.index}


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


def run_forecast_foresight_market(
    real_prices: pd.Series,
    initial_market_prices: pd.Series,
    max_iterations: int = 100,
    convergence_threshold: float = 0.01,
    other_forecasts: dict[str, dict] | None = None,
) -> FuturesMarketEquilibrium:
    """Run the market with PF providing real-price forecasts.

    PF's forecast is the real settlement price.  With PF as the sole
    strategy, the update rule is ``P_{k+1} = P_real`` (instant convergence
    in one iteration) because there is no dampening.

    With other strategies present, the price update is the profit-weighted
    average of all (profitable) strategies' forecasts.
    """
    all_forecasts: dict[str, dict] = {
        "PerfectForesight": _build_pf_forecasts(real_prices),
    }
    if other_forecasts:
        all_forecasts.update(other_forecasts)

    return run_futures_market(
        initial_market_prices=initial_market_prices,
        real_prices=real_prices,
        strategy_forecasts=all_forecasts,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )


# ---------------------------------------------------------------------------
# Phase 8e: Iteration-level smoothing
# ---------------------------------------------------------------------------


def average_last_k_iterations(
    equilibrium: FuturesMarketEquilibrium,
    k: int,
) -> pd.Series:
    """Average market prices over the last *k* iterations (Phase 8e, E1).

    For a 3-step limit cycle, ``k=3`` averages exactly one full period,
    cancelling the oscillation on each day and producing a price close
    to the midpoint of the two forecast poles.
    """
    iters = equilibrium.iterations
    if k > len(iters):
        k = len(iters)
    last_k = iters[-k:]
    prices = pd.DataFrame(
        {f"iter_{it.iteration}": it.market_prices for it in last_k}
    )
    return prices.mean(axis=1)


def ema_iteration_prices(
    equilibrium: FuturesMarketEquilibrium,
    beta: float = 0.3,
) -> pd.Series:
    """Exponential moving average across iterations (Phase 8e, E2).

    EMA_t^(k) = beta * P^m_t^(k) + (1 - beta) * EMA_t^(k-1)

    Lower *beta* values provide heavier smoothing.  The EMA converges
    to a stable value even when the underlying engine oscillates.
    """
    iters = equilibrium.iterations
    if not iters:
        return equilibrium.final_market_prices.copy()
    ema = iters[0].market_prices.copy().astype(float)
    for it in iters[1:]:
        ema = beta * it.market_prices + (1.0 - beta) * ema
    return ema


def best_iteration_prices(
    equilibrium: FuturesMarketEquilibrium,
) -> tuple[pd.Series, int]:
    """Select the iteration with the lowest convergence delta (Phase 8e, E3).

    Returns the market prices and iteration index of the most stable
    iteration — typically the "settle" phase of the limit cycle, which
    is empirically the most accurate.
    """
    iters = equilibrium.iterations
    if len(iters) <= 1:
        return iters[0].market_prices.copy(), iters[0].iteration

    best_iter = iters[0]
    best_delta = float("inf")
    for i in range(1, len(iters)):
        delta = float((iters[i].market_prices - iters[i - 1].market_prices).abs().max())
        if delta < best_delta:
            best_delta = delta
            best_iter = iters[i]
    return best_iter.market_prices.copy(), best_iter.iteration


def delta_weighted_average(
    equilibrium: FuturesMarketEquilibrium,
) -> pd.Series:
    """Average iteration prices weighted by inverse delta (Phase 8e, E4).

    Iterations with smaller max price change (more stable) receive
    higher weight, so the result tilts toward the "settle" phases of
    the limit cycle.

        weight_k = 1 / (1 + delta_k)
    """
    iters = equilibrium.iterations
    if not iters:
        return equilibrium.final_market_prices.copy()

    prices_list: list[pd.Series] = []
    weights_list: list[float] = []

    for i, it in enumerate(iters):
        if i == 0:
            w = 1.0
        else:
            delta = float(
                (it.market_prices - iters[i - 1].market_prices).abs().max()
            )
            w = 1.0 / (1.0 + delta)
        prices_list.append(it.market_prices)
        weights_list.append(w)

    total_w = sum(weights_list)
    result = sum(w * p for w, p in zip(weights_list, prices_list, strict=True)) / total_w
    return result
