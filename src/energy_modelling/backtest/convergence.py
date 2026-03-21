"""Convergence analysis for the synthetic futures market.

Theoretical and empirical tools for analyzing market convergence
under the forecast-based market model (``docs/energy_market_spec.md``).
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
    ema_alpha: float = 0.1,
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
        ema_alpha=ema_alpha,
    )
