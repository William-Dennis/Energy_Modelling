"""Convergence analysis for the synthetic futures market (Phase 7).

Provides theoretical and empirical tools for analyzing whether and how
the iterative market weighting scheme converges under various strategy
configurations, with a focus on perfect foresight.

Key results:
- **Fixed PF**: Directions fixed at sign(P_real - P_initial). Converges
  but with overshoot bias bounded by alpha * spread.
- **Adaptive PF**: Directions re-computed each iteration as sign(P_real - P_market).
  Guarantees convergence to P_real (contraction mapping).
- **Opposing strategies**: Always Long + Always Short cause oscillation
  when their aggregate net profit changes sign across iterations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_modelling.challenge.market import (
    MarketEquilibrium,
    compute_market_prices,
    compute_strategy_profits,
    compute_weights,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Perfect foresight direction helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Theoretical formulas (single-day, single-strategy)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------


def compute_convergence_trajectory(
    equilibrium: MarketEquilibrium,
    real_prices: pd.Series,
) -> ConvergenceTrajectory:
    """Extract convergence metrics from a completed market run.

    Parameters
    ----------
    equilibrium:
        Result of run_market_to_convergence.
    real_prices:
        Ground-truth settlement prices.
    """
    deltas: list[float] = []
    rmses: list[float] = []

    for it in equilibrium.iterations:
        # Compute RMSE for this iteration's market prices
        aligned_real = real_prices.reindex(it.market_prices.index)
        residuals = it.market_prices - aligned_real
        rmse = float(np.sqrt((residuals**2).mean()))
        rmses.append(rmse)

    # Compute deltas from iteration-to-iteration price changes
    for i, it in enumerate(equilibrium.iterations):
        if i == 0:
            # Delta from initial to first iteration is stored in the equilibrium
            # We approximate using the first iteration's market prices
            deltas.append(float("inf"))  # First delta not meaningful
        else:
            prev_prices = equilibrium.iterations[i - 1].market_prices
            delta = float((it.market_prices - prev_prices).abs().max())
            deltas.append(delta)

    # Override: use the equilibrium's own delta tracking
    # Actually the dampened deltas are what matter, but we don't have per-iteration
    # dampened prices stored. Use the RMSE trajectory instead.

    final_rmse = rmses[-1] if rmses else float("inf")

    return ConvergenceTrajectory(
        n_iterations=len(equilibrium.iterations),
        deltas=deltas,
        rmse_per_iteration=rmses,
        final_rmse=final_rmse,
        converged=equilibrium.converged,
    )


# ---------------------------------------------------------------------------
# Adaptive perfect foresight market runner
# ---------------------------------------------------------------------------


def run_adaptive_foresight_market(
    real_prices: pd.Series,
    initial_market_prices: pd.Series,
    max_iterations: int = 100,
    convergence_threshold: float = 0.01,
    forecast_spread: float | None = None,
    dampening: float = 0.5,
    other_directions: dict[str, pd.Series] | None = None,
) -> MarketEquilibrium:
    """Run the market with an adaptive perfect foresight strategy.

    Unlike the standard market engine where directions are fixed, this
    implementation recomputes PF directions each iteration based on the
    current market price. This creates a contraction mapping that
    guarantees convergence to P_real.

    Parameters
    ----------
    real_prices:
        Ground-truth settlement prices.
    initial_market_prices:
        Starting market prices (typically last_settlement_price).
    max_iterations:
        Hard cap on iterations.
    convergence_threshold:
        Max absolute price change to declare convergence.
    forecast_spread:
        Implied forecast spread. Auto-calibrated if None.
    dampening:
        Blend factor for price updates.
    other_directions:
        Optional fixed-direction strategies to include alongside adaptive PF.

    Returns
    -------
    MarketEquilibrium with the convergence result.
    """
    from energy_modelling.challenge.market import MarketIteration

    if forecast_spread is None:
        price_changes = (real_prices - initial_market_prices).dropna()
        forecast_spread = float(price_changes.std()) if len(price_changes) > 1 else 1.0
        forecast_spread = max(forecast_spread, 0.1)

    current_prices = initial_market_prices.copy().astype(float)
    iterations: list[MarketIteration] = []
    converged = False
    delta = float("inf")

    for k in range(max_iterations):
        # Adaptive PF: recompute directions each iteration
        pf_directions = adaptive_perfect_foresight_directions(real_prices, current_prices)

        # Combine with other strategies
        all_directions: dict[str, pd.Series] = {"PerfectForesight": pf_directions}
        if other_directions:
            all_directions.update(other_directions)

        # Compute profits, weights, new prices
        profits = compute_strategy_profits(all_directions, current_prices, real_prices)
        weights = compute_weights(profits)
        active = [name for name, w in weights.items() if w > 0.0]
        new_raw_prices = compute_market_prices(
            all_directions,
            weights,
            current_prices,
            forecast_spread,
        )

        iteration = MarketIteration(
            iteration=k,
            market_prices=new_raw_prices,
            strategy_profits=profits,
            strategy_weights=weights,
            active_strategies=active,
        )
        iterations.append(iteration)

        # Dampened update
        new_prices = dampening * new_raw_prices + (1.0 - dampening) * current_prices
        delta = float((new_prices - current_prices).abs().max())
        current_prices = new_prices

        if delta < convergence_threshold:
            converged = True
            break

    return MarketEquilibrium(
        iterations=iterations,
        final_market_prices=current_prices,
        final_weights=iterations[-1].strategy_weights if iterations else {},
        converged=converged,
        convergence_delta=delta,
    )
