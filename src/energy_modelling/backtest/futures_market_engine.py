"""Synthetic Futures Market engine for strategy aggregation.

Implements the prediction-market model from ``docs/energy_market_spec.md``:

1. **Trading decision**: q_{i,t} = sign(forecast_{i,t} - P^m_t)
2. **Profit**: r_{i,t} = q_{i,t} * (P_real_t - P^m_t)
3. **Selection**: w_i = max(Pi_i, 0) / sum(max(Pi_j, 0))
4. **Price update**: P^m_{t}^(k+1) = alpha * sum_i w_i * forecast_{i,t}
                                      + (1 - alpha) * P^m_{t}^(k)
5. **Iteration**: repeat until convergence.

Every strategy must provide explicit price forecasts.  The market price
is the profit-weighted average of forecasts from profitable strategies.

Phase 8 extensions add dampening (alpha parameter), alternative weighting
schemes (log-profit, weight cap, weighted median), and cluster-aware
aggregation to resolve the 3-step limit cycle observed on real data.
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


# ---------------------------------------------------------------------------
# Phase 8c: Alternative weighting schemes
# ---------------------------------------------------------------------------


def compute_weights_capped(
    strategy_profits: dict[str, float],
    w_max: float = 0.10,
) -> dict[str, float]:
    """Weights with per-strategy cap (Phase 8c, Experiment C1).

    After computing normalised weights, clips any weight exceeding *w_max*
    and redistributes the excess proportionally among uncapped strategies.

    Parameters
    ----------
    strategy_profits:
        Total profit per strategy.
    w_max:
        Maximum allowed weight for any single strategy.  Must be in (0, 1].
    """
    raw = {name: max(profit, 0.0) for name, profit in strategy_profits.items()}
    total = sum(raw.values())
    if total == 0.0:
        return {name: 0.0 for name in raw}

    weights = {name: w / total for name, w in raw.items()}

    # Iterative clipping: each pass may push previously-uncapped weights
    # above w_max after redistribution.  100 iterations is a safe upper
    # bound — in practice convergence happens in < N_strategies passes.
    for _ in range(100):
        excess = sum(max(w - w_max, 0) for w in weights.values())
        if excess < 1e-10:
            break
        uncapped = {n: w for n, w in weights.items() if w <= w_max}
        capped = {n: w_max for n, w in weights.items() if w > w_max}
        uncapped_total = sum(uncapped.values())
        if uncapped_total > 0:
            scale = (uncapped_total + excess) / uncapped_total
            uncapped = {n: w * scale for n, w in uncapped.items()}
        weights = {**uncapped, **capped}

    return weights


def compute_weights_log(
    strategy_profits: dict[str, float],
) -> dict[str, float]:
    """Weights proportional to log(1 + profit) (Phase 8c, Experiment C3).

    Compresses the profit scale so that a 6,363 vs 968 profit difference
    maps to a 1.27:1 weight ratio instead of 6.6:1, reducing the
    winner-take-all dynamics that drive oscillation.
    """
    raw: dict[str, float] = {}
    for name, profit in strategy_profits.items():
        if profit > 0:
            raw[name] = float(np.log1p(profit))
        else:
            raw[name] = 0.0
    total = sum(raw.values())
    if total == 0.0:
        return {name: 0.0 for name in raw}
    return {name: w / total for name, w in raw.items()}


# ---------------------------------------------------------------------------
# Phase 8c: Weighted median and cluster-aware aggregation
# ---------------------------------------------------------------------------


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of *values* given *weights*.

    The weighted median is the value *m* such that the cumulative weight
    of values ≤ m first reaches or exceeds half the total weight.
    """
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative = np.cumsum(sorted_weights)
    half = 0.5 * cumulative[-1]
    median_idx = int(np.searchsorted(cumulative, half))
    median_idx = min(median_idx, len(sorted_values) - 1)
    return float(sorted_values[median_idx])


def compute_market_prices_median(
    weights: dict[str, float],
    strategy_forecasts: dict[str, dict],
    current_market_prices: pd.Series,
) -> pd.Series:
    """Market price as weighted median of forecasts (Phase 8c, Experiment C2).

    Replaces the weighted *mean* with the weighted *median*, making the
    market price robust to outlier forecasts on extreme-volatility days.
    """
    new_prices = current_market_prices.copy().astype(float)
    for t in current_market_prices.index:
        vals: list[float] = []
        wts: list[float] = []
        for name, w in weights.items():
            if w <= 0:
                continue
            f = strategy_forecasts.get(name, {}).get(t)
            if f is not None:
                vals.append(float(f))
                wts.append(w)
        if vals:
            new_prices.loc[t] = weighted_median(np.array(vals), np.array(wts))
    return new_prices


def detect_bimodal_clusters(
    forecasts: list[float],
    gap_threshold: float = 20.0,
) -> tuple[list[float], list[float]] | None:
    """Detect two forecast clusters using sorted-gap analysis (Phase 8c, C4).

    Returns two lists (low_cluster, high_cluster) if a gap ≥ *gap_threshold*
    separates them, otherwise ``None`` (unimodal).
    """
    if len(forecasts) < 2:
        return None
    sorted_f = sorted(forecasts)
    max_gap = 0.0
    max_gap_idx = 0
    for i in range(len(sorted_f) - 1):
        gap = sorted_f[i + 1] - sorted_f[i]
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i
    if max_gap >= gap_threshold:
        return sorted_f[: max_gap_idx + 1], sorted_f[max_gap_idx + 1 :]
    return None


# ---------------------------------------------------------------------------
# Core price computation (with dampening)
# ---------------------------------------------------------------------------


def compute_market_prices(
    weights: dict[str, float],
    strategy_forecasts: dict[str, dict],
    current_market_prices: pd.Series,
    alpha: float = 1.0,
) -> pd.Series:
    """Compute new market prices as dampened weighted average of forecasts.

    Dampened update rule (Phase 8b):

        P^m_{t}^(k+1) = alpha * (sum_i w_i * forecast_{i,t})
                       + (1 - alpha) * P^m_{t}^(k)

    When ``alpha = 1.0`` (default), this recovers the original spec-compliant
    undampened update.  The dampening is a *solver technique* (analogous to a
    learning rate) — the equilibrium price is still the spec-defined weighted
    average; we are simply using a numerically stabler path to reach it.

    Parameters
    ----------
    weights:
        Normalised strategy weights (from ``compute_weights`` or variants).
    strategy_forecasts:
        ``{strategy_name: {date: forecast_price}}``.
    current_market_prices:
        Market prices from the previous iteration.
    alpha:
        Dampening factor in (0, 1].  1.0 = no dampening (spec-compliant).
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
            undampened = numerator / denominator
            new_prices.loc[t] = alpha * undampened + (1.0 - alpha) * float(
                current_market_prices.loc[t]
            )

    return new_prices


# ---------------------------------------------------------------------------
# Phase 8b: Adaptive dampening helper
# ---------------------------------------------------------------------------


def adaptive_alpha(
    delta: float,
    target_delta: float = 1.0,
    alpha_max: float = 0.8,
    alpha_min: float = 0.1,
) -> float:
    """Compute dampening factor proportional to convergence gap (Phase 8b, B2).

    When the observed delta is large the step size is reduced; as the system
    approaches convergence the step size grows toward *alpha_max*.

    Parameters
    ----------
    delta:
        Observed max absolute price change from the previous iteration.
    target_delta:
        Reference delta that maps to ``alpha_max``.
    alpha_max:
        Upper bound on the dampening factor.
    alpha_min:
        Lower bound (prevents stalling when delta is very large).
    """
    if delta <= 0:
        return alpha_max
    alpha = min(alpha_max, target_delta / delta)
    return max(alpha_min, alpha)


# ---------------------------------------------------------------------------
# Iteration and convergence
# ---------------------------------------------------------------------------


def run_futures_market_iteration(
    market_prices: pd.Series,
    real_prices: pd.Series,
    iteration: int,
    strategy_forecasts: dict[str, dict],
    alpha: float = 1.0,
) -> FuturesMarketIteration:
    """Execute one full iteration of the synthetic futures market.

    Steps (matching ``energy_market_spec.md``):
    1. Derive trading decisions from forecasts vs current market prices.
    2. Compute each strategy's profit.
    3. Select and weight strategies (only profitable ones).
    4. Compute new market prices from the weighted forecasts (with optional
       dampening controlled by *alpha*).
    """
    profits = compute_strategy_profits(market_prices, real_prices, strategy_forecasts)
    weights = compute_weights(profits)
    active = [name for name, w in weights.items() if w > 0.0]
    new_prices = compute_market_prices(
        weights, strategy_forecasts, market_prices, alpha=alpha,
    )

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
    alpha: float = 1.0,
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
    alpha:
        Dampening factor in (0, 1].  ``1.0`` = no dampening (original
        spec-compliant behaviour).  Lower values blend the new weighted-
        average forecast with the previous market price, converting
        oscillatory limit cycles into convergent spirals (Phase 8b).
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
            alpha=alpha,
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


# ---------------------------------------------------------------------------
# Phase 8b: Two-phase convergence
# ---------------------------------------------------------------------------


def run_two_phase_market(
    initial_market_prices: pd.Series,
    real_prices: pd.Series,
    strategy_forecasts: dict[str, dict],
    phase1_alpha: float = 0.3,
    phase1_max_iter: int = 30,
    phase1_threshold: float = 1.0,
    phase2_max_iter: int = 20,
    phase2_threshold: float = 0.01,
) -> FuturesMarketEquilibrium:
    """Two-phase convergence: dampened warm-start then undampened refinement.

    Phase 1 uses dampening (``phase1_alpha``) to approach near-equilibrium.
    Phase 2 runs undampened (``alpha=1.0``) from the Phase-1 endpoint to
    find the exact spec-compliant fixed point (if one exists).

    If Phase 2 diverges beyond Phase 1's final delta, Phase 1's result is
    returned instead.
    """
    # Phase 1: dampened approach to near-equilibrium
    eq1 = run_futures_market(
        initial_market_prices=initial_market_prices,
        real_prices=real_prices,
        strategy_forecasts=strategy_forecasts,
        max_iterations=phase1_max_iter,
        convergence_threshold=phase1_threshold,
        alpha=phase1_alpha,
    )

    # Phase 2: undampened refinement from Phase 1 endpoint
    eq2 = run_futures_market(
        initial_market_prices=eq1.final_market_prices,
        real_prices=real_prices,
        strategy_forecasts=strategy_forecasts,
        max_iterations=phase2_max_iter,
        convergence_threshold=phase2_threshold,
        alpha=1.0,
    )

    if eq2.converged:
        return eq2
    # Phase 2 diverged — use whichever result is better
    if eq2.convergence_delta > eq1.convergence_delta:
        return eq1
    return eq2
