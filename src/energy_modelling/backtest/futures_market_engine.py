"""Synthetic Futures Market engine for strategy aggregation.

Implements the prediction-market model from ``docs/energy_market_spec.md``:

1. **Trading decision**: q_{i,t} = sign(forecast_{i,t} - P^m_t)
2. **Profit**: r_{i,t} = q_{i,t} * (P_real_t - P^m_t)
3. **Selection**: w_i = max(Pi_i, 0) / sum(max(Pi_j, 0))
4. **Price update**: P^m_{t}^(k+1) = sum_i w_i * forecast_{i,t}
5. **Iteration**: repeat until convergence.

Every strategy must provide explicit price forecasts.  The market price
is the profit-weighted average of forecasts from profitable strategies.

Convergence criterion
---------------------
The engine runs until the maximum absolute price change between consecutive
iterations falls below ``convergence_threshold``.  This is the direct
implementation of the spec Step 5 fixed-point criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

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
        Normalised weight of each strategy.
    active_strategies:
        Names of strategies that contributed (positive weight).
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
# Spec reference helpers (linear weighting, used by unit tests)
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

    Parameters
    ----------
    strategy_profits:
        Total profit per strategy.
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
    """Compute new market prices as weighted mean of forecasts (spec Step 4).

    P^m_{t}^(k+1) = sum_i w_i * forecast_{i,t}

    If no strategy has weight for a given day, the price carries forward
    from ``current_market_prices``.

    Parameters
    ----------
    weights:
        Normalised strategy weights (from :func:`compute_weights`).
    strategy_forecasts:
        ``{strategy_name: {date: forecast_price}}``.
    current_market_prices:
        Prices from the previous iteration (fallback).
    """
    index = current_market_prices.index
    new_prices = current_market_prices.copy().astype(float)

    for t in index:
        forecasts_t = []
        weights_t = []
        for name, w in weights.items():
            if w <= 0.0:
                continue
            forecast = strategy_forecasts.get(name, {}).get(t)
            if forecast is None:
                continue
            forecasts_t.append(float(forecast))
            weights_t.append(w)

        if not forecasts_t:
            continue

        total_w = sum(weights_t)
        if total_w == 0.0:
            continue
        new_prices.loc[t] = (
            sum(f * w for f, w in zip(forecasts_t, weights_t, strict=True)) / total_w
        )

    return new_prices


# ---------------------------------------------------------------------------
# Vectorised helpers (fast NumPy path used by run_futures_market)
# ---------------------------------------------------------------------------


class _ForecastMatrix(NamedTuple):
    """Pre-built (S × T) matrix for fast iteration.

    Attributes
    ----------
    F:
        Float64 array of shape ``(S, T)``.  ``NaN`` where a strategy has no
        forecast for that date.  Filled with the column mean for NaN entries so
        that the dot product is numerically safe (those strategies will get
        near-zero weight anyway because they have no directional opinion).
    strategy_names:
        List of strategy names, length S — row order matches ``F``.
    dates:
        DatetimeIndex of length T — column order matches ``F``.
    real_vec:
        Float64 array of shape ``(T,)`` — real prices aligned to ``dates``.
    """

    F: np.ndarray
    strategy_names: list[str]
    dates: pd.DatetimeIndex
    real_vec: np.ndarray


def _build_forecast_matrix(
    strategy_forecasts: dict[str, dict],
    dates: pd.Index,
    real_prices: pd.Series,
) -> _ForecastMatrix:
    """Pre-compute the ``(S × T)`` forecast matrix once before the iteration loop."""
    strategy_names = list(strategy_forecasts.keys())
    S = len(strategy_names)
    T = len(dates)
    date_idx: dict = {d: i for i, d in enumerate(dates)}

    F = np.full((S, T), np.nan, dtype=np.float64)
    for s, name in enumerate(strategy_names):
        fcs = strategy_forecasts[name]
        for d, v in fcs.items():
            idx = date_idx.get(d)
            if idx is not None and v is not None:
                F[s, idx] = float(v)

    # Fill NaN with column mean so the weighted dot product is numerically safe.
    col_means = np.nanmean(F, axis=0)
    nan_mask = np.isnan(F)
    F[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    real_vec = real_prices.reindex(dates).to_numpy(dtype=np.float64)

    return _ForecastMatrix(
        F=F,
        strategy_names=strategy_names,
        dates=pd.DatetimeIndex(dates),
        real_vec=real_vec,
    )


def _vec_iteration(
    market_vec: np.ndarray,
    fm: _ForecastMatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One vectorised market iteration implementing the spec exactly.

    Uses linear (spec) weighting: only profitable strategies receive weight,
    proportional to their total profit (spec Step 3).

    Parameters
    ----------
    market_vec:
        Current market prices, shape ``(T,)``.
    fm:
        Pre-built forecast matrix.

    Returns
    -------
    new_market_vec : np.ndarray, shape (T,)
    profits : np.ndarray, shape (S,)
    weights : np.ndarray, shape (S,)
    """
    F, real_vec = fm.F, fm.real_vec

    # Steps 1-2: trading direction and per-strategy profit
    direction = np.sign(F - market_vec[np.newaxis, :])  # (S, T)
    price_change = real_vec - market_vec  # (T,)
    profits = (direction * price_change[np.newaxis, :]).sum(axis=1)  # (S,)

    # Step 3: linear weights — only profitable strategies get weight
    raw = np.maximum(profits, 0.0)
    total = raw.sum()
    weights = raw / total if total > 0.0 else np.zeros_like(raw)

    # Step 4: weighted arithmetic mean of forecasts
    w_sum = weights.sum()
    new_market_vec = (weights @ F) / w_sum if w_sum > 0.0 else market_vec.copy()

    return new_market_vec, profits, weights


# ---------------------------------------------------------------------------
# Spec reference iteration (used by unit tests, slow path)
# ---------------------------------------------------------------------------


def run_futures_market_iteration(
    market_prices: pd.Series,
    real_prices: pd.Series,
    iteration: int,
    strategy_forecasts: dict[str, dict],
) -> FuturesMarketIteration:
    """Execute one full iteration of the synthetic futures market (spec, linear weights).

    Steps (matching ``energy_market_spec.md``):
    1. Derive trading decisions from forecasts vs current market prices.
    2. Compute each strategy's profit.
    3. Select and weight strategies (only profitable ones, linear spec weights).
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


# ---------------------------------------------------------------------------
# Main convergence loop
# ---------------------------------------------------------------------------


def run_futures_market(
    initial_market_prices: pd.Series,
    real_prices: pd.Series,
    strategy_forecasts: dict[str, dict],
    max_iterations: int = 500,
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
        Maximum absolute EUR/MWh change between consecutive iterations
        required to declare convergence (spec Step 5 fixed-point criterion).
    """
    current_prices = initial_market_prices.copy().astype(float)
    iterations: list[FuturesMarketIteration] = []
    converged = False
    delta = float("inf")

    # Pre-build (S × T) forecast matrix once — avoids repeated dict iteration
    fm = _build_forecast_matrix(strategy_forecasts, current_prices.index, real_prices)
    strategy_names = fm.strategy_names
    index = current_prices.index

    for k in tqdm(range(max_iterations), desc="Market simulation", unit="iter"):
        market_vec = current_prices.to_numpy(dtype=np.float64)

        new_vec, profits_arr, weights_arr = _vec_iteration(market_vec, fm)

        published = pd.Series(new_vec, index=index, name="market_price")

        profits_dict = dict(zip(strategy_names, profits_arr.tolist(), strict=True))
        weights_dict = dict(zip(strategy_names, weights_arr.tolist(), strict=True))
        active = [n for n, w in zip(strategy_names, weights_arr, strict=True) if w > 0.0]

        result = FuturesMarketIteration(
            iteration=k,
            market_prices=published,
            strategy_profits=profits_dict,
            strategy_weights=weights_dict,
            active_strategies=active,
        )

        delta = float(np.abs(new_vec - market_vec).max())
        iterations.append(result)
        current_prices = published

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
