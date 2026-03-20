"""Synthetic Futures Market engine for strategy aggregation.

Implements the prediction-market model from ``docs/energy_market_spec.md``:

1. **Trading decision**: q_{i,t} = sign(forecast_{i,t} - P^m_t)
2. **Profit**: r_{i,t} = q_{i,t} * (P_real_t - P^m_t)
3. **Selection**: w_i = max(Pi_i, 0) / sum(max(Pi_j, 0))
4. **Price update**: P^m_{t}^(k+1) = sum_i w_i * forecast_{i,t}
5. **Iteration**: repeat until convergence.

Every strategy must provide explicit price forecasts.  The market price
is the profit-weighted average of forecasts from profitable strategies.

Experiment parameters (Phase 8 research, default values reproduce the spec):
- ``alpha`` (float, default 1.0): dampening factor.  New price is blended as
  ``alpha * candidate + (1-alpha) * current``.  1.0 = no dampening (spec).
- ``weight_mode`` (str, default ``"linear"``): how profits are mapped to weights.
  - ``"linear"``: raw profit proportional (spec).
  - ``"log"``: log(1 + profit) proportional, reduces winner-take-all effect.
  - ``"capped"``: linear but each weight is capped at ``weight_cap``.
- ``weight_cap`` (float, default 1.0): effective only when ``weight_mode="capped"``.
- ``price_mode`` (str, default ``"mean"``): aggregation of weighted forecasts.
  - ``"mean"``: weighted arithmetic mean (spec).
  - ``"median"``: weighted median, more robust to outlier forecasts.
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
    weight_mode: str = "linear",
    weight_cap: float = 1.0,
) -> dict[str, float]:
    """Normalise profits into non-negative weights that sum to 1 (spec Step 3).

    Only strategies with strictly positive total profit receive weight.
    If all strategies are non-positive, returns uniform zero weights.

    Parameters
    ----------
    strategy_profits:
        Total profit per strategy.
    weight_mode:
        ``"linear"`` (spec default): weight proportional to profit.
        ``"log"``: weight proportional to log(1 + profit), dampens
        winner-take-all effect.
        ``"capped"``: linear weights but each capped at ``weight_cap``
        before renormalisation.
    weight_cap:
        Maximum weight for any single strategy when ``weight_mode="capped"``.
        Values outside (0, 1] are clamped to that range.
    """
    if weight_mode == "log":
        raw = {name: float(np.log1p(max(profit, 0.0))) for name, profit in strategy_profits.items()}
    else:
        raw = {name: max(profit, 0.0) for name, profit in strategy_profits.items()}

    total = sum(raw.values())
    if total == 0.0:
        return {name: 0.0 for name in raw}

    normalised = {name: w / total for name, w in raw.items()}

    if weight_mode == "capped":
        cap = max(min(weight_cap, 1.0), 1e-9)
        capped = {name: min(w, cap) for name, w in normalised.items()}
        cap_total = sum(capped.values())
        if cap_total == 0.0:
            return {name: 0.0 for name in capped}
        return {name: w / cap_total for name, w in capped.items()}

    return normalised


def compute_market_prices(
    weights: dict[str, float],
    strategy_forecasts: dict[str, dict],
    current_market_prices: pd.Series,
    price_mode: str = "mean",
) -> pd.Series:
    """Compute new market prices as weighted aggregate of forecasts (spec Step 4).

    P^m_{t}^(k+1) = aggregate_i w_i * forecast_{i,t}

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
    price_mode:
        ``"mean"`` (spec default): weighted arithmetic mean.
        ``"median"``: weighted median — more robust to outlier forecasts.
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

        if price_mode == "median":
            # Weighted median: sort by forecast, find cumulative weight >= 0.5
            total_w = sum(weights_t)
            if total_w == 0.0:
                continue
            pairs = sorted(zip(forecasts_t, weights_t), key=lambda x: x[0])
            cumulative = 0.0
            median_val = pairs[0][0]
            for f_val, w_val in pairs:
                cumulative += w_val / total_w
                if cumulative >= 0.5:
                    median_val = f_val
                    break
            new_prices.loc[t] = median_val
        else:
            # "mean" — weighted arithmetic mean (spec default)
            total_w = sum(weights_t)
            if total_w == 0.0:
                continue
            new_prices.loc[t] = sum(f * w for f, w in zip(forecasts_t, weights_t)) / total_w

    return new_prices


# ---------------------------------------------------------------------------
# Iteration and convergence
# ---------------------------------------------------------------------------


def run_futures_market_iteration(
    market_prices: pd.Series,
    real_prices: pd.Series,
    iteration: int,
    strategy_forecasts: dict[str, dict],
    weight_mode: str = "linear",
    weight_cap: float = 1.0,
    price_mode: str = "mean",
) -> FuturesMarketIteration:
    """Execute one full iteration of the synthetic futures market.

    Steps (matching ``energy_market_spec.md``):
    1. Derive trading decisions from forecasts vs current market prices.
    2. Compute each strategy's profit.
    3. Select and weight strategies (only profitable ones).
    4. Compute new market prices from the weighted forecasts.
    """
    profits = compute_strategy_profits(market_prices, real_prices, strategy_forecasts)
    weights = compute_weights(profits, weight_mode=weight_mode, weight_cap=weight_cap)
    active = [name for name, w in weights.items() if w > 0.0]
    new_prices = compute_market_prices(
        weights, strategy_forecasts, market_prices, price_mode=price_mode
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
    weight_mode: str = "linear",
    weight_cap: float = 1.0,
    price_mode: str = "mean",
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
        Dampening factor in [0, 1].  The updated price is blended as
        ``alpha * candidate + (1 - alpha) * current``.  Default 1.0
        reproduces the original spec (no dampening).
    weight_mode:
        Profit-to-weight mapping.  ``"linear"`` (spec), ``"log"``,
        or ``"capped"``.
    weight_cap:
        Per-strategy weight cap when ``weight_mode="capped"``.
    price_mode:
        Forecast aggregation method.  ``"mean"`` (spec) or ``"median"``.
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
            weight_mode=weight_mode,
            weight_cap=weight_cap,
            price_mode=price_mode,
        )

        # Apply dampening: blend candidate towards current prices
        if alpha < 1.0:
            blended = alpha * result.market_prices + (1.0 - alpha) * current_prices
            # Replace iteration's market_prices with blended version for delta calc
            result = FuturesMarketIteration(
                iteration=result.iteration,
                market_prices=blended,
                strategy_profits=result.strategy_profits,
                strategy_weights=result.strategy_weights,
                active_strategies=result.active_strategies,
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
