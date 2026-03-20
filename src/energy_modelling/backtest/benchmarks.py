"""Entry-price benchmark factories for robustness testing.

Each factory takes a daily DataFrame and returns a pd.Series indexed by
delivery_date that can be passed as ``entry_prices`` to
:func:`~energy_modelling.backtest.runner.run_backtest`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _settlement_series(daily_data: pd.DataFrame) -> pd.Series:
    """Extract lagged settlement series (yesterday's settlement)."""
    df = daily_data
    if "delivery_date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("delivery_date")
    if "last_settlement_price" in df.columns:
        series = df["last_settlement_price"].astype(float)
    else:
        series = df["settlement_price"].shift(1).astype(float)
    if hasattr(series.index, "date") and callable(series.index.date):
        series.index = series.index.date
    return series.dropna()


def yesterday_settlement(daily_data: pd.DataFrame) -> pd.Series:
    """Baseline: yesterday's settlement price (the existing default)."""
    return _settlement_series(daily_data)


def noisy_settlement(
    daily_data: pd.DataFrame, std_eur: float = 5.0, seed: int = 42
) -> pd.Series:
    """Yesterday's settlement + Gaussian noise at given standard deviation."""
    base = _settlement_series(daily_data)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std_eur, size=len(base))
    return (base + noise).rename("noisy_settlement")


def biased_settlement(
    daily_data: pd.DataFrame, bias_eur: float = 5.0
) -> pd.Series:
    """Yesterday's settlement + constant bias (EUR/MWh)."""
    base = _settlement_series(daily_data)
    return (base + bias_eur).rename("biased_settlement")


def perfect_foresight_price(daily_data: pd.DataFrame) -> pd.Series:
    """The real settlement price (oracle upper bound)."""
    df = daily_data
    if "delivery_date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("delivery_date")
    series = df["settlement_price"].astype(float)
    if hasattr(series.index, "date") and callable(series.index.date):
        series.index = series.index.date
    return series.rename("perfect_foresight")


ALL_BENCHMARKS: dict[str, dict] = {
    "baseline": {"factory": yesterday_settlement, "params": {}},
    "noise_1": {"factory": noisy_settlement, "params": {"std_eur": 1.0}},
    "noise_5": {"factory": noisy_settlement, "params": {"std_eur": 5.0}},
    "noise_10": {"factory": noisy_settlement, "params": {"std_eur": 10.0}},
    "noise_20": {"factory": noisy_settlement, "params": {"std_eur": 20.0}},
    "bias_plus_5": {"factory": biased_settlement, "params": {"bias_eur": 5.0}},
    "bias_minus_5": {"factory": biased_settlement, "params": {"bias_eur": -5.0}},
    "oracle": {"factory": perfect_foresight_price, "params": {}},
}


def get_benchmark(benchmark_id: str, daily_data: pd.DataFrame) -> pd.Series:
    """Generate entry prices for a named benchmark configuration."""
    if benchmark_id not in ALL_BENCHMARKS:
        msg = f"Unknown benchmark: {benchmark_id!r}. Available: {list(ALL_BENCHMARKS)}"
        raise ValueError(msg)
    spec = ALL_BENCHMARKS[benchmark_id]
    return spec["factory"](daily_data, **spec["params"])
