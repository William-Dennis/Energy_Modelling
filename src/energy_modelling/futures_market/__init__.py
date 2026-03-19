"""Market simulation sub-package -- shared data utilities.

Provides data loading, feature engineering, settlement price computation,
and PnL calculations for the DE-LU day-ahead power market.
"""

from energy_modelling.futures_market.contract import compute_pnl, compute_settlement_price
from energy_modelling.futures_market.data import (
    build_daily_features,
    compute_daily_settlement,
    load_dataset,
)

__all__ = [
    "build_daily_features",
    "compute_daily_settlement",
    "compute_pnl",
    "compute_settlement_price",
    "load_dataset",
]
