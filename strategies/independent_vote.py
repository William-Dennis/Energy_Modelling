"""Independent vote strategy.

Combines signals from the most independent feature sources (identified
in Phase 10f) into a simple majority vote.  Uses 5 orthogonal signals:
1. Price z-score (mean reversion)
2. Gas trend (fuel cost momentum)
3. Net demand (supply-demand balance)
4. Is weekend (calendar effect)
5. Renewable penetration (supply mix)

Each signal independently votes long/short, and the majority wins.

Signal:
    majority(5 votes) -> direction
    tie (should not happen with 5 voters) -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class IndependentVoteStrategy(BacktestStrategy):
    """Majority vote from 5 independent feature signals.

    Combines price z-score, gas trend, net demand, weekend effect,
    and renewable penetration into a single directional vote.
    """

    def __init__(self) -> None:
        self._median_zscore: float = 0.0
        self._median_net_demand: float = 0.0
        self._median_renew: float = 30.0
        self._mean_abs_change: float = 1.0
        self._fitted: bool = False

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True

        if "price_zscore_20d" in train_data.columns:
            self._median_zscore = float(train_data["price_zscore_20d"].median())
        if "net_demand_mw" in train_data.columns:
            self._median_net_demand = float(train_data["net_demand_mw"].median())
        if "renewable_penetration_pct" in train_data.columns:
            self._median_renew = float(train_data["renewable_penetration_pct"].median())

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.3

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("IndependentVoteStrategy.forecast() called before fit()")

        votes = 0  # positive = long, negative = short

        # 1. Price z-score: high -> overbought -> short; low -> long
        zscore = float(state.features.get("price_zscore_20d", 0.0))
        votes += -1 if zscore > self._median_zscore else 1

        # 2. Gas trend: rising gas -> electricity up -> long
        gas_trend = float(state.features.get("gas_trend_3d", 0.0))
        votes += 1 if gas_trend > 0 else -1

        # 3. Net demand: high -> scarcity -> long
        net_demand = float(state.features.get("net_demand_mw", 0.0))
        votes += 1 if net_demand > self._median_net_demand else -1

        # 4. Weekend: weekend -> lower demand -> short
        is_weekend = bool(state.features.get("is_weekend", 0))
        votes += -1 if is_weekend else 1

        # 5. Renewable penetration: high -> surplus -> short
        renew = float(state.features.get("renewable_penetration_pct", 0.0))
        votes += -1 if renew > self._median_renew else 1

        if votes == 0:
            return state.last_settlement_price  # tie -> neutral

        direction = 1 if votes > 0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
