"""Day-of-week calendar strategy for the challenge dashboard.

Exploits the strongest pattern found in EDA (Phase 3, H1): Monday settlement
prices are almost always higher than Sunday's (90.7% up rate across all years),
while Friday→Saturday and Saturday→Sunday transitions are reliably negative.

Signal:
    Monday    → long  (+1)   ~90.7% up
    Tuesday   → long  (+1)   ~59.6% up
    Wednesday → skip  (None) ~50/50
    Thursday  → skip  (None) ~50/50
    Friday    → short (-1)   ~61.3% down
    Saturday  → short (-1)   ~85.6% down
    Sunday    → short (-1)   ~73.5% down
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

# isoweekday(): Mon=1, Tue=2, Wed=3, Thu=4, Fri=5, Sat=6, Sun=7
_DAY_SIGNAL: dict[int, int | None] = {
    1: 1,  # Monday  → long
    2: 1,  # Tuesday → long
    3: None,  # Wednesday → skip
    4: None,  # Thursday  → skip
    5: -1,  # Friday   → short
    6: -1,  # Saturday → short
    7: -1,  # Sunday   → short
}


class DayOfWeekStrategy(BacktestStrategy):
    """Go long Mon/Tue, short Fri/Sat/Sun, skip Wed/Thu.

    Based on the structural weekly pattern in European electricity
    settlement prices: weekend prices are systematically lower due to
    reduced industrial demand.
    """

    def __init__(self) -> None:
        self._mean_change_by_day: dict[int, float] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        if "price_change_eur_mwh" in train_data.columns and "delivery_date" in train_data.columns:
            df = train_data.copy()
            df["_dow"] = pd.to_datetime(df["delivery_date"]).dt.weekday + 1  # Mon=1..Sun=7
            self._mean_change_by_day = df.groupby("_dow")["price_change_eur_mwh"].mean().to_dict()

    def act(self, state: BacktestState) -> int | None:
        return _DAY_SIGNAL[state.delivery_date.isoweekday()]

    def forecast(self, state: BacktestState) -> float:
        dow = state.delivery_date.isoweekday()
        mean_change = self._mean_change_by_day.get(dow, 0.0)
        return state.last_settlement_price + mean_change

    def reset(self) -> None:
        pass
