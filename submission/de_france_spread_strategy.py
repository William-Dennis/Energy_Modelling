"""Cross-border spread baseline using France vs DE-LU."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy
from submission.common import state_value


class DEFranceSpreadStrategy(ChallengeStrategy):
    """Go long when France's prior-day price looks rich versus DE-LU."""

    def fit(self, train_data: pd.DataFrame) -> None:
        spread = train_data["price_fr_eur_mwh_mean"] - train_data["last_settlement_price"]
        self.threshold = float(spread.median())

    def act(self, state: ChallengeState) -> int | None:
        france_price = state_value(state, "price_fr_eur_mwh_mean", state.last_settlement_price)
        spread = france_price - state.last_settlement_price
        return 1 if spread >= self.threshold else -1

    def reset(self) -> None:
        pass
