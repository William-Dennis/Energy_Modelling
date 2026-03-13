"""Price-level mean-reversion baseline."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class PriceLevelMeanReversionStrategy(ChallengeStrategy):
    """Fade unusually high or low prior settlements relative to history."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.anchor = float(train_data["last_settlement_price"].median())

    def act(self, state: ChallengeState) -> int | None:
        return -1 if state.last_settlement_price >= self.anchor else 1

    def reset(self) -> None:
        pass
