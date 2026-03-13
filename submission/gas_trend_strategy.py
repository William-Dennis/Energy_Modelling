"""Commodity-trend baseline using gas price history."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class GasTrendStrategy(ChallengeStrategy):
    """Go long after gas rises and short after gas falls."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.training_rows = len(train_data)

    def act(self, state: ChallengeState) -> int | None:
        if len(state.history) < 2:
            return 1
        latest = float(state.history.iloc[-1]["gas_price_usd_mean"])
        previous = float(state.history.iloc[-2]["gas_price_usd_mean"])
        return 1 if latest >= previous else -1

    def reset(self) -> None:
        pass
