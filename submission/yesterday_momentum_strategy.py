"""Momentum baseline using yesterday's realised move."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class YesterdayMomentumStrategy(ChallengeStrategy):
    """Go with yesterday's realised price change direction."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.training_rows = len(train_data)

    def act(self, state: ChallengeState) -> int | None:
        if state.history.empty:
            return 1
        previous_change = float(state.history.iloc[-1]["price_change_eur_mwh"])
        return 1 if previous_change >= 0.0 else -1

    def reset(self) -> None:
        pass
