"""No-trade baseline for the challenge dashboard."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class SkipAllStrategy(ChallengeStrategy):
    """Never trades. This provides a zero-risk benchmark."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.training_rows = len(train_data)

    def act(self, state: ChallengeState) -> int | None:
        return None

    def reset(self) -> None:
        pass
