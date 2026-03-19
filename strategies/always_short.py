"""Always-short baseline for the challenge dashboard."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class AlwaysShortStrategy(ChallengeStrategy):
    """Always short. Useful as a symmetry check against naive copy."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.training_rows = len(train_data)

    def act(self, state: ChallengeState) -> int | None:
        return -1

    def reset(self) -> None:
        pass
