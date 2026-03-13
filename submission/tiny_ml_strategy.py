"""Tiny ML baseline using a ridge-style linear classifier."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy
from submission.common import LinearDirectionModel


class TinyMLStrategy(ChallengeStrategy):
    """Fit a small linear model on lagged daily features and predict direction."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.model = LinearDirectionModel.fit(train_data, ridge_penalty=2.0)

    def act(self, state: ChallengeState) -> int | None:
        score = self.model.predict_score(state)
        if abs(score) < 0.05:
            return None
        return 1 if score > 0.0 else -1

    def reset(self) -> None:
        pass
