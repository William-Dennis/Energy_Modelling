"""Always-long baseline for the challenge dashboard."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class AlwaysLongStrategy(ChallengeStrategy):
    """Always go long. Useful as a naive baseline (bet price goes up)."""

    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def act(self, state: ChallengeState) -> int | None:
        return 1

    def reset(self) -> None:
        pass
