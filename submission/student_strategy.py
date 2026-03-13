"""Baseline student submission for the DE-LU futures hackathon.

This baseline follows the naive-copy idea: it always goes long.
Students only need to edit this file to create a valid submission.
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class StudentStrategy(ChallengeStrategy):
    """Naive-copy baseline that is easy to beat."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.training_rows = len(train_data)

    def act(self, state: ChallengeState) -> int | None:
        return 1

    def reset(self) -> None:
        pass
