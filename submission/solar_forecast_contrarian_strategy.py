"""Solar-forecast contrarian baseline."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy
from submission.common import state_value


class SolarForecastContrarianStrategy(ChallengeStrategy):
    """Short when solar forecast is high relative to history."""

    def fit(self, train_data: pd.DataFrame) -> None:
        self.threshold = float(train_data["forecast_solar_mw_mean"].median())

    def act(self, state: ChallengeState) -> int | None:
        solar_forecast = state_value(state, "forecast_solar_mw_mean", self.threshold)
        return -1 if solar_forecast >= self.threshold else 1

    def reset(self) -> None:
        pass
