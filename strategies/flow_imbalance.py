"""Flow imbalance strategy using combined cross-border net imports.

Uses the combined FR + NL net import flow to detect cross-border imbalances.
When Germany is importing heavily (high net imports), the market is tight and
prices are expected to mean-revert downward. When Germany is exporting heavily
(low net imports), the market is loose and prices are expected to rise.

Signal:
    net_flow > P75 (heavy import)  -> short (-1) [expect mean-reversion down]
    net_flow < P25 (heavy export)  -> long  (+1) [expect mean-reversion up]
    otherwise                      -> neutral (skip)

Source: Phase 10g candidate #9. Cross-border flow dynamics are
underrepresented (only 2 flow strategies vs 6 price-spread strategies).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_FR_FLOW = "flow_fr_net_import_mw_mean"
_NL_FLOW = "flow_nl_net_import_mw_mean"


class FlowImbalanceStrategy(BacktestStrategy):
    """Short when DE net imports are high (tight market, expect reversion);
    long when DE is exporting heavily (loose market, expect price recovery).
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._p25: float = 0.0
        self._p75: float = 0.0
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True

        fr = train_data[_FR_FLOW].values.astype(float)
        nl = train_data[_NL_FLOW].values.astype(float)
        combined = fr + nl

        clean = combined[np.isfinite(combined)]
        if len(clean) > 0:
            self._p25 = float(np.percentile(clean, 25))
            self._p75 = float(np.percentile(clean, 75))
        else:
            self._p25 = 0.0
            self._p75 = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

        self.skip_buffer = self._mean_abs_change * 0.3

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("FlowImbalanceStrategy.forecast() called before fit()")

        fr = float(state.features[_FR_FLOW])
        nl = float(state.features[_NL_FLOW])
        combined = fr + nl

        if combined > self._p75:
            direction = -1  # heavy imports -> expect price to fall
        elif combined < self._p25:
            direction = 1  # heavy exports -> expect price to rise
        else:
            return state.last_settlement_price  # neutral zone

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
