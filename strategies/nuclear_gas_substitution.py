"""Nuclear-gas substitution strategy.

When nuclear output drops, gas plants must fill the gap, raising
marginal costs and electricity prices.  When nuclear is high, gas
plants are pushed out of the merit order, lowering prices.

Combines nuclear generation level with gas price to detect substitution
effects.

Signal:
    nuclear < median AND gas > median -> long  (+1) [expensive gap-fill]
    nuclear >= median AND gas <= median -> short (-1) [cheap baseload]
    otherwise -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_NUCLEAR_COL = "gen_nuclear_mw_mean"
_GAS_COL = "gas_price_usd_mean"


class NuclearGasSubstitutionStrategy(BacktestStrategy):
    """Trade based on nuclear-gas substitution dynamics.

    Low nuclear + high gas -> expensive replacement -> prices rise.
    High nuclear + low gas -> cheap baseload dominates -> prices fall.
    """

    def __init__(self) -> None:
        self._median_nuclear: float | None = None
        self._median_gas: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _NUCLEAR_COL in train_data.columns:
            self._median_nuclear = float(train_data[_NUCLEAR_COL].median())
        else:
            self._median_nuclear = 0.0

        if _GAS_COL in train_data.columns:
            self._median_gas = float(train_data[_GAS_COL].median())
        else:
            self._median_gas = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_nuclear is None or self._median_gas is None:
            raise RuntimeError("NuclearGasSubstitutionStrategy.forecast() called before fit()")

        nuclear = float(state.features.get(_NUCLEAR_COL, 0.0))
        gas = float(state.features.get(_GAS_COL, 0.0))

        low_nuclear = nuclear < self._median_nuclear
        high_gas = gas > self._median_gas

        if low_nuclear and high_gas:
            direction = 1  # expensive substitution -> prices rise
        elif not low_nuclear and not high_gas:
            direction = -1  # cheap baseload -> prices fall
        else:
            return state.last_settlement_price  # neutral

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
