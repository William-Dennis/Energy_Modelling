# Phase 4: Implement Strategies

## Status: NOT STARTED

## Objective

Implement new `ChallengeStrategy` classes in the `strategies/` directory based on
the hypotheses identified in Phase 3. Follow TDD: write tests first, then implement.

## Prerequisites

- Phase 0 complete (consolidated framework, `strategies/` working)
- Phase 3 complete (hypotheses ranked and specified)

## Checklist

### 4a. Implement PerfectForesightStrategy (reference)
- [ ] Create `strategies/perfect_foresight.py` — ChallengeStrategy that looks at `settlement_price` from history
- [ ] Write tests in `tests/challenge/test_perfect_foresight_challenge.py`
- [ ] Verify: always non-negative PnL, dominates naive strategies

### 4b. Implement SkipAllStrategy (baseline)
- [ ] Create `strategies/skip_all.py` — always returns None
- [ ] Write test: zero PnL, zero trades

### 4c. Implement hypothesis-driven strategies
*(List to be populated from Phase 3 output)*

- [ ] Strategy 1: [name] — `strategies/[name].py`
- [ ] Strategy 2: [name] — `strategies/[name].py`
- [ ] Strategy 3: [name] — `strategies/[name].py`
- [ ] ...

### 4d. Update `strategies/__init__.py`
- [ ] Add imports for all new strategies
- [ ] Verify dashboard discovery picks them all up

### 4e. Testing
- [ ] Each strategy has a dedicated test file
- [ ] Tests cover: correct interface implementation, expected direction for known inputs, edge cases
- [ ] All tests green

## Strategy Implementation Template

```python
"""[Description] strategy for the challenge dashboard."""

from __future__ import annotations

import pandas as pd

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy


class [Name]Strategy(ChallengeStrategy):
    """[Docstring explaining the hypothesis and signal logic]."""

    def fit(self, train_data: pd.DataFrame) -> None:
        # Optional: compute parameters from training data
        pass

    def act(self, state: ChallengeState) -> int | None:
        # Return +1 (long), -1 (short), or None (skip)
        pass

    def reset(self) -> None:
        pass
```

## ChallengeStrategy Interface Reference

```python
class ChallengeStrategy(ABC):
    def fit(self, train_data: pd.DataFrame) -> None: ...   # optional training
    def act(self, state: ChallengeState) -> int | None: ... # abstract
    def reset(self) -> None: ...                             # reset between runs
```

```python
@dataclass(frozen=True)
class ChallengeState:
    delivery_date: date
    last_settlement_price: float
    features: pd.Series           # current-day features (no labels)
    history: pd.DataFrame         # all prior rows (with labels)
```

## PnL Formula

```
PnL = direction * (settlement_price - last_settlement_price) * 24
```

Where `direction` is `+1` (long) or `-1` (short). `None` means skip (PnL = 0).

## Strategies Implemented

*(To be filled in during execution)*

| Strategy | File | Hypothesis | Tests | Status |
|----------|------|-----------|-------|--------|
| PerfectForesight | `strategies/perfect_foresight.py` | Oracle | | NOT STARTED |
| SkipAll | `strategies/skip_all.py` | Baseline | | NOT STARTED |
