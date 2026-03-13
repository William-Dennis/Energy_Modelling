# Code Review: Energy_Modelling

This review treats the repository as both a quantitative research platform and a codebase that could evolve into a live trading system. The project is already unusually clean for a research repository: the domain model is explicit, the timing contract is documented, the test suite is broad, and the split between market simulation and strategy logic is thoughtful. The main concerns are not basic code hygiene; they are metric correctness, a few fragile architectural couplings, and gaps between a research-grade backtester and a production-grade trading evaluation framework.

## 1. Major Architectural Issues

- `submission/` is outside the installable package and is discovered via runtime import tricks rather than a supported extension mechanism. `src/energy_modelling/dashboard/challenge_submissions.py:23` inserts both the repo root and `src/` into `sys.path`, and `src/energy_modelling/dashboard/challenge_submissions.py:58` imports `submission` as an ad hoc package. This works locally, but it is brittle for packaging, CI, notebooks, and future deployment.
- The dashboard mutates interpreter state at import time. `src/energy_modelling/dashboard/challenge_submissions.py:25`-`src/energy_modelling/dashboard/challenge_submissions.py:28` changes `sys.path` during module import, which is a hidden global side effect and makes import behavior depend on launch location.
- The challenge layer depends on private symbols from the market simulation layer. `src/energy_modelling/challenge/data.py:11`-`src/energy_modelling/challenge/data.py:17` imports `_FORECAST_FEATURE_COLS` and `_REALISED_FEATURE_COLS` from `src/energy_modelling/market_simulation/data.py:68`-`src/energy_modelling/market_simulation/data.py:79`. The underscore prefix says "private", but another package relies on them as API. That is fragile coupling and makes refactors hazardous.
- There are two separate strategy abstractions with overlapping behavior. `src/energy_modelling/strategy/base.py:14`-`src/energy_modelling/strategy/base.py:46` defines `Strategy`, while `src/energy_modelling/challenge/types.py:36`-`src/energy_modelling/challenge/types.py:47` defines `ChallengeStrategy`. Both expose `act()` and `reset()`, but differ just enough to duplicate concepts and tooling.
- The project mixes "framework code" and "competition submission code" without a clean boundary. The dashboard imports organizer code, participant code, and data-generation code in one place: `src/energy_modelling/dashboard/challenge_submissions.py:30`-`src/energy_modelling/dashboard/challenge_submissions.py:34`. This raises the maintenance cost because UI concerns, evaluation concerns, and plugin discovery all evolve together.
- Metric definitions are duplicated across two modules with different semantics. `src/energy_modelling/strategy/analysis.py:18`-`src/energy_modelling/strategy/analysis.py:95` and `src/energy_modelling/challenge/scoring.py:12`-`src/energy_modelling/challenge/scoring.py:41` both compute Sharpe, win rate, drawdown, and summary stats, but not in a shared implementation. The result is metric drift rather than intentional variation.
- Backtest economics are hard-coded in multiple places instead of centralized in one contract abstraction. `src/energy_modelling/strategy/runner.py:28`-`src/energy_modelling/strategy/runner.py:29` fixes quantity and hours; `src/energy_modelling/challenge/runner.py:98`-`src/energy_modelling/challenge/runner.py:99` re-implements the PnL formula inline; `src/energy_modelling/market_simulation/contract.py:40`-`src/energy_modelling/market_simulation/contract.py:59` separately defines the canonical formula. This is correct today, but it invites silent divergence later.
- The challenge runner passes full historical rows, including realized labels, to user strategies by design. `src/energy_modelling/challenge/runner.py:93` gives `history=data.loc[data.index < delivery_date].copy()`, and `src/energy_modelling/challenge/types.py:25`-`src/energy_modelling/challenge/types.py:27` explicitly permits historical labels. This is not leakage for a daily directional task, but it is a sharp edge: future extensions could accidentally let strategies learn against organizer-only diagnostics rather than intended public features.
- The package dependency boundary is not cleanly separated between runtime and development tooling. `pyproject.toml:22` puts `ruff` in runtime dependencies, and `pyproject.toml:27`-`pyproject.toml:28` put notebook tooling in the main install set. This increases installation surface area and makes downstream environments heavier than necessary.
- The codebase is research-friendly, but there is no clear production boundary. There are no service interfaces for market data providers, no execution adapter, no portfolio/risk layer, and no transaction-cost model. That is acceptable for research, but it means the architecture should be treated as a backtesting platform, not a near-live trading stack.

## 2. Potential Quant / Backtest Errors

- `annualized_return_pct` is misnamed and economically misleading. `src/energy_modelling/strategy/analysis.py:78`-`src/energy_modelling/strategy/analysis.py:85` computes `total_pnl / years`, which is annualized PnL in EUR, not a percentage return. Labeling it as a percent can lead readers to compare strategies on a false return scale.
- The docstring for `annualized_return_pct` is internally inconsistent. `src/energy_modelling/strategy/analysis.py:33`-`src/energy_modelling/strategy/analysis.py:35` describes an annualized return percentage based on notional, but the implementation never computes a notional denominator. That is a reporting bug, not just a naming nit.
- Win rate is inconsistent between the strategy backtester and the challenge scorer. `src/energy_modelling/strategy/analysis.py:55` uses profitable days divided by all evaluated days, while `src/energy_modelling/challenge/scoring.py:27`-`src/energy_modelling/challenge/scoring.py:36` uses profitable active trades divided by `trade_count`. The same strategy can therefore display two different win rates depending on which UI the user looks at.
- Sharpe is diluted for strategies that skip days. `src/energy_modelling/challenge/scoring.py:23`-`src/energy_modelling/challenge/scoring.py:25` computes mean and standard deviation over all evaluation days, including `0.0` PnL days caused by `None` predictions in `src/energy_modelling/challenge/runner.py:99`. A sparse strategy can therefore appear smoother than it really is, especially if it trades only on a subset of days.
- The same skip-day Sharpe issue exists in rolling analysis for the main backtester if users later introduce explicit zero-PnL days rather than omitted days. `src/energy_modelling/strategy/analysis.py:137`-`src/energy_modelling/strategy/analysis.py:140` assumes the PnL series definition itself captures the intended exposure convention. That contract is currently implicit, not explicit.
- `max_drawdown_pct` is unstable when the running peak is small. `src/energy_modelling/strategy/analysis.py:74`-`src/energy_modelling/strategy/analysis.py:76` divides by peak cumulative PnL, which can make the percentage explode or become unintuitive when equity has barely risen above zero. That makes it a weak risk metric for low-edge or short samples.
- No transaction costs are modeled. `src/energy_modelling/market_simulation/contract.py:40`-`src/energy_modelling/market_simulation/contract.py:59` and `src/energy_modelling/challenge/runner.py:98`-`src/energy_modelling/challenge/runner.py:99` assume frictionless trading. For an always-in strategy such as `src/energy_modelling/strategy/naive_copy.py:25`-`src/energy_modelling/strategy/naive_copy.py:59`, this can materially overstate viability.
- No slippage or auction execution uncertainty is modeled. The framework assumes the trade is always established exactly at the prior settlement proxy exposed as `last_settlement_price` in `src/energy_modelling/strategy/runner.py:101` and `src/energy_modelling/challenge/runner.py:91`. That is fine as a simplified benchmark, but it is a silent optimistic assumption if results are interpreted as deployable.
- There is no explicit treatment of capital, leverage, or exposure normalization. PnL is always absolute EUR on a fixed 1 MW position in `src/energy_modelling/strategy/runner.py:28` and `src/energy_modelling/strategy/runner.py:102`. That keeps comparisons simple, but it also means Sharpe, drawdown percentage, and annualized return language can be misread as portfolio-level metrics.
- `target_direction` is built with `np.sign` in `src/energy_modelling/challenge/data.py:71`-`src/energy_modelling/challenge/data.py:74`, so flat days become `0`. That is reasonable for labels, but it creates a three-state target in a challenge API that only permits predictions `-1`, `1`, or `None` in `src/energy_modelling/challenge/runner.py:53`-`src/energy_modelling/challenge/runner.py:59`. This mismatch is subtle and can bias model training, especially for linear models fit directly to `target_direction`.
- The settlement calculation uses the mean of available hourly prices after dropping NaNs in `src/energy_modelling/market_simulation/contract.py:33`-`src/energy_modelling/market_simulation/contract.py:37`. That is robust to missing data, but if a day is incomplete the framework quietly changes the economic meaning of the contract from "24-hour average" to "average of whatever exists".
- The `PerfectForesightStrategy` is correctly marked as a theoretical upper bound in `src/energy_modelling/strategy/perfect_foresight.py:1`-`src/energy_modelling/strategy/perfect_foresight.py:23`, but this strategy should never share a leaderboard or visual framing with causal strategies without strong labeling. Otherwise it can normalize look-ahead logic in exploratory work.

## 3. Code Quality Issues

- The strongest aspect of the repository is readability. Docstrings are consistently strong, domain naming is clear, dataclasses are used well, and the data-flow from `MarketEnvironment` to `BacktestRunner` to metrics is easy to follow.
- The timing contract for features is one of the best-designed parts of the codebase. `src/energy_modelling/market_simulation/data.py:6`-`src/energy_modelling/market_simulation/data.py:13` and `src/energy_modelling/market_simulation/data.py:141`-`src/energy_modelling/market_simulation/data.py:176` make the anti-leakage rule explicit and auditable.
- Metric naming does not always match implementation. The clearest example is `annualized_return_pct` in `src/energy_modelling/strategy/analysis.py:80`, but it is a broader maintainability issue because a future contributor will trust the API surface before re-deriving the math.
- Similar business logic appears in several places with slight variations. Examples include the exclusion column sets in `src/energy_modelling/challenge/runner.py:13`-`src/energy_modelling/challenge/runner.py:21` and `submission/common.py:12`-`submission/common.py:20`, plus metric calculations split across `analysis.py` and `scoring.py`. This increases drift risk.
- The code relies on convention more than enforceable types at some boundaries. For example, `DayState.features` is a single-row `pd.DataFrame` in `src/energy_modelling/market_simulation/types.py:63`-`src/energy_modelling/market_simulation/types.py:74`, while `ChallengeState.features` is a `pd.Series` in `src/energy_modelling/challenge/types.py:22`-`src/energy_modelling/challenge/types.py:33`. That makes adapter code and strategy reuse harder than necessary.
- The dashboard module is too large and too stateful. `src/energy_modelling/dashboard/challenge_submissions.py` is more than 500 lines and handles discovery, data generation, evaluation, plotting, formatting, and UI event flow in one file. It is readable now, but it is already at the size where local changes will create accidental coupling.
- Runtime discovery suppresses all import exceptions in submission loading. `src/energy_modelling/dashboard/challenge_submissions.py:69`-`src/energy_modelling/dashboard/challenge_submissions.py:73` catches `Exception` and continues silently. That is user-friendly for a demo dashboard, but poor for debugging because broken submissions simply disappear.
- There is no single authoritative schema object for daily challenge data. Feature timing, label columns, excluded columns, and glossary rules are spread across `src/energy_modelling/challenge/data.py`, `src/energy_modelling/challenge/runner.py`, and `submission/common.py`. The logic is still understandable, but schema drift is likely over time.
- The dependency declaration is heavier than necessary for a library. `pyproject.toml:10`-`pyproject.toml:29` mixes runtime, notebook, lint, and plotting dependencies together. This is acceptable for a monorepo-style research project, but weakens packaging discipline.

## 4. Suggested Refactors

### Refactor 1: Create a public feature schema module

Problem: challenge code imports private feature lists from the market simulation package.

Recommendation: move feature group definitions into a small public schema module, then import that schema from both layers.

Example:

```python
# src/energy_modelling/schema.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DailyFeatureSchema:
    realised_feature_cols: tuple[str, ...]
    forecast_feature_cols: tuple[str, ...]
    price_stat_cols: tuple[str, ...] = ("price_mean", "price_max", "price_min", "price_std")


DAILY_FEATURE_SCHEMA = DailyFeatureSchema(
    realised_feature_cols=(
        "gen_solar_mw",
        "gen_wind_onshore_mw",
        "gen_wind_offshore_mw",
        # ...
    ),
    forecast_feature_cols=(
        "load_forecast_mw",
        "forecast_solar_mw",
        "forecast_wind_onshore_mw",
        "forecast_wind_offshore_mw",
    ),
)
```

Then use `DAILY_FEATURE_SCHEMA` from both `src/energy_modelling/market_simulation/data.py` and `src/energy_modelling/challenge/data.py` rather than importing underscore-prefixed names across package boundaries.

### Refactor 2: Centralize metric definitions and exposure conventions

Problem: the repository has two metric engines with inconsistent semantics.

Recommendation: define one metrics module with explicit knobs for `active_only`, `capital_base`, and `skip_day_policy`.

Example:

```python
from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class MetricPolicy:
    active_days_only: bool = False
    annualization_days: int = 252


def sharpe_ratio(pnl: pd.Series, policy: MetricPolicy) -> float:
    series = pnl[pnl != 0.0] if policy.active_days_only else pnl
    if len(series) < 2:
        return 0.0
    std = float(series.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(series.mean()) / std * math.sqrt(policy.annualization_days)
```

This makes differences between challenge scoring and research analysis explicit rather than accidental.

### Refactor 3: Introduce a shared PnL contract helper for challenge evaluation

Problem: the challenge runner re-implements contract economics inline.

Recommendation: route both paths through one helper so entry-price, quantity, and hours assumptions stay aligned.

Example:

```python
from energy_modelling.market_simulation.types import Trade
from energy_modelling.market_simulation.contract import compute_pnl


def challenge_pnl(last_settlement_price: float, settlement_price: float, prediction: int) -> float:
    trade = Trade(
        delivery_date=date.min,
        entry_price=last_settlement_price,
        position_mw=float(prediction),
        hours=24,
    )
    return compute_pnl(trade, settlement_price)
```

If object allocation feels too heavy, a dedicated pure function in `contract.py` is still better than duplicating the formula in runners.

### Refactor 4: Replace `sys.path` mutation with a plugin or package-based submission loader

Problem: dashboard imports depend on launch path and interpreter mutation.

Recommendation: either move `submission/` under `src/` as `energy_modelling.submission`, or define a plugin discovery mechanism with a configured directory.

Example:

```python
def discover_submission_modules(submission_dir: Path) -> list[str]:
    return [
        f"energy_modelling.submission.{path.stem}"
        for path in submission_dir.glob("*.py")
        if path.stem not in {"__init__", "common"}
    ]
```

This removes the need for `sys.path` edits in `src/energy_modelling/dashboard/challenge_submissions.py:25`-`src/energy_modelling/dashboard/challenge_submissions.py:28`.

### Refactor 5: Add an adapter layer to unify strategy interfaces

Problem: internal strategies and challenge strategies express the same concept in two nearly parallel APIs.

Recommendation: use one minimal protocol plus adapters for state shape differences.

Example:

```python
from __future__ import annotations

from typing import Protocol


class DirectionStrategy(Protocol):
    def reset(self) -> None: ...
    def act(self, state: object) -> int | None: ...


class ChallengeStrategyAdapter:
    def __init__(self, strategy: DirectionStrategy) -> None:
        self._strategy = strategy

    def reset(self) -> None:
        self._strategy.reset()

    def act(self, state: ChallengeState) -> int | None:
        return self._strategy.act(state)
```

Even if the code keeps separate public classes, an adapter reduces duplicate tooling and creates a clearer migration path.

### Refactor 6: Split the dashboard into smaller modules

Problem: `src/energy_modelling/dashboard/challenge_submissions.py` handles too many responsibilities.

Recommendation: split it into at least:

- `dashboard/discovery.py` for submission loading
- `dashboard/evaluation.py` for period evaluation
- `dashboard/formatting.py` for tables and derived frames
- `dashboard/challenge_submissions.py` for Streamlit UI composition only

This will make future UI changes much safer and allow unit tests for non-UI logic.

## 5. Missing Tests

- Add tests that verify `annualized_return_pct` is either renamed or truly converted into a return percentage. Right now the implementation and documentation disagree in `src/energy_modelling/strategy/analysis.py:33`-`src/energy_modelling/strategy/analysis.py:35` and `src/energy_modelling/strategy/analysis.py:78`-`src/energy_modelling/strategy/analysis.py:85`.
- Add explicit tests for win-rate semantics across both metric modules so the difference is intentional and documented, or so both implementations are harmonized.
- Add tests for Sharpe under sparse trading behavior in the challenge runner: all days skipped, one trade only, alternating active and skipped days, and identical active-day PnL.
- Add tests for near-zero volatility and zero-volatility metric cases to avoid `inf` or unstable values in `src/energy_modelling/strategy/analysis.py:65`-`src/energy_modelling/strategy/analysis.py:66` and rolling Sharpe in `src/energy_modelling/strategy/analysis.py:137`-`src/energy_modelling/strategy/analysis.py:140`.
- Add tests for flat price-change days where `target_direction == 0` in `src/energy_modelling/challenge/data.py:72`, especially for strategies or models trained directly on that target.
- Add tests for incomplete daily hourly data to define whether settlement should fail fast or average the available hours in `src/energy_modelling/market_simulation/contract.py:33`-`src/energy_modelling/market_simulation/contract.py:37`.
- Add DST and calendar-boundary tests around daily aggregation to verify there are no silent date-assignment issues when the raw dataset contains irregular hour counts.
- Add tests for `submission/common.py:55`-`submission/common.py:94`, especially `LinearDirectionModel.fit()` and `predict_score()`, including missing values, constant columns, and deterministic predictions.
- Add tests for submission discovery failure paths in `src/energy_modelling/dashboard/challenge_submissions.py:69`-`src/energy_modelling/dashboard/challenge_submissions.py:73` so broken modules are surfaced intentionally rather than silently dropped.
- Add tests for the feature glossary classification in `src/energy_modelling/challenge/data.py:98`-`src/energy_modelling/challenge/data.py:127` to guard against schema drift if feature lists change.
- Add tests for hidden-label stripping and state feature exclusion together so organizer-only columns can never leak into student-visible decision features.
- Add UI-adjacent tests for the dashboard's non-visual helper functions such as `_leaderboard_frame`, `_period_summary_frame`, `_comparison_curve_frame`, and `_format_leaderboard` in `src/energy_modelling/dashboard/challenge_submissions.py:139`-`src/energy_modelling/dashboard/challenge_submissions.py:255`.

## 6. Risk Assessment

Overall rating: HIGH for live-trading interpretation, MEDIUM for research use.

Why this is not LOW risk:

- The repository has several silent reporting issues rather than loud failures. Misnamed annualized return, inconsistent win-rate definitions, and skip-day Sharpe dilution will not crash a run, but they can change strategy rankings and lead to overconfidence.
- The core anti-leakage design looks sound. The most important quantitative safeguard, the lagging of realized features in `src/energy_modelling/market_simulation/data.py:175`, is implemented correctly and supported by the overall state construction flow. That materially reduces classic look-ahead risk.
- The economic model is intentionally simplified. Fixed 1 MW sizing, frictionless execution, no transaction costs, no slippage, and no portfolio capital model are fine for educational benchmarking, but too optimistic for operational decision-making.
- The architecture is good enough for iterative research. Clean types, strong tests, and explicit strategy runners make it a solid platform for experimentation and hackathon usage.
- The architecture is not yet ready to support production accountability. There is no hardened data-quality layer, no explicit schema contract across modules, no audit trail for model versions, and no separation between research metrics and deployment metrics.

Recommended interpretation:

- Safe to use for research, teaching, and leaderboard-style offline evaluation after metric cleanup.
- Not safe to use as evidence of deployable trading profitability until metric semantics, cost modeling, and package boundaries are tightened.
- If the goal is a credible research platform, fix the metric inconsistencies first.
- If the goal is a path toward live trading, fix the metric inconsistencies first, then add a cost model, formal schema module, and stricter packaging/plugin boundaries before extending strategy complexity.
