# Agents Guide — Energy Modelling Platform

Lessons learned and operational notes for AI agents working on this codebase.
Written 2026-03-21 after a full 20-item audit and cleanup session.

---

## 1. Environment & Tooling

### Use `uv`, not system Python

- **Always** use `uv run ...` for running scripts, tests, and linting.
- The system Python is missing `pytest-mock` and may have incompatible
  NumPy/SciPy versions. The `uv`-managed venv is the single source of truth.
- Install dev deps: `uv sync --group dev`
- Run tests: `uv run pytest -q`
- Run linter: `uv run ruff check .`
- Auto-fix lint: `uv run ruff check --fix .`

### Key commands

| Task | Command |
|------|---------|
| Full test suite | `uv run pytest -q` |
| Theorem verification | `uv run python scripts/verify_theorems.py` |
| Ruff lint | `uv run ruff check .` |
| Ruff auto-fix | `uv run ruff check --fix .` |
| Type check | `uv run ty check src/` |
| Collect data | `uv run collect-data` |
| Build backtest data | `uv run build-backtest-data` |
| Run dashboard | `uv run streamlit run src/energy_modelling/dashboard/app.py` |
| Recompute all results | `uv run recompute-all` |

### Test count baseline

- As of 2026-03-22: **1279 tests pass** in ~43s.
- If you see "11 collection errors", you're using system Python instead of `uv`.

---

## 2. Repository Layout

```
src/energy_modelling/
  backtest/          # Core framework: 16 modules, the heart of the project
  dashboard/         # Streamlit app: ~20 modules, 4 tabs
  data_collection/   # ENTSO-E + weather + commodity data: 14 modules
  futures_market/    # Shared data utilities: 4 modules

strategies/           # 100 registered strategies + ml_base + ensemble_base
tests/                # Mirrors src/ structure
scripts/              # Standalone experiment scripts (phase8/9/10, verify, etc.)
docs/                 # phases/, expansion/, eda/, specs
data/                 # raw/, processed/, backtest/, results/
issues/               # GitHub-style issue tracking docs
```

### Important files

| File | What it does |
|------|-------------|
| `src/.../backtest/types.py` | `BacktestStrategy` ABC, `BacktestState`, `STATE_EXCLUDE_COLUMNS` |
| `src/.../backtest/runner.py` | `run_backtest()` — the main backtest loop |
| `src/.../backtest/futures_market_engine.py` | Synthetic futures market: weights, prices, convergence loop |
| `src/.../backtest/futures_market_runner.py` | Orchestrator: runs all strategies then feeds into market engine |
| `src/.../backtest/convergence.py` | Convergence analysis tools |
| `strategies/__init__.py` | Registry: imports all 74 strategies |
| `scripts/verify_theorems.py` | Validates 4 mathematical theorems about the market engine |
| `docs/phases/ROADMAP.md` | Master roadmap — always update this when completing phases |

---

## 3. Architecture Understanding

### The Market Engine

The synthetic futures market (`futures_market_engine.py`) implements:

1. **Trading decision**: `q_{i,t} = sign(forecast_{i,t} - P^m_t)`
2. **Profit**: `r_{i,t} = q_{i,t} * (P_real_t - P^m_t)`
3. **Selection**: `w_i = max(Pi_i, 0) / sum(max(Pi_j, 0))`
4. **Price update**: `P^m_{t}^(k+1) = ema_alpha * weighted_avg + (1 - ema_alpha) * P_k`
5. **Iteration**: repeat until convergence delta < threshold

Key parameters:
- `ema_alpha=0.003` (production default as of Phase 14; engine function default is 0.1 for backwards compat; `1.0` = undampened spec)
- `convergence_threshold=0.01`
- `max_iterations=10_000`

### Strategy Interface

Every strategy implements `BacktestStrategy` (in `types.py`):
- `fit(train_data)` — optional, train on historical data
- `forecast(state) -> float` — **required**, return price forecast
- `act(state) -> int|None` — auto-derived from forecast via `skip_buffer`

### Data Flow

```
raw data → data_collection → processed/ → backtest/data.py → runner.py
  → per-strategy predictions + forecasts
  → futures_market_engine.py (iterative convergence)
  → futures_market_runner.py (orchestrates everything)
  → results saved to data/results/*.pkl
```

### Saved Artifacts

- `data/results/market_2024.pkl` — `FuturesMarketResult` for 2024 (100 strategies, 366 dates, converged with ema_alpha=0.003, ~1195 iters)
- `data/results/market_2025.pkl` — `FuturesMarketResult` for 2025 (100 strategies, 365 dates, converged with ema_alpha=0.003, ~1103 iters)
- `data/results/backtest_val_2024.pkl` / `backtest_hid_2025.pkl` — standalone backtest results
- `data/results/phase8/` — forecast pickles from Phase 8 experiments
- `data/results/phase10/results.csv` — EMA alpha sweep results

---

## 4. Documentation Conventions

### Phase docs

- Live in `docs/phases/phase_N_name.md` with optional sub-dirs for sub-phases.
- Each doc has: Status, Objective, Checklist, and usually a Working Definitions section.
- **Historical clarity principle**: Never delete stale info. Add WARNING blocks
  with `> **WARNING**` markdown syntax at the top of docs that are historically
  accurate but no longer operationally current.
- Always update `docs/phases/ROADMAP.md` when completing a phase.

### Phase numbering (current)

| Phase | Topic |
|-------|-------|
| 0-6 | Foundation: consolidation, EDA, hypotheses, strategies, backtest, feedback |
| 7 | Convergence analysis (undampened model, `ema_alpha=1.0`) |
| 8 | Oscillation research (historical record; winner never implemented) |
| 9 | EMA price update experiments (`ema_alpha=0.1` adopted as initial default) |
| 10 | Market behaviour & strategy robustness (complete) |
| 11 | New strategies & hyperparameter tuning (complete, 74 strategies) |
| 12 | Forecast cache & strategy expansion (complete, 100 strategies, 1279 tests) |
| 13 | ema_alpha production fix (complete, `ema_alpha=0.01` applied in recompute.py) |
| 14 | Engine default update (complete, `ema_alpha=0.003`, `max_iterations=10_000` as engine defaults; both years converge) |
| A-G | Expansion phases (parallel track, all complete, 67 strategies) |

### Expansion docs

- Live in `docs/expansion/phase_X_name.md`
- Each has a breadcrumb link back to ROADMAP and expansion README.
- `docs/expansion/strategy_registry.md` — the canonical 100-strategy list.
- `docs/expansion/signal_registry.md` — the signal catalog.

---

## 5. Common Pitfalls

### 1. `ema_alpha` matters everywhere

- Theorems in `verify_theorems.py` require `ema_alpha=1.0` (undampened).
- Production uses `ema_alpha=0.003` (applied in `recompute.py` as of Phase 13; engine defaults updated in Phase 14). Engine function defaults are 0.1 for backwards compat. Don't confuse the two.
- If you add a new `run_futures_market()` call for theorem testing, always pass `ema_alpha=1.0`.

### 2. Phase 8's `running_avg_k` was never implemented

- Phase 8 docs discuss `running_avg_k=5` as the winner. This parameter does not exist in the codebase.
- The actual dampening mechanism is `ema_alpha` (Phase 9). Don't go looking for `running_avg_k`.

### 3. `strategy_forecasts` is `{name: {date: float}}`

- Not a DataFrame. It's a nested dict.
- Dates are `datetime.date` objects (or pandas Timestamps depending on context).
- 67 strategies x ~365 dates per year (74 after Phase 11).

### 4. Import order matters for scripts

- Scripts that manipulate `sys.path` need `# noqa: E402` on subsequent imports.
- After any `# noqa: E402` change, ruff may flag I001 (import sort). Run `--fix` to auto-sort.

### 5. Frozen dataclasses

- `FuturesMarketResult`, `FuturesMarketIteration`, `FuturesMarketEquilibrium`,
  `BacktestState`, `BacktestResult` are all `@dataclass(frozen=True)`.
- Don't try to mutate them. Create new instances.

### 6. `STATE_EXCLUDE_COLUMNS` lives in `types.py`

- Canonical definition: `src/energy_modelling/backtest/types.py`
- Both `runner.py` and `futures_market_runner.py` have backwards-compat aliases `_STATE_EXCLUDE_COLUMNS`.

### 7. 2024 vs 2025 are fundamentally different

- 2024: 366 dates, market does NOT converge after 500 iterations (delta ~1.77)
- 2025: 365 dates, market converges at iteration 327 but via active-strategy collapse to 0 strategies
- Any analysis script must handle both cases.

---

## 6. Testing Patterns

### Running tests

```bash
uv run pytest -q                    # all tests
uv run pytest tests/backtest/ -q    # just backtest tests
uv run pytest -k "convergence" -q   # by keyword
```

### Test structure

- Tests mirror `src/` structure: `tests/backtest/test_*.py`, `tests/data_collection/test_*.py`, etc.
- Strategy tests follow pattern: `tests/backtest/test_strategy_*.py`
- Phase-group strategy tests: `test_strategy_phase_c.py`, `test_strategy_phase_d.py`, etc.
- Convergence warnings from scikit-learn are suppressed in `pyproject.toml`.

### Writing new tests

- Use `pytest-mock` for mocking (it's a dev dependency).
- Use small synthetic datasets, not real data files.
- Strategies should be testable with ~10-20 rows of data.

---

## 7. Ruff Configuration

From `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

Active rule groups:
- **E**: pycodestyle errors (E402 module-level import, E501 line length)
- **F**: pyflakes (F401 unused imports, F541 f-strings)
- **I**: isort (I001 import sorting)
- **UP**: pyupgrade
- **B**: bugbear (B905 zip strict)
- **SIM**: simplify (SIM102 nested if, SIM108 ternary, SIM113 enumerate)

---

## 8. Script Conventions

### Experiment scripts (`scripts/phase*_*.py`)

- Each script is standalone, manipulates `sys.path` to import from `src/`.
- Results go to `data/results/phaseN/`.
- Use `argparse` for CLI args (typically `--year`, `--max-iters`).
- Load forecasts from pickles in `data/results/phase8/` (the canonical forecast set).

### Adding a new analysis script

1. Create `scripts/phase10X_name.py`
2. Add `sys.path` setup at top (copy pattern from existing scripts)
3. Add `# noqa: E402` on imports after `sys.path` manipulation
4. Save results to `data/results/phase10/`
5. Use `logging` not `print` for status messages
6. Use `tqdm` for progress bars on long loops

---

## 9. Working with Market Artifacts

### Loading pickles

```python
import pickle
from pathlib import Path

with open(Path("data/results/market_2024.pkl"), "rb") as f:
    result = pickle.load(f)  # FuturesMarketResult

eq = result.equilibrium  # FuturesMarketEquilibrium
eq.converged             # bool
eq.convergence_delta     # float
eq.iterations            # list[FuturesMarketIteration]
eq.final_market_prices   # pd.Series
eq.final_weights         # dict[str, float]

# Per-iteration data
for it in eq.iterations:
    it.iteration            # int
    it.market_prices        # pd.Series
    it.strategy_profits     # dict[str, float]
    it.strategy_weights     # dict[str, float]
    it.active_strategies    # list[str]

# Forecasts
result.strategy_forecasts   # dict[str, dict[date, float]]
```

### Key metrics to extract per iteration

- **convergence_delta**: `max|P_{k} - P_{k-1}|`
- **MAE**: `mean|P_market - P_real|`
- **RMSE**: `sqrt(mean((P_market - P_real)^2))`
- **bias**: `mean(P_market - P_real)`
- **active_count**: `len(it.active_strategies)`
- **weight_entropy**: `-sum(w * log(w))` for `w > 0`
- **top1_weight**: `max(weights.values())`
- **top5_concentration**: sum of top 5 weights

---

## 10. Commit & Documentation Discipline

### Before committing

1. `uv run ruff check .` — must be clean
2. `uv run pytest -q` — must pass (1279+ tests)
3. `uv run python scripts/verify_theorems.py` — ALL THEOREMS VERIFIED

### After completing a phase or sub-phase

1. Update the phase doc: mark checklist items `[x]`, set Status to COMPLETE
2. Update `docs/phases/ROADMAP.md`: status column, changelog entry
3. Update `issues/ROADMAP.md` if relevant issues changed

### Git discipline

- Frequent, descriptive commits tied to specific phase goals.
- Never force-push to main.
- Use conventional-ish messages: "Phase 10b: add behaviour inventory script"

---

## 11. Phase 10 Context (COMPLETE), Phase 11 (COMPLETE), and Phase 12 (COMPLETE)

Phase 10 is complete. All 8 sub-phases (10a-10h) are finished. Phase 11 is
complete. 7 new strategies added, bringing the total from 67 to 74. Phase 12
is complete. SQLite forecast cache + 26 new strategies, bringing the total to 100.

### Phase 10 Summary
- **10a**: Baseline reconciliation -- canonical config frozen
- **10b**: Behaviour inventory -- per-iteration metrics, behaviour classifications
- **10c**: Mechanism attribution -- 56-run ablation suite, key causal findings
- **10d**: Regime and cluster analysis -- ML cluster dominance, regime stability
- **10e**: Sentinel case studies -- 5 iteration-level causal traces
- **10f**: Strategy robustness -- LOO analysis, robustness classifications
- **10g**: Stronger strategy design -- 5 design rules, 10 candidates, acceptance criteria
- **10h**: Synthesis -- unified causal explanation, engine/dashboard recommendations

### Key Phase 10 Findings
- Positive-profit truncation is the root cause of active-strategy collapse
- ML regression cluster (11 strategies) captures >90% weight; 49 strategies are dead weight
- `ema_alpha=0.01` is the only alpha achieving healthy convergence for both years
- 2024 oscillates (non-convergent), 2025 collapses to absorbing zero-active state
- Engine improvements (active-strategy floor, early stopping) should precede new strategies

### Phase 11 Summary
- 7 new strategies: SpreadMomentum, SelectiveHighConviction, TemperatureCurve,
  NuclearEvent, FlowImbalance, RegimeRidge, PrunedMLEnsemble
- 44 new tests (1089 total)
- Engine code frozen; hyperparameter recommendations documented:
  `ema_alpha=0.01`, `max_iterations=200`
- Strategy registry updated to 74 strategies

### Phase 12 Summary
- **12A**: SQLite forecast cache with per-strategy fingerprinting
  - Warm cache recompute-all: 2.6 seconds (down from ~8-10 minutes)
  - 19 new tests
- **12B**: 26 new strategies in 6 batches, reaching 100 total
  - Batch 1: RadiationSolar, IntradayRange, OffshoreWindAnomaly, ForecastPriceError, PolandSpread
  - Batch 2: DenmarkSpread, CzechAustrianMean, SparkSpread, CarbonGasRatio, WeeklyAutocorrelation
  - Batch 3: MonthlyMeanReversion, LoadGenerationGap, RenewableRamp, NuclearGasSubstitution, VolatilityBreakout
  - Batch 4: SeasonalRegimeSwitch, WeekendMeanReversion, HighVolSkip, RadiationRegime, IndependentVote
  - Batch 5: MedianIndependent, SpreadConsensus, SupplyDemandBalance, ContrarianMomentum, ConvictionWeighted
  - Batch 6: BalancedLongShort (#100)
  - 171 new tests (1279 total)

### Next Phase (13 -- not yet planned)
Potential scope: engine convergence improvements (active-strategy floor, early
stopping), dashboard convergence reporting, full market re-simulation with
100 strategies and recommended hyperparameters (`ema_alpha=0.01`,
`max_iterations=200`).

---

## 12. Performance Notes

- The full test suite runs in ~43s.
- `verify_theorems.py` runs in ~2-3s.
- `run_full_backtest.py` uses `ProcessPoolExecutor` and can take several minutes.
- Market simulation with 100 strategies and 500 iterations takes ~0.1s per year (cached).
- Cold cache recompute-all with 100 strategies takes ~595 seconds.
- Warm cache recompute-all with 100 strategies takes ~2.6 seconds.
- The vectorised path in `futures_market_engine.py` (`_vec_iteration`) is much
  faster than the reference path (`run_futures_market_iteration`).

---

## 13. Things NOT to Do

1. **Don't delete historical docs** — add WARNING blocks instead.
2. **Don't use system Python** — always `uv run`.
3. **Don't assume convergence** — 2024 doesn't converge. Handle both cases.
4. **Don't confuse `running_avg_k` with `ema_alpha`** — the former doesn't exist.
5. **Don't modify frozen dataclasses** — create new instances.
6. **Don't skip ruff/tests before committing** — the CI will catch you.
7. **Don't create new files when editing existing ones would work**.
8. **Don't reference Phase 9 as "Market Behaviour"** — that's Phase 10. Phase 9 is EMA experiments.
