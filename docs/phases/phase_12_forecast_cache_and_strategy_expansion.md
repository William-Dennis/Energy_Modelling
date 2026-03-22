# Phase 12: Forecast Cache & Strategy Expansion

**Status**: IN PROGRESS (12A complete, 12B pending)

**Depends On**: Phase 11

**Objective**: (A) Implement a SQLite-based forecast cache so that `recompute-all`
runs in <20 seconds when the cache is warm. (B) Add 26 new strategies to reach
100 total, focusing on orthogonal signals and unused features.

---

## Phase 12A: Forecast Cache Database -- COMPLETE

### Problem

`recompute-all` trained every strategy 6+ times per run (3 benchmarks + 1 hidden
+ 2 market sims), taking ~8-10 minutes even when nothing changed. The legacy
pickle fingerprint cache was coarse-grained (single SHA-256 over ALL strategy
files), so changing one strategy invalidated ALL caches.

### Solution

Two-tier caching:

1. **Legacy pickle fingerprint** (retained): Skips benchmark and hidden-test
   backtest pickles when nothing has changed.
2. **New SQLite forecast cache** (`data/results/forecast_cache.db`): Per-strategy
   fingerprinting. Each strategy's source file + dataset is hashed individually.
   Forecasts and backtest results are stored per-strategy per-year. Market
   simulation loads from cache, bypassing `fit()` + `forecast()` entirely.

### Results

| Metric | Before | After |
|--------|--------|-------|
| First run (cold cache) | ~8-10 min | ~8-10 min (same) |
| Second run (warm cache) | ~8-10 min | **2.4 seconds** |
| Single-strategy change | ~8-10 min | ~30-60 seconds (only that strategy re-trained) |

### Files Changed

| File | Change |
|------|--------|
| `src/.../backtest/forecast_cache.py` | **NEW** -- SQLite cache module (~330 lines) |
| `src/.../backtest/futures_market_runner.py` | Added `cached_forecasts`/`cached_results` params |
| `src/.../backtest/recompute.py` | Rewritten with forecast cache integration |
| `src/.../backtest/__init__.py` | Added cache function exports |
| `tests/backtest/test_forecast_cache.py` | **NEW** -- 19 tests |

### Checklist

- [x] SQLite schema (WAL mode, per-year tables)
- [x] Per-strategy fingerprinting (SHA-256 of source + dataset)
- [x] Store/load forecasts and backtest results
- [x] `_populate_forecast_cache()` in recompute.py
- [x] `cached_forecasts` parameter in `run_futures_market_evaluation()`
- [x] `--clear-cache` CLI flag
- [x] 19 tests passing
- [x] Full test suite passing (1108 tests)
- [x] `recompute-all` warm cache: 2.4 seconds
- [x] All theorems verified

---

## Phase 12B: Strategy Expansion to 100 -- PENDING

### Design Principles

1. **Orthogonal forecasts**: Max pairwise correlation < 0.85 with existing strategies
2. **Balanced long/short**: Target 35-65% long ratio
3. **Exploit unused features**: `weather_shortwave_radiation_wm2_mean`,
   `price_de_eur_mwh_max`, `gen_wind_offshore_mw_mean`, `rolling_vol_14d`,
   individual PL/AT/CZ/DK prices
4. **No new ML regression/classification**: Avoids overcrowding the dominant ML cluster
5. **Fast execution**: `fit()` < 1s, `forecast()` < 1ms
6. **Regime-aware where possible**

### Batch Plan (26 strategies in 6 batches)

| Batch | Strategies | Theme |
|-------|-----------|-------|
| 1 (75-79) | RadiationSolar, IntradayRange, OffshoreWindAnomaly, ForecastPriceError, PolandSpread | Unused features |
| 2 (80-84) | DenmarkSpread, CzechAustrianMean, SparkSpread, CarbonGasRatio, WeeklyAutocorrelation | Spreads & ratios |
| 3 (85-89) | MonthlyMeanReversion, LoadGenerationGap, RenewableRamp, NuclearGasSubstitution, VolatilityBreakout | Supply-demand |
| 4 (90-94) | SeasonalRegimeSwitch, WeekendMeanReversion, HighVolSkip, RadiationRegime, IndependentVote | Regime & calendar |
| 5 (95-99) | MedianIndependent, SpreadConsensus, SupplyDemandBalance, ContrarianMomentum, ConvictionWeighted | Meta & ensemble |
| 6 (100) | BalancedLongShort | Milestone |

### Checklist

- [ ] Batch 1 (75-79): implement + tests
- [ ] Batch 2 (80-84): implement + tests
- [ ] Batch 3 (85-89): implement + tests
- [ ] Batch 4 (90-94): implement + tests
- [ ] Batch 5 (95-99): implement + tests
- [ ] Batch 6 (100): implement + tests
- [ ] Update strategy_registry.md to 100
- [ ] Update test_submission_strategies.py count assertion
- [ ] Full validation: ruff + pytest + verify_theorems + recompute-all
- [ ] Update ROADMAP.md, AGENTS.md
