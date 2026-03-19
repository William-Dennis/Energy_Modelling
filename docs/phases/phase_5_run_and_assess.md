# Phase 5: Run Challenge and Assess

## Status: COMPLETE ✅

## Objective

Execute all implemented strategies through the challenge runner and market simulation.
Analyze performance, identify which hypotheses held, which failed, and why.
This assessment feeds back into Phase 6 (deeper EDA) and Phase 7 (convergence analysis).

## Prerequisites

- Phase 4 complete (strategies implemented and unit-tested)

## Bug Fix: `reset()` Must Not Clear Fitted Parameters

**Discovery**: The runner calls `strategy.fit(train_data)` then immediately
`strategy.reset()` (runner.py lines 81-82). Original H2-H7 implementations
had `reset()` clearing fitted parameters (thresholds, means, stds), causing
`RuntimeError` on the first `act()` call.

**Fix**: Changed all `reset()` implementations to `pass` (no-op). None of
these strategies carry per-run ephemeral state. Updated tests to assert
`reset()` preserves fitted parameters.

**Affected files**: `wind_forecast.py`, `load_forecast.py`, `lag2_reversion.py`,
`fossil_dispatch.py`, `composite_signal.py` + their test files.

## Checklist

### 5a. Run challenge backtest (yesterday-settlement pricing)
- [x] Run all strategies on 2024 validation period
- [x] Record leaderboard rankings

### 5b. Run market simulation
- [x] Run market evaluation with all strategies (50 iterations)
- [x] Record convergence status — **DID NOT CONVERGE** (oscillates)
- [x] Record market-adjusted rankings
- [x] Compare original vs market-adjusted rankings

### 5c. Performance analysis per strategy
- [x] For each strategy: Total PnL, Sharpe, max drawdown, win rate, trade count
- [x] Identify which strategies gained/lost rank under market pricing
- [x] Identify which hypotheses from Phase 3 held vs failed — **ALL 7 HELD**
- [x] Analyze failure modes (when does each strategy lose money?)

### 5d. Cross-strategy analysis
- [x] Correlation matrix of daily returns across strategies
- [x] Identify strategy clusters
- [x] Identify truly independent signals
- [x] Analyze which strategies contribute to vs harm market convergence

### 5e. Document findings
- [x] Update this file with results tables
- [x] Identify insights that should feed back into EDA (Phase 6)
- [x] Identify questions for convergence analysis (Phase 7)

---

## Results

### Yesterday-Settlement Leaderboard (2024)

| Rank | Strategy | Total PnL | Sharpe | Max DD | Win Rate | Trades | PF |
|------|----------|-----------|--------|--------|----------|--------|----|
| 1 | Composite Signal | 100,241 | 6.22 | 3,093 | 0.66 | 366 | 3.15 |
| 2 | Day Of Week | 93,213 | 6.84 | 1,518 | 0.72 | 262 | 4.75 |
| 3 | Weekly Cycle | 72,640 | 4.35 | 5,893 | 0.60 | 366 | 2.20 |
| 4 | Wind Forecast | 56,808 | 3.35 | 5,226 | 0.60 | 366 | 1.83 |
| 5 | Lag2 Reversion | 51,041 | 3.44 | 4,302 | 0.61 | 235 | 2.18 |
| 6 | Load Forecast | 31,467 | 1.83 | 5,226 | 0.56 | 366 | 1.39 |
| 7 | Fossil Dispatch | 29,259 | 1.70 | 7,084 | 0.55 | 366 | 1.36 |
| 8 | Always Long | 1,241 | 0.07 | 8,992 | 0.46 | 366 | 1.01 |
| 9 | Always Short | -1,241 | -0.07 | 9,520 | 0.54 | 366 | 0.99 |

**Key Takeaways**:
- All 7 hypothesis-derived strategies outperform both naive baselines.
- Day Of Week has the best risk-adjusted return: highest Sharpe (6.84), lowest drawdown (1,518), highest profit factor (4.75).
- Composite Signal has the highest raw PnL (100,241) but worse risk metrics than Day Of Week.
- Always Long is marginally positive (+1,241) confirming the slight upward bias in 2024 prices.

### Market-Adjusted Leaderboard (2024)

| Mkt Rank | Strategy | Mkt PnL | Orig PnL | Alpha | Sharpe | Orig Rank | Change |
|----------|----------|---------|----------|-------|--------|-----------|--------|
| 1 | Always Short | 121,086 | -1,241 | +122,327 | 7.71 | 9 | +8 |
| 2 | Load Forecast | 0 | 31,467 | -31,467 | 0.00 | 6 | +4 |
| 3 | Wind Forecast | 0 | 56,808 | -56,808 | 0.00 | 4 | +1 |
| 4 | Weekly Cycle | -20,302 | 72,640 | -92,942 | -1.17 | 3 | -1 |
| 5 | Lag2 Reversion | -26,211 | 51,041 | -77,252 | -1.82 | 5 | 0 |
| 6 | Day Of Week | -27,058 | 93,213 | -120,272 | -1.88 | 2 | -4 |
| 7 | Composite Signal | -57,188 | 100,241 | -157,430 | -3.35 | 1 | -6 |
| 8 | Fossil Dispatch | -106,026 | 29,259 | -135,285 | -6.58 | 7 | -1 |
| 9 | Always Long | -121,086 | 1,241 | -122,327 | -7.71 | 8 | -1 |

**Key Takeaways**:
- The market simulation **completely inverts** the rankings. The best original
  strategies become the worst under market pricing, because the market price
  shifts toward the consensus long direction, eliminating their edge.
- Always Short wins under market pricing because the consensus is overwhelmingly
  long, so the market price drifts above settlement → shorts profit.
- This confirms the market mechanism's fundamental limitation: it rewards
  contrarian positions, not accurate predictions.

### Convergence Status

- **Converged**: No
- **Iterations**: 50 (hit max)
- **Final delta**: 15.65 EUR/MWh (far above 0.01 threshold)
- **Pattern**: Stable 2-cycle oscillation between Always Long (odd iters) and
  Always Short (even iters) from iteration 5 onward.

**Oscillation Mechanism**:
1. Iter 0-2: Sophisticated strategies dominate (Composite Signal, Day Of Week) → market price shifts toward their consensus → they lose money against the new market price.
2. Iter 3: Only Fossil Dispatch + Always Long remain profitable.
3. Iter 4: Market swings the other way → Always Short + Wind/Load Forecast profit.
4. Iter 5+: Locked into a 2-cycle: {Always Long} ↔ {Always Short, Load Forecast, Wind Forecast}.

This oscillation is **structural**, not a tuning issue. It occurs because:
- Always Long and Always Short are perfectly anti-correlated (-1.000)
- The market price is a weighted average of implied forecasts
- When longs dominate, the market price rises above settlement → shorts profit
- When shorts dominate, the market price falls below settlement → longs profit
- The dampening factor (0.5) reduces amplitude but cannot break the cycle

**Implication for Phase 7**: A perfect-foresight strategy would NOT converge under
this market mechanism. The mechanism has a fundamental instability when
opposing constant-direction strategies are present.

---

## Strategy Assessment

### Hypothesis Verdicts

| Strategy | Hypothesis | PnL | Sharpe | WR | PF | Verdict |
|----------|-----------|-----|--------|----|----|---------|
| Day Of Week (H1) | Mon/Tue long, Fri/Sat/Sun short | 93,213 | 6.84 | 0.72 | 4.75 | **HELD** |
| Wind Forecast (H2) | High wind → short | 56,808 | 3.35 | 0.60 | 1.83 | **HELD** |
| Load Forecast (H3) | High load → long | 31,467 | 1.83 | 0.56 | 1.39 | **HELD** |
| Lag2 Reversion (H4) | Fade large lag-2 move | 51,041 | 3.44 | 0.61 | 2.18 | **HELD** |
| Weekly Cycle (H5) | Follow lag-7 direction | 72,640 | 4.35 | 0.60 | 2.20 | **HELD** |
| Fossil Dispatch (H6) | High fossil → short | 29,259 | 1.70 | 0.55 | 1.36 | **HELD** |
| Composite Signal (H7) | Weighted z-score of 6 features | 100,241 | 6.22 | 0.66 | 3.15 | **HELD** |

All 7 hypotheses produced positive PnL with Sharpe > 1.0 and Profit Factor > 1.0.

### Strategy Tiers

- **Tier 1 — Strong** (Sharpe > 4.0): Day Of Week (6.84), Composite Signal (6.22), Weekly Cycle (4.35)
- **Tier 2 — Moderate** (Sharpe 3.0-4.0): Lag2 Reversion (3.44), Wind Forecast (3.35)
- **Tier 3 — Weak** (Sharpe < 2.0): Load Forecast (1.83), Fossil Dispatch (1.70)

### Monthly PnL Breakdown (2024, EUR)

| Strategy | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Composite Signal | 4,952 | 4,662 | 9,405 | 12,417 | 3,771 | 2,859 | 6,455 | 5,194 | 6,030 | 18,340 | 12,391 | 13,766 |
| Day Of Week | 5,037 | 3,766 | 5,319 | 8,105 | 5,326 | 11,648 | 7,171 | 6,581 | 3,732 | 9,804 | 12,585 | 14,138 |
| Weekly Cycle | 879 | 1,909 | 3,868 | 5,146 | 6,112 | 11,219 | 9,499 | 6,037 | -663 | 7,167 | 8,956 | 12,511 |
| Wind Forecast | -1,419 | 898 | 6,190 | 7,181 | -682 | 11,261 | 6,669 | 4,961 | 8,652 | 2,905 | 5,064 | 5,127 |
| Lag2 Reversion | -1,286 | -384 | 998 | 3,157 | 6,101 | 7,456 | 5,208 | 1,828 | 8,236 | 10,229 | 3,239 | 6,259 |
| Load Forecast | 2,592 | 2,069 | 3,073 | 204 | 682 | 1,130 | -1,015 | 45 | -636 | 5,580 | 9,731 | 8,013 |
| Fossil Dispatch | 4,719 | 1,093 | 1,310 | 1,316 | 495 | -1,044 | 810 | -45 | 3,593 | 4,639 | 4,957 | 7,416 |

**Observations**:
- **Day Of Week** is the most consistent: positive every single month, no month below 3,700.
- **Weekly Cycle** has one losing month (Sep: -663), otherwise solid.
- **Wind Forecast** has two losing months (Jan: -1,419; May: -682).
- **Lag2 Reversion** starts slow (Jan-Feb negative), then accelerates.
- **Load Forecast** has 3 losing months (Jul, Sep), very uneven.
- **Fossil Dispatch** is similar to Load Forecast — inconsistent mid-year.
- **Q4 is strong across all strategies**: Oct-Dec contributed the most PnL.

### Worst Days Per Strategy (2024)

| Strategy | #1 Worst | #2 | #3 | #4 | #5 |
|----------|----------|-----|-----|-----|-----|
| Composite Signal | Dec 12: -3,093 | Jun 15: -1,712 | Nov 26: -1,310 | May 12: -1,241 | Nov 12: -1,071 |
| Day Of Week | Sep 10: -979 | May 3: -955 | Oct 11: -823 | Sep 24: -783 | Apr 9: -770 |
| Weekly Cycle | Dec 12: -3,093 | Nov 6: -1,590 | Sep 5: -1,531 | Apr 8: -1,414 | Dec 15: -1,335 |
| Wind Forecast | Dec 13: -5,226 | Nov 7: -1,728 | Jun 10: -1,533 | Oct 21: -1,433 | Nov 25: -1,318 |
| Lag2 Reversion | Dec 12: -3,093 | Nov 24: -1,653 | Nov 6: -1,590 | Dec 15: -1,335 | Dec 10: -1,210 |
| Load Forecast | Dec 13: -5,226 | Jun 3: -2,162 | Dec 5: -1,928 | Nov 7: -1,728 | Aug 26: -1,697 |
| Fossil Dispatch | Dec 12: -3,093 | Dec 11: -2,782 | Jun 15: -1,712 | Nov 24: -1,653 | May 1: -1,638 |

**Observations**:
- **Dec 12, 2024** was a shared disaster day for Composite Signal, Weekly Cycle, Lag2 Reversion, and Fossil Dispatch (-3,093 each).
- **Dec 13** was worst for Wind Forecast and Load Forecast (-5,226).
- **Day Of Week's worst day** (-979) is less than a third of other strategies' worst days. Its skip-days-with-no-edge approach effectively caps tail risk.
- The December cluster suggests a regime shift or outlier event.

### Direction Accuracy by Day of Week (2024)

**Composite Signal** (overall: 65.8%, 366 trades):
| Day | Accuracy | Detail |
|-----|----------|--------|
| Mon | 79.2% | 42/53 |
| Tue | 50.9% | 27/53 |
| Wed | 69.2% | 36/52 |
| Thu | 76.9% | 40/52 |
| Fri | 67.3% | 35/52 |
| Sat | 57.7% | 30/52 |
| Sun | 59.6% | 31/52 |

**Day Of Week** (overall: 71.8%, 262 trades, 104 skipped):
| Day | Accuracy | Detail |
|-----|----------|--------|
| Mon | 92.5% | 49/53 |
| Tue | 50.9% | 27/53 |
| Wed | skipped | 52 skipped |
| Thu | skipped | 52 skipped |
| Fri | 61.5% | 32/52 |
| Sat | 78.8% | 41/52 |
| Sun | 75.0% | 39/52 |

**Key findings**:
- Day Of Week's Monday accuracy (92.5%) is the single strongest signal in the entire strategy set.
- Day Of Week's Tuesday accuracy (50.9%) is coin-flip — the Tuesday-long rule is not adding value. **Phase 6 candidate: drop Tuesday from Day Of Week.**
- Composite Signal is surprisingly strong on Wednesday (69.2%) and Thursday (76.9%) — days that Day Of Week skips. **Phase 6 candidate: blend Day Of Week with Composite Signal on Wed/Thu.**

---

## Cross-Strategy Analysis

### Correlation Matrix Summary

**High positive correlation** (betting the same way):
| Pair | Correlation | Implication |
|------|-------------|-------------|
| Day Of Week ↔ Weekly Cycle | +0.622 | Both exploit weekly seasonality |
| Lag2 Reversion ↔ Fossil Dispatch | +0.516 | Fossil dispatch follows recent momentum |
| Weekly Cycle ↔ Lag2 Reversion | +0.480 | Weekly cycle partly captures mean reversion |
| Composite Signal ↔ Fossil Dispatch | +0.476 | Composite weights fossil gas signal |
| Composite Signal ↔ Day Of Week | +0.471 | Composite captures weekly patterns |
| Day Of Week ↔ Lag2 Reversion | +0.445 | Both exploit Mon/Sat patterns |
| Composite Signal ↔ Weekly Cycle | +0.431 | Shared weekly structure |
| Composite Signal ↔ Lag2 Reversion | +0.417 | Overlap in timing |
| Lag2 Reversion ↔ Load Forecast | -0.410 | Conflicting signals |

**High negative correlation** (natural diversifiers):
| Pair | Correlation | Implication |
|------|-------------|-------------|
| Load Forecast ↔ Fossil Dispatch | -0.595 | Opposite direction signals — excellent diversification pair |

**Independent signal** (low correlation with all others):
- **Wind Forecast**: max |corr| = 0.325 — the most independent signal

### Strategy Clusters

1. **Weekly Structure Cluster**: Day Of Week, Weekly Cycle, Lag2 Reversion (corr 0.44-0.62)
   - All exploit temporal patterns — overlapping edge
   - Composite Signal partially overlaps this cluster (0.42-0.47)

2. **Fundamental Feature Signals**: Wind Forecast, Load Forecast, Fossil Dispatch
   - These use D-0 forecasts / D-1 generation data
   - Load ↔ Fossil are anti-correlated (-0.595) — natural pair
   - Wind is independent of both

3. **Baselines**: Always Long / Always Short — perfectly anti-correlated (-1.000)

### Diversification Opportunities

An optimal portfolio should combine:
1. **Day Of Week** (Tier 1, Sharpe 6.84) — core position
2. **Wind Forecast** (independent signal, Sharpe 3.35) — diversifier
3. **Load Forecast OR Fossil Dispatch** (anti-correlated pair) — pick one based on which complements the rest

---

## Insights for Phase 6 (Feedback Loop)

1. **Drop Tuesday from Day Of Week**: 50.9% accuracy = coin flip. The strategy should skip Tuesdays to improve Sharpe.
2. **Composite Signal on Wed/Thu**: 69-77% accuracy on days Day Of Week skips → blend these.
3. **December regime shift**: Multiple strategies hit worst days in Dec 2024. Investigate what happened (gas crisis? cold snap?).
4. **Lag2 Reversion slow start**: Negative in Jan-Feb, then strong. Is this a seasonal effect or parameter issue?
5. **Wind Forecast independence**: At corr < 0.33 with everything, this is the purest diversifier. Could it be improved?
6. **Load Forecast weakness**: Only 56.3% accuracy, 3 losing months. May need different threshold or interaction with other features.
7. **Q4 outperformance**: All strategies are strongest in Q4. Is this stable across years?

## Questions for Phase 7 (Convergence Analysis)

1. **Why doesn't the market converge?** The 2-cycle oscillation between Always Long and Always Short is a structural property of the weighting scheme. Can we prove this analytically?
2. **Would removing constant-direction strategies help?** If only hypothesis strategies participate, does the market converge?
3. **Is perfect foresight a fixed point?** If a strategy knows settlement prices, does it dominate under the market mechanism — or does its own weight shift the market price and eliminate its edge?
4. **Dampening analysis**: Would different dampening values (0.3, 0.7, 0.9) change convergence behavior?
5. **Mechanism design**: The current weight-by-profit scheme has known limitations. Could an exponential weighting or Bayesian update mechanism avoid the oscillation?
