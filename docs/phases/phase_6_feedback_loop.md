# Phase 6: EDA Feedback Loop

## Status: COMPLETE

## Objective

Feed insights from strategy performance (Phase 5) back into deeper EDA analysis.
The goal is to understand *why* certain strategies succeeded or failed, and whether
new patterns emerge when conditioning on strategy behavior.

## Prerequisites

- Phase 5 complete (strategy results analyzed) ✅

## Checklist

### 6a. Identify questions from Phase 5
- [x] List specific questions raised by strategy performance
- [x] Identify unexpected behaviors that need investigation
- [x] Identify regime/condition dependencies observed in results

### 6b. Add targeted EDA sections
- [x] Day-of-week edge stability analysis (Section 19)
- [x] Feature drift analysis: train vs validation distribution shifts (Section 20)
- [x] Quarterly direction rate patterns (Section 21)
- [x] Volatility regime performance (Section 22)
- [x] Wind quintile direction analysis (Section 23)
- [x] Strategy correlation insights — Wed/Thu feature drivers (Section 24)

### 6c. Refine strategies (optional)
- [x] Evaluated all findings — no refinements warranted (see analysis below)

### 6d. Update documentation
- [x] Update this file with findings
- [x] Cross-reference with Phase 3 hypotheses
- [x] Note implications for Phase 7

## Questions from Phase 5

1. **Is the Tuesday edge decaying?** Day Of Week strategy treats Tue as a long day, but is this still reliable?
2. **What caused December 2024 losses?** Multiple strategies had their worst day in Dec 2024.
3. **Are feature distributions drifting?** Threshold-based strategies (Wind, Load, Fossil) use train-period medians.
4. **Is Q4 consistently the best quarter?** Or was 2024 Q4 outperformance a volatility artefact?
5. **What drives direction on Wed/Thu?** These are skip days for Day Of Week — what signal works there?
6. **How stable is the wind quintile signal?** Wind Forecast was Tier 2 — could it be Tier 1?
7. **How do volatility regimes affect directional predictability?**

## Findings

### 6b-1: Tuesday Edge Stability
**Finding**: Tuesday up-edge is decaying — from +23.5% in 2022 to +4.5% in 2024 — but it has been positive *every single year* (2019-2024). Even at its weakest, it still contributes positive PnL.

**Implication**: Not yet actionable enough to drop Tuesday from the Day Of Week strategy. The edge is weakening but hasn't reversed. Monitor in 2025 hidden test.

### 6b-2: December 2024 Extreme Volatility
**Finding**: December 2024 had std=60.74 EUR/MWh (double November's 36.08), driven by a Dec 11-13 price spike where settlement hit 395.67 EUR on Dec 12. This single event was the worst day for 4 of 7 strategies; Dec 13 was worst for 2 more.

**Implication**: This is a tail-risk regime event, not a systematic strategy failure. All strategies recovered. No strategy change needed — this is the kind of regime where position sizing (not implemented in this challenge) would matter.

### 6b-3: Feature Drift (Train → Validation)
**Finding**: Significant drift in multiple features:
- Solar forecast: +30.9% shift (largest)
- Settlement price: -20.8% (prices normalized post-crisis)
- Wind onshore: +10.7%
- Gas price: -15.3%

**Implication**: Threshold-based strategies (Wind, Load, Fossil) fit medians on training data. The +30.9% solar shift is irrelevant (solar has near-zero correlation with direction). The wind shift (+10.7%) could slightly affect Wind Forecast thresholds, but the quintile-based signal is robust to level shifts because it's rank-based. No recalibration needed.

### 6b-4: Quarterly Stability
**Finding**: Q4 is NOT consistently the best quarter. Up-rates vary wildly across year-quarters with no stable seasonal pattern. 2024 Q4 outperformance is driven by higher volatility (mean |change| 32.2 vs 11.9-22.9 for other quarters), which amplifies PnL magnitude regardless of directional accuracy.

**Implication**: No seasonal strategy adjustment warranted. Strategy PnL is a function of (accuracy × magnitude), and magnitude varies more by volatility regime than by quarter.

### 6b-5: Wed/Thu Feature Drivers
**Finding**: On Wed/Thu (skip days for Day Of Week), the top direction predictor is wind offshore with correlation -0.303 (Wed) and -0.300 (Thu). This is stronger than the all-day average of -0.218 for wind offshore.

**Implication**: Composite Signal's relative strength on Wed/Thu comes from its wind component. A hypothetical "Day Of Week + Wind" hybrid could trade Wed/Thu using wind signal instead of skipping, but the added complexity may not be worth the marginal gain given Composite Signal already captures this.

### 6b-6: Wind Quintile Signal Stability
**Finding**: The wind quintile signal is rock-solid:
- Q1 (low wind): 62.2% up rate
- Q5 (high wind): 32.3% up rate
- The 20-37% spread between Q1 and Q5 is stable across ALL 6 years (2019-2024)

**Implication**: This is the second-most reliable signal after Monday. However, the existing Wind Forecast strategy already captures this via a median threshold (which approximates a binary quintile split at Q2-Q3). Moving to explicit quintile binning would add complexity without substantial edge improvement.

### 6b-7: Volatility Regime Effects
**Finding**: Direction rates are similar across volatility regimes (low ~45%, mid ~47%, high ~48%). The main difference is magnitude: high-vol regime has mean |change| ~3x that of low-vol. This means high-vol days contribute disproportionately to PnL — both wins and losses are larger.

**Implication**: Volatility regime doesn't help predict direction, but it determines bet sizing relevance. In a fixed-size challenge (direction only, no sizing), this is academic. In a real portfolio, you'd size up in high-vol regimes if accuracy is maintained.

## Strategy Refinement Decision (6c)

**Decision: No strategy refinements implemented.**

Rationale:
1. **Tuesday edge**: Still positive, just weakening. Premature to drop.
2. **Feature drift**: Rank-based signals (quintiles, medians) are naturally robust to level shifts.
3. **Wed/Thu wind signal**: Already captured by Composite Signal. A hybrid strategy would add complexity for marginal gain.
4. **Wind quintile**: Already captured by Wind Forecast's median threshold.
5. **Volatility regimes**: Don't predict direction — only affect magnitude.
6. **December tail risk**: One-off event, not a systematic failure.

All 7 Phase 4 strategies remain as-is. The existing strategy portfolio is well-diversified and properly captures the identified signals.

## New EDA Analyses Added

| Section | Analysis | Key Finding | Implication |
|---------|----------|-------------|-------------|
| 19 | Day-of-Week Edge Stability | Tuesday edge decaying but still positive all years | Monitor, don't drop |
| 20 | Feature Drift (Train→Val) | Solar +30.9%, settlement -20.8%, wind +10.7% | Rank-based signals robust |
| 21 | Quarterly Direction Rates | No stable seasonal pattern; Q4 is volatility-driven | No seasonal strategy |
| 22 | Volatility Regime Performance | Direction similar across regimes; magnitude differs 3x | Sizing matters, not direction |
| 23 | Wind Quintile Analysis | Q1-Q5 spread 20-37%, stable all 6 years | Rock-solid signal, already captured |
| 24 | Strategy Correlation Insights | Wind offshore strongest Wed/Thu driver (-0.30) | Composite Signal captures this |

## Code Changes

### New pure computation functions in `eda_analysis.py` (5 functions)
- `day_of_week_edge_by_year()` — Edge (up_rate - baseline) by day-of-week and year
- `feature_drift()` — Distribution shift between train and validation features
- `quarterly_direction_rates()` — Up-rate and mean |change| by year-quarter
- `volatility_regime_performance()` — Direction stats by low/mid/high volatility regime
- `wind_quintile_analysis()` — Direction rates by wind power quintile

### New tests in `test_eda_analysis.py` (14 tests)
All 38 tests pass (24 Phase 2 + 14 Phase 6).

### New dashboard sections in `_eda.py` (6 sections, 19-24)
- Added `_load_challenge_data()` cached loader for daily challenge CSV
- Added 6 section renderers using daily challenge data
- Dashboard now has 24 total sections (12 original + 6 Phase 2 + 6 Phase 6)

## Cross-References

- **Phase 3 Hypotheses**: All 7 hypotheses (H1-H7) remain validated. Phase 6 findings reinforce rather than contradict Phase 3.
- **Phase 5 Results**: All strategy tiers confirmed. No tier changes from Phase 6 analysis.
- **Phase 7**: The market simulation oscillation (discovered in Phase 5) remains the key open question. Phase 6 analysis of feature stability suggests the oscillation is structural, not data-dependent.
