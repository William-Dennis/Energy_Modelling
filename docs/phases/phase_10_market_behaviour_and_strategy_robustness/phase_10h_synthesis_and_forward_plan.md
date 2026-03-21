# Phase 10h: Synthesis and Forward Plan

## Status: COMPLETE

## Objective

Combine the outputs of Phase 10 into one coherent explanation of current market
behaviour and one practical roadmap for subsequent implementation work.

---

## 1. Unified Explanation of Current Market Dynamics

### What the Market Is Doing

The synthetic futures market implements an iterative equilibrium-finding process
where 67 strategies trade against a market price, earn profits, receive weights
proportional to their positive cumulative profits, and update the market price
as a weighted average of their forecasts. This process repeats until prices
converge or an iteration cap is reached.

Under the current production configuration (`ema_alpha=0.1`, 500 max
iterations, `convergence_threshold=0.01`), the market exhibits **two distinct
failure modes** depending on the data year:

**2024 (366 dates): Oscillating Non-Convergence**
- The market does not converge after 500 iterations (final delta ~1.77).
- Active strategies collapse rapidly from 60 to ~5-8 within the first 50
  iterations, then oscillate between 5-8 for the remaining 450 iterations.
- MAE improves from ~20 to ~10.3 by iteration 90, then plateaus and
  occasionally worsens.
- The oscillation is driven by ML regression strategies that produce
  near-identical forecasts. When one ML strategy gains weight, it pulls the
  market price toward its forecast, causing other ML strategies to lose
  profitability. This creates a feedback loop where leadership rotates among
  a small set of highly correlated strategies (57 leadership changes observed
  in the high-volatility sentinel window).

**2025 (365 dates): Absorbing Collapse**
- The market converges at iteration 327 but only because all active strategies
  are eliminated (0 active strategies at convergence).
- Active strategies collapse from 60 to 4 by iteration 300, then the last 4
  strategies lose profitability in rapid succession (iterations 301-326).
- Once all strategies have zero or negative profit, the weight normalisation
  produces uniform zero weights, the market price freezes, and the convergence
  criterion is trivially satisfied (delta = 0).
- This is a degenerate convergence: the market "converges" by ceasing to
  function, not by finding an informative equilibrium.

### Why This Happens: The Causal Chain

The root cause is the interaction of three design features:

1. **Positive-profit truncation** (`w_i = max(Pi_i, 0) / sum(max(Pi_j, 0))`):
   This creates a one-way ratchet. Once a strategy's cumulative profit goes
   negative, it receives zero weight and can never recover. This is the primary
   driver of active-strategy collapse. It is mathematically equivalent to a
   barrier absorption process.

2. **High forecast redundancy**: The ML regression cluster (11 strategies)
   produces pairwise forecast correlations above 0.99. When the market price
   approaches one ML strategy's forecast, it simultaneously approaches all ML
   strategies' forecasts, causing them to become unprofitable at the same rate.
   This amplifies the concentration dynamics.

3. **EMA dampening** (`ema_alpha=0.1`): While EMA dampening prevents the
   violent oscillations seen in the undampened model (`ema_alpha=1.0`), it
   slows convergence dramatically. In 2024, 500 iterations is insufficient for
   convergence. In 2025, the slow convergence gives the positive-profit trap
   time to eliminate all strategies before a genuine equilibrium is reached.

### Historical vs Current Truth

| Claim | Source | Current Status |
|-------|--------|---------------|
| Market converges as a contraction mapping | Phase 7 | TRUE only for `ema_alpha=1.0` (undampened). Not applicable to production. |
| `running_avg_k=5` is the adopted oscillation fix | Phase 8 | FALSE. Never implemented. EMA dampening (`ema_alpha=0.1`) is the actual production mechanism. |
| Market produces informative equilibrium prices | Phase 5 | PARTIALLY TRUE. Early iterations (1-100) improve price quality, but later iterations either plateau (2024) or degrade via collapse (2025). |
| 67 strategies provide diverse market information | Expansion | FALSE for market weighting. 11 ML strategies capture >90% of weight; 49 strategies contribute <5% combined. |
| 2024 and 2025 show qualitatively similar dynamics | Implicit | FALSE. 2024 oscillates without converging; 2025 collapses to absorbing state. Different mechanisms dominate. |

### Which Behaviours Are Acceptable vs Pathological

| Behaviour | Verdict | Reasoning |
|-----------|---------|-----------|
| Rapid early MAE improvement (iters 1-50) | **Acceptable** | The market genuinely reprices forecasts toward reality in early iterations. |
| Active strategy collapse from 60 to <10 | **Pathological** | Eliminates 85%+ of strategy information. Most strategies never meaningfully participate. |
| Convergence via zero-active absorbing state | **Pathological** | Not a genuine equilibrium. Market produces frozen prices, not informed prices. |
| Oscillating non-convergence at 500 iters | **Acceptable but wasteful** | The oscillation itself is informative (leadership competition), but 500 iterations with no convergence wastes compute without improving quality. |
| ML cluster capturing >90% weight | **Pathological** | Contradicts the goal of diverse strategy aggregation. The market effectively becomes a single ML model. |
| Weight entropy collapsing near zero | **Pathological** | A healthy market should maintain weight diversity across strategy families. |

---

## 2. Implications for Engine Interpretation

### Should `ema_alpha` Default Change?

**Recommendation: Yes, consider lowering to `ema_alpha=0.01` for future runs.**

Phase 10c found that `ema_alpha=0.01` is the **only** alpha value that achieves
healthy convergence (strategies surviving, genuine equilibrium) for both years.
Higher alpha values either fail to converge (2024) or converge via collapse
(2025). Lower alpha values (0.001) over-dampen and prevent meaningful price
updates.

However, this change should be treated as experimental, not as an immediate
production default. The alpha=0.01 result was observed with the current
67-strategy pool and should be re-validated after any strategy pool changes.

| Alpha | 2024 Outcome | 2025 Outcome | Recommendation |
|-------|-------------|-------------|----------------|
| 0.001 | No convergence, frozen | No convergence, frozen | Too low |
| 0.01 | Healthy convergence | Healthy convergence | **Best candidate** |
| 0.05 | Non-convergence, slow oscillation | Collapse near 500 iters | Marginal |
| 0.10 | Non-convergence (current) | Collapse at iter 327 (current) | Problematic |
| 0.50 | Violent oscillation | Rapid collapse | Too high |
| 1.00 | Extreme oscillation | Immediate collapse | Theoretical only |

### Should the Iteration Cap Change?

**Recommendation: Reduce to 200 iterations for production runs.**

Phase 10b showed that MAE improvement plateaus by iteration 90-100 in both
years. Iterations 100-500 provide no additional price quality improvement in
2024 (oscillation) and actively degrade quality in 2025 (collapse). A cap of
200 iterations captures all meaningful price improvement while saving 60% of
compute and avoiding the collapse pathology in 2025.

If `ema_alpha` is lowered to 0.01, the iteration cap should be re-evaluated
since slower dampening requires more iterations for convergence.

### Should the Convergence Criterion Change?

**Recommendation: Add an active-strategy floor alongside the delta criterion.**

The current criterion (`max|P_k - P_{k-1}| < 0.01`) is necessary but not
sufficient. It is trivially satisfied when all strategies have zero weight
(the absorbing state). A healthier criterion would be:

```
converged = (delta < threshold) AND (active_strategies >= min_active)
```

where `min_active` could be set to 3-5 strategies. This would prevent
degenerate convergence and force the market to find a genuine equilibrium or
report non-convergence honestly.

Additionally, an **early stopping** criterion based on MAE plateau detection
would improve efficiency:

```
early_stop = (MAE has not improved by > 0.1 in the last 50 iterations)
```

---

## 3. Implications for Dashboard Interpretation

### How Should Market Results Be Displayed?

**Recommendation: Add convergence status badges and quality indicators.**

Currently, the dashboard displays market results without distinguishing between
genuine convergence and degenerate convergence. Users may incorrectly interpret
a "converged" 2025 result as a high-quality equilibrium when it is actually an
absorbing-state artifact.

Proposed display changes:

1. **Convergence badge**: Show one of three states:
   - "Converged (healthy)" -- delta < threshold AND active strategies >= 5
   - "Converged (degenerate)" -- delta < threshold AND active strategies < 5
   - "Not converged" -- delta >= threshold after max iterations

2. **Active strategy count**: Display the number of strategies with non-zero
   weight at the final iteration, alongside the total strategy count. E.g.,
   "8 / 67 strategies active".

3. **Effective iteration count**: Show the iteration at which MAE plateaued,
   not just the total iterations run. This communicates how much of the
   computation was informative.

### Should Non-Converged Runs Be Flagged Differently?

**Yes.** Non-converged runs (2024) should display:
- A visible warning that the market did not reach equilibrium
- The best MAE achieved and the iteration at which it occurred
- The final oscillation amplitude (delta) as a stability indicator

### Should Active-Strategy Collapse Be Surfaced?

**Yes.** The dashboard should show:
- A time series or sparkline of active strategy count over iterations
- The weight distribution at the final iteration (to show concentration)
- A warning when weight entropy drops below a threshold (e.g., <1.0 nats),
  indicating that the market is dominated by a small number of strategies

---

## 4. Next-Phase Implementation Recommendations

### Priority 1: Engine Improvements (Phase 11a)

These changes address the pathological behaviours identified in Phase 10 and
should be implemented before adding new strategies, since new strategies will
be designed and tested against the improved engine.

| # | Change | Rationale | Difficulty | Impact |
|---|--------|-----------|------------|--------|
| 1 | Add active-strategy floor to convergence criterion | Prevent degenerate convergence (absorbing state) | Low | High |
| 2 | Add early stopping on MAE plateau | Save compute, prevent over-iteration | Low | Medium |
| 3 | Experiment with `ema_alpha=0.01` as new default | Only alpha achieving healthy convergence for both years | Low | High |
| 4 | Replace positive-profit truncation with softmax weights | Eliminate one-way ratchet; allow strategy recovery | Medium | High |
| 5 | Add weight entropy floor (redistribute when entropy < threshold) | Prevent ML cluster from monopolising >90% weight | Medium | Medium |

**Recommendation**: Implement changes 1-3 first (low difficulty, high impact),
then evaluate whether changes 4-5 are still needed.

### Priority 2: Strategy Pool Refinement (Phase 11b)

After engine improvements stabilise, refine the strategy pool using the design
rules from Phase 10g.

| # | Action | Rationale | Difficulty |
|---|--------|-----------|------------|
| 1 | Implement Diverse Ensemble v2 | Diversity-weighted ensemble to reduce ML cluster dominance | Medium |
| 2 | Implement Spread Momentum | Exploits most orthogonal signal family (cross-border) | Low |
| 3 | Implement Selective High-Conviction wrapper | Addresses per-date accuracy loss by skipping low-confidence days | Low |
| 4 | Prune redundant ML strategies (keep top-3 by orthogonality) | 11 -> 3 reduces redundancy from >0.99 to <0.90 correlation | Low |
| 5 | Re-run LOO analysis on refined pool | Validate that changes improve market quality | Medium |

### Priority 3: Dashboard Enhancements (Phase 11c)

Surface the convergence quality and strategy health information identified
above.

| # | Enhancement | Difficulty |
|---|-------------|------------|
| 1 | Add convergence status badge (healthy/degenerate/not converged) | Low |
| 2 | Add active strategy count to market results display | Low |
| 3 | Add weight distribution chart at final iteration | Medium |
| 4 | Add MAE-over-iteration sparkline | Medium |

### Recommended Phase 11 Scope

**Phase 11: Engine Hardening and Strategy Refinement**

- **11a**: Engine convergence improvements (items P1.1-P1.3, ~1-2 days)
- **11b**: Strategy pool refinement (items P2.1-P2.4, ~2-3 days)
- **11c**: Dashboard convergence reporting (items P3.1-P3.4, ~1-2 days)
- **11d**: Re-run full market simulation with improved engine + refined pool
  and validate against Phase 10 baselines (~1 day)
- **11e**: Documentation and synthesis (~0.5 day)

**Success criteria for Phase 11**:
1. Both 2024 and 2025 achieve healthy convergence (delta < threshold, active
   strategies >= 5)
2. Final MAE improves by at least 5% over current baseline
3. Weight entropy at convergence > 1.5 nats (vs current ~0.2)
4. At least 3 strategy families have > 5% weight at convergence
5. All 1000+ tests pass, ruff clean, theorems verified

---

## 5. Answers to Final Questions

### Q1: What is the best current explanation of the market dynamics?

The market is a profit-weighted iterative aggregation process that suffers from
two interacting pathologies: (a) positive-profit truncation creates irreversible
strategy elimination, and (b) extreme forecast redundancy among ML strategies
causes weight concentration and oscillation. The EMA dampening (`alpha=0.1`)
masks but does not fix these issues. See Section 1 for the full causal chain.

### Q2: Which behaviours should the platform preserve versus mitigate?

**Preserve**: Early-iteration price improvement (iters 1-50) and the general
principle of profit-weighted strategy selection. **Mitigate**: Absorbing-state
convergence, active-strategy collapse below 5, weight concentration above 90%
in a single cluster, and wasteful iteration beyond the MAE plateau.

### Q3: Which strategy improvements are most likely to matter in practice?

1. Pruning redundant ML strategies (11 -> 3) to reduce forecast correlation
2. Adding the Diverse Ensemble v2 to redistribute weight by diversity
3. Adding Spread Momentum and Selective High-Conviction for orthogonal signals

These three changes together address the redundancy, concentration, and
signal-diversity problems that drive all observed pathologies.

### Q4: What should the next implementation phase build first?

Engine convergence improvements (active-strategy floor, early stopping,
alpha=0.01) should come first because they change the environment in which
strategies are evaluated. Implementing new strategies against the current
pathological engine would produce misleading LOO results.

---

## Checklist

- [x] Write final synthesis summary (1-2 pages):
  - unified explanation of current market dynamics
  - clear distinction: what is historical vs what is current
  - which behaviours are acceptable vs pathological
- [x] Record implications for engine interpretation:
  - should `ema_alpha` default change?
  - should iteration cap change?
  - should convergence criterion change?
- [x] Record implications for dashboard interpretation:
  - how should market results be displayed to users?
  - should non-converged runs be flagged differently?
  - should active-strategy collapse be surfaced?
- [x] Produce the next-phase implementation recommendations:
  - prioritised list of engine changes (if any)
  - prioritised list of new/revised strategies (from Phase 10g)
  - recommended Phase 11 scope and structure
- [x] Update `docs/phases/ROADMAP.md` with Phase 10 completion status and
  Phase 11 placeholder
