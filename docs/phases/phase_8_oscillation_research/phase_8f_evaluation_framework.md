# Phase 8f: Evaluation Framework

## Status: PENDING

## Objective

Define a rigorous, reproducible evaluation framework for comparing all proposed
oscillation remedies across Phases 8b-8e.  This ensures apples-to-apples
comparison and prevents cherry-picking results.

---

## Datasets

All experiments must be evaluated on both available years:

| Dataset | Split | Days | Real Prices | Notes |
|---------|-------|------|-------------|-------|
| 2024 (validation) | `backtest_val_2024.pkl` | 366 | Available | Primary evaluation dataset |
| 2025 (hidden test) | `backtest_hid_2025.pkl` | 365 | Not in CSV | Secondary; can compare market prices across methods but not MAE vs real |

For 2024, all accuracy metrics can be computed.  For 2025, only convergence
metrics (delta, iterations, stability) can be compared — not accuracy vs real.

---

## Baselines

Every experiment must be compared against these baselines:

| Baseline | Description | 2024 MAE | 2024 RMSE |
|----------|-------------|----------|-----------|
| **Prev-day** | `last_settlement_price` (no market) | 22.02 | 31.26 |
| **Iter 0** | First iteration of undampened engine | 15.10 | — |
| **Undampened final** | Current engine, 20 iterations | 19.39 | 28.39 |

The key comparison is **iter 0 MAE = 15.10**.  Any solution that converges but
produces MAE > 15.10 is *worse than running one iteration and stopping*.

---

## Primary Metrics

### 1. Convergence

| Metric | Definition | Target |
|--------|-----------|--------|
| `converged` | max |delta_t| < 0.01 EUR/MWh | True |
| `convergence_delta` | max |P^m_t^(K) - P^m_t^(K-1)| at final iter | < 0.01 |
| `iterations_to_converge` | K where delta first drops below threshold | < 50 |
| `stable_cycle` | Whether delta enters a repeating cycle (no convergence) | False |

### 2. Accuracy (2024 only)

| Metric | Definition | Target |
|--------|-----------|--------|
| `MAE` | mean |P^m_final - P_real| across all days | < 15.10 (beat iter 0) |
| `RMSE` | sqrt(mean (P^m_final - P_real)^2) | < 28.39 (beat undampened) |
| `Bias` | mean (P^m_final - P_real) | ~ 0 (unbiased) |
| `Max error` | max |P^m_final - P_real| | Report (context for tail risk) |

### 3. Stability

| Metric | Definition | Target |
|--------|-----------|--------|
| `delta_trajectory` | Full sequence of deltas across iterations | Monotonically decreasing |
| `oscillation_amplitude` | max(delta) - min(delta) in last 5 iterations | < 1.0 EUR/MWh |
| `weight_entropy` | -sum w_i log w_i across strategies at final iter | Higher = more distributed |

---

## Per-Day Diagnostics

For each method, produce per-day diagnostics on the 5 worst oscillation days
identified in Phase 8a:

| Date | Method | P_final | P_real | Error | Iter-0 Price | Improvement vs Iter 0 |
|------|--------|---------|--------|-------|--------------|----------------------|

The 5 sentinel days for 2024:
1. 2024-12-15 (worst oscillation: 57.8 EUR/MWh swing)
2. 2024-10-14 (54.1 swing)
3. 2024-01-13 (46.2 swing)
4. 2024-11-24 (40.8 swing)
5. 2024-03-25 (40.4 swing)

---

## Experiment Registry

Each experiment is assigned a unique ID for tracking:

| ID | Phase | Method | Key Hyperparams |
|----|-------|--------|-----------------|
| B1 | 8b | Fixed dampening | alpha |
| B2 | 8b | Adaptive dampening | alpha_max, gamma |
| B3 | 8b | Two-phase convergence | phase1_alpha, phase1_threshold |
| C1 | 8c | Weight cap | w_max |
| C2 | 8c | Weighted median | (none) |
| C3 | 8c | Log-profit weighting | (none) |
| C4 | 8c | Cluster-aware averaging | gap_threshold |
| D1 | 8d | Rolling mean init | window |
| D2 | 8d | Forecast mean init | (none) |
| D3 | 8d | Forecast clipping | max_deviation |
| D4 | 8d | Percentile init | window, percentile |
| E1 | 8e | Running average smoothing | K |
| E2 | 8e | EMA smoothing | beta |
| E3 | 8e | Best-iteration selection | (none) |
| E4 | 8e | Delta-weighted average | (none) |

Total: 15 experiments across 4 research tracks.

---

## Combination Experiments

After evaluating individual methods, test the most promising combinations:

| ID | Components | Rationale |
|----|-----------|-----------|
| X1 | B1 + C3 | Dampening + log-profit (attack both gain and weight sensitivity) |
| X2 | B2 + C1 | Adaptive dampening + weight cap (two complementary controls) |
| X3 | D1 + B1 | Rolling init + dampening (reduce initial amplitude, then converge) |
| X4 | B1 + C2 | Dampening + weighted median (robust aggregation with convergence) |
| X5 | E1 + C3 | Post-hoc smoothing + log-profit (no engine change needed for smoothing) |

---

## Implementation: Evaluation Script

```python
# scripts/evaluate_oscillation_remedies.py

"""
Evaluate all oscillation remedies from Phase 8.

Usage:
    uv run python scripts/evaluate_oscillation_remedies.py --experiments B1,B2,C1 --year 2024
    uv run python scripts/evaluate_oscillation_remedies.py --all --year 2024

Output:
    - Console table with primary metrics
    - CSV file: data/results/phase8_evaluation.csv
    - Per-day diagnostics for sentinel days
"""
```

The script should:
1. Load the backtest results and strategy forecasts from existing pkl files
2. For each experiment, run the modified market engine
3. Compute all metrics from the tables above
4. Output a comparison table sorted by MAE (primary) and convergence (secondary)

---

## Reporting Template

Each completed experiment should fill in this template:

```markdown
### Experiment [ID]: [Name]

**Configuration**: [hyperparams]

| Metric | 2024 | 2025 |
|--------|------|------|
| Converged | | |
| Delta | | |
| Iterations | | |
| MAE | | N/A |
| RMSE | | N/A |
| Bias | | N/A |

**Sentinel days (2024)**:
| Date | P_final | P_real | Error |
|------|---------|--------|-------|

**Assessment**: [Pass/Fail/Partial] — [1-2 sentence summary]
```

---

## Success Criteria for Phase 8 Overall

Phase 8 is considered successful if:

1. **At least one method achieves convergence** on both 2024 and 2025 (delta < 0.01)
2. **Converged MAE <= 15.10** on 2024 (no worse than iter 0)
3. **Iterations <= 50** (computationally practical)
4. **Method is principled** — has a clear theoretical justification, not just
   empirical hyperparameter tuning
5. **Method generalises** — does not rely on properties specific to the current
   67-strategy set

If no single method achieves all 5 criteria, the best combination (X1-X5) is
the recommended solution.

---

## Files to Create

- `scripts/evaluate_oscillation_remedies.py` — unified evaluation script
- `data/results/phase8_evaluation.csv` — results table (generated)
- `docs/phases/phase_8_oscillation_research/results.md` — filled-in reporting template (generated after experiments)
