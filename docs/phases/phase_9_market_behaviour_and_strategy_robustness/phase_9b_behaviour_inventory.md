# Phase 9b: Behaviour Inventory

## Status: PLANNED

## Objective

Catalogue the full set of behaviours exhibited by the current synthetic futures
market before attempting deeper causal explanation or intervention.

## Why This Matters

The project can no longer treat market behaviour as a single binary outcome such
as "converges" or "oscillates". The current artifacts suggest several distinct
behaviour classes that need separate explanation.

## Candidate Behaviour Classes

- fast convergence
- slow convergence
- damped non-convergence within the current iteration budget
- absorbing carry-forward convergence (all weights go to zero)
- active-strategy collapse
- early-iteration accuracy improvement followed by later degradation
- cluster switching / regime switching

## Core Metrics

- convergence delta by iteration
- MAE / RMSE / bias by iteration
- active strategy count by iteration
- top-strategy weight and top-5 concentration
- weight entropy by iteration
- distance from iter-0 and from final state

## Initial Focus

Based on the current artifacts, Phase 9b should prioritise:

1. 2024 slow damped motion without convergence by 500 iterations
2. 2025 convergence via active-strategy collapse to zero
3. shrinking active sets in both years
4. cases where "converged" may not imply "good consensus"

## Planned Outputs

- one compact per-year behaviour summary table
- one run-type classification per saved artifact
- one list of high-priority behaviours for Phase 9c-9e

## Checklist

- [ ] Compute per-iteration metric panels for 2024 and 2025
- [ ] Classify each run into one or more behaviour types
- [ ] Identify the highest-information behaviours to explain next
