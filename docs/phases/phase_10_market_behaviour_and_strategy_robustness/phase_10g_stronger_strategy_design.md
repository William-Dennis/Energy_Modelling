# Phase 10g: Stronger Strategy Design

## Status: PLANNED

## Objective

Translate the findings from Phase 10a-10f into concrete design principles and a
prioritised backlog of stronger market-robust strategies.

## Strategy Theme

This sub-phase makes explicit that better understanding of the market should
lead to better strategies, not just better explanation.

## Design Principles To Test

- orthogonal signals over crowded signal families
- balanced long/short behaviour over persistent directional bias
- regime-aware forecasting over single-regime logic
- selective high-conviction forecasts over noisy always-on models
- market-aware ensembles over naive forecast averaging

## Candidate Outcome Types

- revise existing strategies that overfit standalone entry-price assumptions
- add strategies from underrepresented feature families
- design market-aware ensembles that reward diversity, not just average skill
- prune or de-emphasise strategies that mainly destabilise the market

## Acceptance Criteria For A Stronger Strategy

- competitive standalone performance
- acceptable market-adjusted performance
- positive or neutral effect on aggregate market accuracy
- non-redundant information contribution
- defensible economic or statistical rationale

## Checklist

- [ ] Convert Phase 10 findings into explicit design rules:
  - document which signal families are overcrowded vs underrepresented
  - document which forecast properties (balanced, regime-aware, selective) improve
    market quality
- [ ] Review the expansion strategy registry (`docs/expansion/strategy_registry.md`)
  for gaps that align with the design rules
- [ ] Define a shortlist of 5-10 candidate new strategies or revisions:
  - for each candidate: name, rationale, expected market contribution, difficulty
- [ ] Prioritise candidates using a scoring matrix:
  - expected standalone performance
  - expected market contribution (from Phase 10f metrics)
  - implementation complexity
  - orthogonality to existing pool
- [ ] For the top-3 candidates, write a 1-paragraph implementation brief
- [ ] Connect candidates to existing issue tracker:
  - update `issues/issue_3_new_strategies.md` if new strategies are proposed
  - link to `docs/expansion/` where applicable
- [ ] Specify acceptance criteria for the next build phase:
  - minimum standalone PnL threshold
  - non-negative market contribution (leave-one-out MAE test)
  - pass all existing tests after integration
