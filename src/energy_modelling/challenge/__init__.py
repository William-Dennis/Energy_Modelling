"""Hackathon challenge helpers for daily DE-LU futures strategies."""

from energy_modelling.challenge.data import (
    HIDDEN_TEST_YEARS,
    PUBLIC_TRAIN_YEARS,
    PUBLIC_VALIDATION_YEARS,
    build_daily_challenge_frame,
    build_feature_glossary,
    build_public_daily_dataset,
    strip_hidden_labels,
    write_challenge_data,
)
from energy_modelling.challenge.market import (
    MarketEquilibrium,
    MarketIteration,
    run_market_to_convergence,
)
from energy_modelling.challenge.market_runner import (
    MarketEvaluationResult,
    run_market_evaluation,
)
from energy_modelling.challenge.runner import ChallengeBacktestResult, run_challenge_backtest
from energy_modelling.challenge.scoring import (
    compute_challenge_metrics,
    compute_market_adjusted_metrics,
    leaderboard_score,
    market_leaderboard_score,
)
from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy

__all__ = [
    "ChallengeBacktestResult",
    "ChallengeState",
    "ChallengeStrategy",
    "HIDDEN_TEST_YEARS",
    "MarketEquilibrium",
    "MarketEvaluationResult",
    "MarketIteration",
    "PUBLIC_TRAIN_YEARS",
    "PUBLIC_VALIDATION_YEARS",
    "build_daily_challenge_frame",
    "build_feature_glossary",
    "build_public_daily_dataset",
    "compute_challenge_metrics",
    "compute_market_adjusted_metrics",
    "leaderboard_score",
    "market_leaderboard_score",
    "run_challenge_backtest",
    "run_market_evaluation",
    "run_market_to_convergence",
    "strip_hidden_labels",
    "write_challenge_data",
]
