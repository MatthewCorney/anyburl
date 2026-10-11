"""Learned aggregation of rule evidence into candidate rankings."""

from .features import CandidateFeature, CandidateFeaturizer, build_features, column_of
from .model import AggregatorConfig, LinearAggregator
from .scorer import AggregatedScorer
from .training import (
    CandidateSampling,
    KnownAnswers,
    RankingQuery,
    TrainingSet,
    build_training_set,
    ranking_queries,
)

__all__ = [
    "AggregatedScorer",
    "AggregatorConfig",
    "CandidateFeature",
    "CandidateFeaturizer",
    "CandidateSampling",
    "KnownAnswers",
    "LinearAggregator",
    "RankingQuery",
    "TrainingSet",
    "build_features",
    "build_training_set",
    "column_of",
    "ranking_queries",
]
