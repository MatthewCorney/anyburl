"""Learned aggregation of rule evidence into candidate rankings."""

from .features import (
    CandidateFeature,
    CandidateFeaturizer,
    QueryFeaturizer,
    build_features,
    column_of,
)
from .model import AggregatorConfig, LinearAggregator
from .scorer import AggregatedScorer
from .shared import SharedFeature, SharedFeaturizer, build_shared_features
from .training import (
    CandidateSampling,
    CandidateScope,
    KnownAnswers,
    RankingQuery,
    TrainingSet,
    build_training_set,
    concat_training_sets,
    ranking_queries,
)

__all__ = [
    "AggregatedScorer",
    "AggregatorConfig",
    "CandidateFeature",
    "CandidateFeaturizer",
    "CandidateSampling",
    "CandidateScope",
    "KnownAnswers",
    "LinearAggregator",
    "QueryFeaturizer",
    "RankingQuery",
    "SharedFeature",
    "SharedFeaturizer",
    "TrainingSet",
    "build_features",
    "build_shared_features",
    "build_training_set",
    "column_of",
    "concat_training_sets",
    "ranking_queries",
]
