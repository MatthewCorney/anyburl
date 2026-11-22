"""Fit a learned aggregator for one relation of a learned rule set."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import assert_never

from ..core import EdgeTypeTuple, Triple
from ..exceptions import ConfigurationError
from ..graph import HeteroGraph
from ..prediction import PredictionConfig, RuleEvidence
from ..rule_evaluation import RuleSet
from ..stratification import RelationDegrees, relation_degrees
from .features import CandidateFeaturizer, QueryFeaturizer
from .model import AggregatorConfig, LinearAggregator
from .residual import JointFeaturizer
from .scorer import AggregatedScorer
from .shared import SharedFeaturizer
from .training import (
    CandidateSampling,
    CandidateScope,
    KnownAnswers,
    build_training_set,
    ranking_queries,
)

__all__ = ["AggregationSetup", "FeaturizerKind", "fit_aggregated_scorer"]


class FeaturizerKind(StrEnum):
    """Which features the aggregator ranks candidates from.

    Attributes
    ----------
    CANDIDATE : str
        Summary features, then one column per body chain
        (:class:`CandidateFeaturizer`).
    SHARED : str
        Relation-independent features (:class:`SharedFeaturizer`).
    JOINT : str
        The shared features, then the per-chain ones
        (:class:`JointFeaturizer`).
    """

    CANDIDATE = auto()
    SHARED = auto()
    JOINT = auto()


@dataclass(frozen=True, slots=True)
class AggregationSetup:
    """How a learned aggregator is featurized, trained and applied.

    Parameters
    ----------
    featurizer : FeaturizerKind
        Which features the model reads.
    sampling : CandidateSampling
        Negatives kept per training query. Its ``scope`` also decides which
        candidates the fitted scorer ranks.
    aggregator : AggregatorConfig
        Regularisation and iteration budget of the model.
    prediction : PredictionConfig
        Grounding settings of the rule evidence.
    """

    featurizer: FeaturizerKind = FeaturizerKind.CANDIDATE
    sampling: CandidateSampling = field(default_factory=CandidateSampling)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)

    @property
    def scope(self) -> CandidateScope:
        """Return the candidates the model is trained on and ranks."""
        return self.sampling.scope


def fit_aggregated_scorer(
    graph: HeteroGraph,
    rule_set: RuleSet,
    train_triples: Sequence[Triple],
    *,
    edge_type: EdgeTypeTuple | None = None,
    setup: AggregationSetup | None = None,
) -> AggregatedScorer:
    """Fit a linear aggregator on held-out triples of one relation.

    Candidates are featurized against ``graph``; true answers in ``graph``
    and in ``train_triples`` are never used as negatives.

    Parameters
    ----------
    graph : HeteroGraph
        The graph the rules are grounded on, without ``train_triples``.
    rule_set : RuleSet
        Learned rules of one or more relations.
    train_triples : Sequence[Triple]
        Triples of the relation to learn from, all of one edge type.
    edge_type : EdgeTypeTuple | None
        The relation to fit. ``None`` takes it from ``train_triples``.
    setup : AggregationSetup | None
        Featurizer, sampling and model settings. ``None`` uses the defaults.

    Returns
    -------
    AggregatedScorer
        Ranks queries of the relation against ``graph``.

    Raises
    ------
    ConfigurationError
        If ``train_triples`` is empty or spans several edge types, differs
        from ``edge_type``, no rule predicts the relation, or the scope
        keeps no training query.
    """
    config = setup if setup is not None else AggregationSetup()
    target = _resolve_edge_type(train_triples, edge_type)
    rules = rule_set.for_edge_type(target)
    if not rules:
        raise ConfigurationError(f"no rules predict {target!r}")
    evidence = RuleEvidence(graph, rules, config=config.prediction)
    featurizer = _build_featurizer(
        config.featurizer, evidence, relation_degrees(graph, target)
    )
    known = _known_answers(graph, target, train_triples)
    training_set = build_training_set(
        featurizer, ranking_queries(train_triples), known, config.sampling
    )
    if training_set.mask.size(0) == 0:
        raise ConfigurationError(
            f"no training query of {target!r} has an answer in scope {config.scope}"
        )
    aggregator = LinearAggregator.fit(training_set, config.aggregator)
    return AggregatedScorer(featurizer, aggregator, scope=config.scope)


def _resolve_edge_type(
    triples: Sequence[Triple], edge_type: EdgeTypeTuple | None
) -> EdgeTypeTuple:
    """Return the one edge type of ``triples``, checked against ``edge_type``."""
    found = {triple.edge_type for triple in triples}
    if len(found) != 1:
        raise ConfigurationError(
            f"train_triples must have exactly one edge type, got {sorted(found)}"
        )
    (only,) = found
    if edge_type is not None and edge_type != only:
        raise ConfigurationError(
            f"train_triples are of {only!r}, not of edge_type {edge_type!r}"
        )
    return only


def _build_featurizer(
    kind: FeaturizerKind, evidence: RuleEvidence, degrees: RelationDegrees
) -> QueryFeaturizer:
    """Return the featurizer ``kind`` names over ``evidence``."""
    match kind:
        case FeaturizerKind.CANDIDATE:
            return CandidateFeaturizer(evidence, degrees)
        case FeaturizerKind.SHARED:
            return SharedFeaturizer(evidence, degrees)
        case FeaturizerKind.JOINT:
            return JointFeaturizer(
                SharedFeaturizer(evidence, degrees),
                CandidateFeaturizer(evidence, degrees),
            )
        case _ as unreachable:
            assert_never(unreachable)


def _known_answers(
    graph: HeteroGraph, edge_type: EdgeTypeTuple, triples: Sequence[Triple]
) -> KnownAnswers:
    """Return the relation's pairs in ``graph`` together with ``triples``."""
    edges = graph.edge_index(edge_type)
    graph_pairs = zip(edges[0].tolist(), edges[1].tolist(), strict=True)
    triple_pairs = ((triple.head_id, triple.tail_id) for triple in triples)
    return KnownAnswers.from_pairs([*graph_pairs, *triple_pairs])
