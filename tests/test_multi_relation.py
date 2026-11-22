"""Tests for rule sets, prediction and evaluation spanning several relations."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl import (
    AnyBURL,
    AnyBURLConfig,
    EvaluationConfig,
    LinkPredictionEvaluator,
    MultiRelationPredictor,
    RulePredictor,
    RuleSet,
)
from anyburl.baselines import ZeroScorer
from anyburl.core import Triple
from anyburl.evaluation import rank_queries_by_edge_type
from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.rule import Atom, Rule, RuleType, Term
from tests.rules import metrics_with

LIVES_IN = ("person", "lives_in", "city")
KNOWS = ("person", "knows", "person")
NEAR = ("city", "near", "city")


def _cyclic(head: tuple[str, str, str], first: tuple[str, str, str]) -> Rule:
    """Return ``head(X, Y) :- first(X, Z0), head(Z0, Y)``."""
    x = Term.variable("X", node_type=head[0])
    y = Term.variable("Y", node_type=head[2])
    z = Term.variable("Z0", node_type=first[2])
    return Rule(
        head=Atom(relation=head[1], subject=x, object_=y),
        body=(
            Atom(relation=first[1], subject=x, object_=z),
            Atom(relation=head[1], subject=z, object_=y),
        ),
        rule_type=RuleType.CYCLIC,
    )


LIVES_IN_RULE = _cyclic(LIVES_IN, KNOWS)
NEAR_RULE = _cyclic(NEAR, NEAR)


@pytest.fixture
def rule_set() -> RuleSet:
    return RuleSet.of(
        [(LIVES_IN_RULE, metrics_with(0.5)), (NEAR_RULE, metrics_with(0.4))]
    )


def _triple(edge_type: tuple[str, str, str], head: int, tail: int) -> Triple:
    return Triple(
        head_id=head,
        tail_id=tail,
        head_type=edge_type[0],
        tail_type=edge_type[2],
        relation=edge_type[1],
    )


class TestRuleSet:
    def test_head_edge_types_in_learned_order(self, rule_set: RuleSet) -> None:
        assert rule_set.head_edge_types() == (LIVES_IN, NEAR)

    def test_for_edge_type_selects_one_relation(self, rule_set: RuleSet) -> None:
        assert rule_set.for_edge_type(NEAR) == RuleSet.of(
            [(NEAR_RULE, metrics_with(0.4))]
        )
        assert rule_set.for_edge_type(KNOWS) == RuleSet.of(())


class TestZeroScorer:
    def test_scores_every_candidate_zero(self, simple_graph: HeteroGraph) -> None:
        scorer = ZeroScorer(simple_graph, LIVES_IN)

        assert torch.equal(scorer.score_tails(0), torch.zeros(3))
        assert torch.equal(scorer.score_heads(0), torch.zeros(5))


class TestMultiRelationPredictor:
    def test_matches_single_relation_predictor(
        self, simple_graph: HeteroGraph, rule_set: RuleSet
    ) -> None:
        multi = MultiRelationPredictor(simple_graph, rule_set)
        single = RulePredictor(simple_graph, rule_set.for_edge_type(LIVES_IN))

        scorer = multi.scorer_for(LIVES_IN)

        for head_id in range(5):
            assert torch.equal(scorer.score_tails(head_id), single.score_tails(head_id))

    def test_relation_without_rules_scores_zero(
        self, simple_graph: HeteroGraph, rule_set: RuleSet
    ) -> None:
        scorer = MultiRelationPredictor(simple_graph, rule_set).scorer_for(KNOWS)

        assert torch.equal(scorer.score_tails(0), torch.zeros(5))

    def test_predictor_for_relation_without_rules_raises(
        self, simple_graph: HeteroGraph, rule_set: RuleSet
    ) -> None:
        with pytest.raises(ConfigurationError, match="no rules"):
            MultiRelationPredictor(simple_graph, rule_set).predictor_for(KNOWS)

    def test_predict_covers_every_relation_sorted_by_score(
        self, simple_graph: HeteroGraph, rule_set: RuleSet
    ) -> None:
        predictions = MultiRelationPredictor(simple_graph, rule_set).predict()

        assert {p.relation for p in predictions} == {"lives_in", "near"}
        scores = [p.score for p in predictions]
        assert scores == sorted(scores, reverse=True)


class TestRankQueriesByEdgeType:
    def test_ranks_each_relation_with_its_own_scorer(
        self, simple_graph: HeteroGraph, rule_set: RuleSet
    ) -> None:
        multi = MultiRelationPredictor(simple_graph, rule_set)
        config = EvaluationConfig()
        lives_in = [_triple(LIVES_IN, 1, 2), _triple(LIVES_IN, 3, 1)]
        near = [_triple(NEAR, 0, 2)]

        ranks = rank_queries_by_edge_type(
            multi.scorer_for, simple_graph, [lives_in[0], *near, lives_in[1]], config
        )

        expected = (
            *LinkPredictionEvaluator(
                multi.scorer_for(LIVES_IN), simple_graph, config
            ).rank_queries(lives_in),
            *LinkPredictionEvaluator(
                multi.scorer_for(NEAR), simple_graph, config
            ).rank_queries(near),
        )
        assert ranks == expected


class TestPipelineAcrossRelations:
    @pytest.fixture
    def pipeline(self, simple_hetero_data: HeteroData) -> AnyBURL:
        config = AnyBURLConfig(
            sample_size=40, min_walk_length=1, max_walk_length=3, max_walk_attempts=200
        )
        return AnyBURL(config).fit(simple_hetero_data)

    def test_evaluates_relations_without_rules(self, pipeline: AnyBURL) -> None:
        triples = [_triple(LIVES_IN, 1, 2), _triple(KNOWS, 0, 2)]

        metrics = pipeline.evaluate_predictions(triples)

        assert metrics.num_queries == 4

    def test_known_defaults_to_the_fitted_graph(self, pipeline: AnyBURL) -> None:
        triples = [_triple(LIVES_IN, 1, 2)]

        assert pipeline.evaluate_predictions(triples) == (
            pipeline.evaluate_predictions(triples, known=pipeline.graph)
        )

    def test_known_edges_are_filtered(
        self, pipeline: AnyBURL, simple_hetero_data: HeteroData
    ) -> None:
        """Also knowing lives_in(1, 0) removes a competitor of the target (1, 2)."""
        triples = [_triple(LIVES_IN, 1, 2)]
        known = simple_hetero_data.clone()
        known[LIVES_IN].edge_index = torch.cat(
            [known[LIVES_IN].edge_index, torch.tensor([[1], [0]])], dim=1
        )

        default = pipeline.evaluate_predictions(triples)
        with_extra = pipeline.evaluate_predictions(triples, known=known)

        assert with_extra.mrr > default.mrr
