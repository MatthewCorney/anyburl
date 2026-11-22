"""Tests for RulePredictor: predictions, scores and scoring configuration."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.core import EdgeTypeTuple, QueryDirection
from anyburl.exceptions import ConfigurationError, GraphSchemaError
from anyburl.graph import HeteroGraph
from anyburl.prediction import (
    ConfidenceAggregation,
    GroundingWeighting,
    PredictionConfig,
    RulePredictor,
)
from anyburl.prediction.grounding import body_chain_key, build_groundings
from anyburl.prediction.scoring import TIE_BREAK_WEIGHT
from anyburl.rule import Atom, Rule, RuleMetrics, RuleType, Term
from anyburl.rule_evaluation import RuleSet
from tests import adjacency
from tests.rules import (
    cyclic_and_grounded_rule_set,
    cyclic_born_in_near,
    metrics_with,
    multi_grounding_rule_set,
    object_grounded,
    subject_grounded,
)


def _works_in_rule() -> Rule:
    """Return ``lives_in(X, Y) :- works_in(X, Y)``."""
    return Rule(
        head=cyclic_born_in_near().head,
        body=(
            Atom(
                relation="works_in",
                subject=Term.variable("X", node_type="person"),
                object_=Term.variable("Y", node_type="city"),
            ),
        ),
        rule_type=RuleType.CYCLIC,
    )


@pytest.fixture
def two_chain_graph() -> HeteroGraph:
    """Person 0 reaches city 1 via ``born_in, near`` and city 2 via ``works_in``."""
    data = HeteroData()
    data["person"].num_nodes = 1
    data["city"].num_nodes = 3
    data["person", "born_in", "city"].edge_index = torch.tensor([[0], [0]])
    data["city", "near", "city"].edge_index = torch.tensor([[0], [1]])
    data["person", "works_in", "city"].edge_index = torch.tensor([[0], [2]])
    data["person", "lives_in", "city"].edge_index = torch.tensor([[0], [0]])
    return HeteroGraph(data)


class TestRulePredictorConstruction:
    def test_empty_results_raises(self, evaluator_graph: HeteroGraph) -> None:
        with pytest.raises(ConfigurationError, match="results must not be empty"):
            RulePredictor(evaluator_graph, RuleSet.of([]))

    def test_rejects_an_edge_type_missing_from_the_graph(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        person = Term.variable("X", node_type="person")
        city = Term.variable("Y", node_type="city")
        rule = Rule(
            head=Atom(relation="lives_in", subject=person, object_=city),
            body=(Atom(relation="visited", subject=person, object_=city),),
            rule_type=RuleType.CYCLIC,
        )

        with pytest.raises(GraphSchemaError):
            RulePredictor(
                evaluator_graph, RuleSet.of([(rule, RuleMetrics(1, 0.5, 0.5, 2))])
            )


class TestPredict:
    def test_cyclic_predictions_contain_expected_pairs(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        """born_in @ near produces exactly {(0,1),(1,2),(2,1),(3,0)}."""
        results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.25))])
        predictor = RulePredictor(evaluator_graph, results)

        predictions = predictor.predict()

        pair_ids = {(p.head_id, p.tail_id) for p in predictions}
        assert pair_ids == {(0, 1), (1, 2), (2, 1), (3, 0)}

    def test_predictions_sorted_by_score(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.25))])
        predictor = RulePredictor(evaluator_graph, results)

        predictions = predictor.predict()

        scores = [p.score for p in predictions]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize(
        ("rule", "expected_pair"),
        [(subject_grounded(0), (0, 1)), (object_grounded(0), (3, 0))],
        ids=["subject_grounded", "object_grounded"],
    )
    def test_ac1_rule_predicts_via_its_body(
        self,
        evaluator_graph: HeteroGraph,
        rule: Rule,
        expected_pair: tuple[int, int],
    ) -> None:
        """An AC1 rule grounded on either side predicts via born_in @ near."""
        results = RuleSet.of([(rule, metrics_with(confidence=0.5))])
        predictor = RulePredictor(evaluator_graph, results)

        predictions = predictor.predict()

        pair_ids = {(p.head_id, p.tail_id) for p in predictions}
        assert expected_pair in pair_ids

    def test_two_rules_same_pair_noisy_or(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        """Two rules both predicting (3,0) with conf 0.25 and 0.5."""
        results = RuleSet.of(
            [
                (cyclic_rule, metrics_with(confidence=0.25)),
                (object_grounded(0), metrics_with(confidence=0.5)),
            ]
        )
        predictor = RulePredictor(evaluator_graph, results)

        predictions = predictor.predict()

        pair_30 = next(p for p in predictions if p.head_id == 3 and p.tail_id == 0)
        assert pair_30.score == pytest.approx(0.625)

    @pytest.mark.parametrize(
        ("filter_known", "is_kept"),
        [(True, False), (False, True)],
        ids=["filtered", "unfiltered"],
    )
    def test_known_edge_is_kept_only_without_filter(
        self,
        evaluator_graph: HeteroGraph,
        cyclic_rule: Rule,
        filter_known: bool,
        is_kept: bool,
    ) -> None:
        """(3,0) is a known lives_in edge."""
        results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.25))])
        predictor = RulePredictor(evaluator_graph, results)

        predictions = predictor.predict(filter_known=filter_known)

        pair_ids = {(p.head_id, p.tail_id) for p in predictions}
        assert ((3, 0) in pair_ids) is is_kept

    def test_predict_leaves_explanations_empty(
        self, multi_grounding_graph: HeteroGraph
    ) -> None:
        """Attaching provenance to every pair would dwarf the scores."""
        predictor = RulePredictor(multi_grounding_graph, multi_grounding_rule_set())

        assert all(p.explanations == () for p in predictor.predict())


class TestScoreQueries:
    def test_score_tails_cyclic(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        """score_tails for person 0 should score city 1."""
        results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.5))])
        predictor = RulePredictor(evaluator_graph, results)

        scores = predictor.score_tails(0)

        assert scores.shape == (3,)
        assert scores[1].item() > 0
        assert scores[0].item() == pytest.approx(0.0)
        assert scores[2].item() == pytest.approx(0.0)

    def test_score_tails_with_ac1(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        """Subject-grounded AC1 for person 0 also contributes to score_tails(0)."""
        results = RuleSet.of(
            [
                (cyclic_rule, metrics_with(confidence=0.25)),
                (subject_grounded(0), metrics_with(confidence=0.5)),
            ]
        )
        predictor = RulePredictor(evaluator_graph, results)

        scores = predictor.score_tails(0)

        assert scores[1].item() == pytest.approx(0.625)

    def test_score_heads_cyclic(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        """score_heads for city 0 should score person 3."""
        results = RuleSet.of([(cyclic_rule, metrics_with(confidence=0.5))])
        predictor = RulePredictor(evaluator_graph, results)

        scores = predictor.score_heads(0)

        assert scores.shape == (4,)
        assert scores[3].item() > 0


def _weighted(graph: HeteroGraph, weighting: GroundingWeighting) -> RulePredictor:
    return RulePredictor(
        graph,
        multi_grounding_rule_set(),
        config=PredictionConfig(grounding_weighting=weighting),
    )


class TestGroundingWeighting:
    def test_once_ignores_grounding_count(
        self, multi_grounding_graph: HeteroGraph
    ) -> None:
        """Under ONCE weighting a candidate is only reachable or not."""
        scores = _weighted(multi_grounding_graph, GroundingWeighting.ONCE).score_tails(
            0
        )

        assert scores[2].item() == pytest.approx(scores[3].item())

    @pytest.mark.parametrize(
        "config",
        [
            PredictionConfig(grounding_weighting=GroundingWeighting.LOG_COUNT),
            PredictionConfig(),
        ],
        ids=["log_count", "default"],
    )
    def test_more_groundings_rank_higher(
        self, multi_grounding_graph: HeteroGraph, config: PredictionConfig
    ) -> None:
        """Two body groundings must outrank one; path multiplicity must survive."""
        predictor = RulePredictor(
            multi_grounding_graph, multi_grounding_rule_set(), config=config
        )

        scores = predictor.score_tails(0)

        assert scores[2].item() > scores[3].item()

    def test_log_count_matches_once_for_single_grounding(
        self, multi_grounding_graph: HeteroGraph
    ) -> None:
        """log2(1 + 1) == 1, so one grounding weighs exactly one firing."""
        weighted = _weighted(multi_grounding_graph, GroundingWeighting.LOG_COUNT)
        plain = _weighted(multi_grounding_graph, GroundingWeighting.ONCE)

        assert weighted.score_tails(0)[3].item() == pytest.approx(
            plain.score_tails(0)[3].item()
        )

    def test_score_heads_is_path_weighted_too(
        self, multi_grounding_graph: HeteroGraph
    ) -> None:
        """The reverse direction must weight groundings the same way."""
        predictor = _weighted(multi_grounding_graph, GroundingWeighting.LOG_COUNT)
        plain = _weighted(multi_grounding_graph, GroundingWeighting.ONCE)

        assert predictor.score_heads(2)[0].item() > plain.score_heads(2)[0].item()


def _product(graph: HeteroGraph, chain: tuple[EdgeTypeTuple, ...]) -> torch.Tensor:
    """Return the dense product of a chain's adjacency matrices."""
    product = adjacency.adjacency_matrix(graph, chain[0]).to_dense()
    for edge_type in chain[1:]:
        product = product @ adjacency.adjacency_matrix(graph, edge_type).to_dense()
    return product


def _dense(entries: tuple[torch.Tensor, torch.Tensor], size: int) -> torch.Tensor:
    """Scatter ``(indices, counts)`` into a dense vector."""
    dense = torch.zeros(size)
    dense[entries[0]] = entries[1]
    return dense


class TestBuildGroundings:
    @pytest.mark.parametrize(
        "rule", [cyclic_born_in_near(), subject_grounded(0)], ids=["cyclic", "ac1"]
    )
    def test_matches_the_chain_product(
        self, multi_grounding_graph: HeteroGraph, rule: Rule
    ) -> None:
        """Each row and column is the chain product's, path counts included."""
        chain = body_chain_key(rule)
        grounding = build_groundings(
            multi_grounding_graph, RuleSet.of([(rule, RuleMetrics(2, 0.5, 0.5, 4))])
        )[chain]
        product = _product(multi_grounding_graph, chain)

        for head_id in range(product.size(0)):
            row = grounding.row(head_id)
            assert torch.equal(_dense(row, product.size(1)), product[head_id])
            assert torch.equal(row[0], torch.sort(row[0]).values)
        for tail_id in range(product.size(1)):
            column = grounding.column(tail_id)
            assert torch.equal(_dense(column, product.size(0)), product[:, tail_id])


class TestMaximumAggregation:
    def test_scores_strongest_chain_with_noisy_or_tie_break(
        self, evaluator_graph: HeteroGraph
    ) -> None:
        """MAXIMUM is the best chain's contribution plus a scaled noisy-or."""
        predictor = RulePredictor(
            evaluator_graph,
            cyclic_and_grounded_rule_set(),
            config=PredictionConfig(aggregation=ConfidenceAggregation.MAXIMUM),
        )

        for direction, query_ids, score in (
            (QueryDirection.TAIL, range(4), predictor.score_tails),
            (QueryDirection.HEAD, range(3), predictor.score_heads),
        ):
            for query_id in query_ids:
                contributions = predictor.evidence.chain_evidence(
                    direction, query_id
                ).contributions
                noisy_or = 1.0 - torch.prod(1.0 - contributions, dim=0)
                expected = contributions.max(dim=0).values + TIE_BREAK_WEIGHT * noisy_or
                assert torch.allclose(expected, score(query_id), atol=1e-6)

    def test_keeps_one_strong_chain_above_many_weak_ones(
        self, two_chain_graph: HeteroGraph
    ) -> None:
        """Weak chains that noisy-or past a strong one must not outrank it."""
        results = RuleSet.of(
            [
                (cyclic_born_in_near(), metrics_with(confidence=0.4)),
                (object_grounded(1), metrics_with(confidence=0.4)),
                (_works_in_rule(), metrics_with(confidence=0.6)),
            ]
        )
        noisy_or = RulePredictor(
            two_chain_graph,
            results,
            config=PredictionConfig(grounding_weighting=GroundingWeighting.ONCE),
        ).score_tails(0)
        maximum = RulePredictor(
            two_chain_graph,
            results,
            config=PredictionConfig(
                grounding_weighting=GroundingWeighting.ONCE,
                aggregation=ConfidenceAggregation.MAXIMUM,
            ),
        ).score_tails(0)

        assert noisy_or[1] > noisy_or[2]
        assert maximum[2] > maximum[1]


class TestUnseenNegatives:
    def test_negative_unseen_negatives_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="unseen_negatives"):
            PredictionConfig(unseen_negatives=-1)

    def test_unseen_negatives_reorder_candidates(
        self, two_chain_graph: HeteroGraph
    ) -> None:
        """A perfect rule with two predictions falls below a broad 0.9 rule."""
        results = RuleSet.of(
            [
                (cyclic_born_in_near(), RuleMetrics(2, 1.0, 0.1, 2)),
                (_works_in_rule(), RuleMetrics(90, 0.9, 0.1, 100)),
            ]
        )

        def tail_scores(unseen_negatives: int) -> torch.Tensor:
            config = PredictionConfig(
                grounding_weighting=GroundingWeighting.ONCE,
                aggregation=ConfidenceAggregation.MAXIMUM,
                unseen_negatives=unseen_negatives,
            )
            return RulePredictor(two_chain_graph, results, config=config).score_tails(0)

        raw, corrected = tail_scores(0), tail_scores(5)

        assert raw[1] > raw[2]
        assert corrected[2] > corrected[1]
