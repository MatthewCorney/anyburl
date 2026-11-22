"""Tests for RuleEvidence: per-chain evidence, memoised firings, contributions."""

import math

import pytest
import torch

from anyburl.core import QueryDirection
from anyburl.graph import HeteroGraph
from anyburl.prediction import (
    GroundingWeighting,
    PredictionConfig,
    RuleEvidence,
    RulePredictor,
)
from anyburl.prediction.scoring import GROUNDING_WEIGHT_OFFSET
from anyburl.rule import Rule, RuleMetrics, RuleType
from anyburl.rule_evaluation import RuleSet
from tests.rules import (
    ac2_born_in,
    cyclic_and_grounded_rule_set,
    cyclic_born_in_near,
    metrics_with,
    multi_grounding_rule_set,
    object_grounded,
    subject_grounded,
)

_NUM_CITIES = 3
"""Query ids valid in both directions of ``evaluator_graph``."""

_NUM_QUERIES = {QueryDirection.TAIL: 4, QueryDirection.HEAD: _NUM_CITIES}


class TestChainEvidence:
    @pytest.mark.parametrize("grounding_weighting", list(GroundingWeighting))
    def test_composes_into_the_score(
        self,
        evaluator_graph: HeteroGraph,
        grounding_weighting: GroundingWeighting,
    ) -> None:
        """Per-chain contributions must noisy-or back into the predictor's score."""
        predictor = RulePredictor(
            evaluator_graph,
            cyclic_and_grounded_rule_set(),
            config=PredictionConfig(grounding_weighting=grounding_weighting),
        )

        for direction, query_ids, score in (
            (QueryDirection.TAIL, range(4), predictor.score_tails),
            (QueryDirection.HEAD, range(3), predictor.score_heads),
        ):
            for query_id in query_ids:
                evidence = predictor.evidence.chain_evidence(direction, query_id)
                composed = 1.0 - torch.prod(1.0 - evidence.contributions, dim=0)
                assert torch.allclose(composed, score(query_id), atol=1e-6)

    def test_rows_follow_chains(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        results = RuleSet.of(
            [
                (cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3)),
                (subject_grounded(0), RuleMetrics(1, 0.7, 0.33, 3)),
            ]
        )
        predictor = RulePredictor(evaluator_graph, results)

        evidence = predictor.evidence.chain_evidence(QueryDirection.TAIL, 0)

        assert [rule_type for _, rule_type in predictor.evidence.chains] == [
            RuleType.CYCLIC,
            RuleType.AC1,
        ]
        assert evidence.contributions.shape == (2, evaluator_graph.node_count("city"))

    def test_reports_grounding_counts(self, multi_grounding_graph: HeteroGraph) -> None:
        """City 2 is reached by two body groundings, city 3 by one, city 0 by none."""
        predictor = RulePredictor(multi_grounding_graph, multi_grounding_rule_set())

        groundings = predictor.evidence.chain_evidence(
            QueryDirection.TAIL, 0
        ).groundings[0]

        assert groundings[2].item() == 2
        assert groundings[3].item() == 1
        assert groundings[0].item() == 0

    def test_is_empty_without_scoring_rules(self, evaluator_graph: HeteroGraph) -> None:
        """AC2 rules are ignored, which can leave no chains at all."""
        predictor = RulePredictor(
            evaluator_graph, RuleSet.of([(ac2_born_in(), metrics_with(0.5))])
        )

        evidence = predictor.evidence.chain_evidence(QueryDirection.TAIL, 0)

        assert evidence.contributions.shape == (0, evaluator_graph.node_count("city"))


def _memo_results() -> RuleSet:
    """A cyclic rule and a subject-grounded AC1 rule over the same chain."""
    return RuleSet.of(
        [
            (cyclic_born_in_near(), metrics_with(confidence=0.25)),
            (subject_grounded(0), metrics_with(confidence=0.5)),
        ]
    )


class TestRuleEvidenceMemo:
    @pytest.fixture
    def evidence(self, evaluator_graph: HeteroGraph) -> RuleEvidence:
        return RulePredictor(evaluator_graph, _memo_results()).evidence

    def test_reached_candidates_unites_chain_and_acyclic_evidence(
        self, evidence: RuleEvidence
    ) -> None:
        for direction in QueryDirection:
            for query_id in range(_NUM_CITIES):
                acyclic_ids, _ = evidence.acyclic_firings(direction, query_id)
                chains = evidence.chain_evidence(direction, query_id)
                chain_ids = torch.nonzero(chains.contributions.sum(dim=0) > 0)
                expected = torch.unique(torch.cat([chain_ids.flatten(), acyclic_ids]))

                reached = evidence.reached_candidates(direction, query_id)

                assert reached.tolist() == expected.tolist()

    def test_repeated_query_reuses_its_firings(self, evidence: RuleEvidence) -> None:
        first = evidence.acyclic_firings(QueryDirection.TAIL, 0)

        assert evidence.acyclic_firings(QueryDirection.TAIL, 0) is first

    def test_alternating_queries_match_fresh_evidence(
        self, evaluator_graph: HeteroGraph, evidence: RuleEvidence
    ) -> None:
        queries = [
            (direction, query_id)
            for query_id in range(_NUM_CITIES)
            for direction in QueryDirection
        ]

        for direction, query_id in queries * 2:
            fresh = RulePredictor(evaluator_graph, _memo_results()).evidence
            reached = evidence.reached_candidates(direction, query_id)
            ids, confidences = evidence.acyclic_firings(direction, query_id)
            fresh_ids, fresh_confidences = fresh.acyclic_firings(direction, query_id)

            assert (
                reached.tolist()
                == fresh.reached_candidates(direction, query_id).tolist()
            )
            assert ids.tolist() == fresh_ids.tolist()
            assert confidences.tolist() == fresh_confidences.tolist()


def _batch_results() -> RuleSet:
    return RuleSet.of(
        [
            (cyclic_born_in_near(), metrics_with(confidence=0.25)),
            (subject_grounded(0), metrics_with(confidence=0.5)),
            (object_grounded(1), metrics_with(confidence=0.4)),
        ]
    )


def _per_chain_contributions(
    evidence: RuleEvidence, direction: QueryDirection, query_id: int
) -> list[tuple[int, float, float]]:
    expected: list[tuple[int, float, float]] = []
    for side in evidence.sides(direction):
        candidates, counts = side.fetch(query_id)
        query_confidence = side.confidence_for(query_id) or 0.0
        for candidate, count in zip(candidates.tolist(), counts.tolist(), strict=True):
            weight = math.log2(GROUNDING_WEIGHT_OFFSET + count)
            grounded = side.candidate_confidences.get(candidate, 0.0)
            complement = (1.0 - query_confidence) ** weight * (1.0 - grounded) ** weight
            expected.append((candidate, 1.0 - complement, float(count)))
    return expected


class TestChainContributions:
    @pytest.fixture
    def evidence(self, evaluator_graph: HeteroGraph) -> RuleEvidence:
        config = PredictionConfig(grounding_weighting=GroundingWeighting.LOG_COUNT)
        return RulePredictor(evaluator_graph, _batch_results(), config=config).evidence

    @pytest.mark.parametrize("direction", list(QueryDirection))
    def test_matches_each_chain_computed_alone(
        self, evidence: RuleEvidence, direction: QueryDirection
    ) -> None:
        for query_id in range(_NUM_QUERIES[direction]):
            found = evidence.chain_contributions(direction, query_id)
            expected = _per_chain_contributions(evidence, direction, query_id)

            assert found.candidates.tolist() == [c for c, _, _ in expected]
            assert found.contributions.tolist() == pytest.approx(
                [c for _, c, _ in expected], abs=1e-6
            )
            assert found.groundings.tolist() == [g for _, _, g in expected]
