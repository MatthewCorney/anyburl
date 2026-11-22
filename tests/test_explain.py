"""Tests for prediction provenance: explain, top_tails and RuleFiring."""

import pytest

from anyburl.exceptions import ConfigurationError
from anyburl.graph import HeteroGraph
from anyburl.prediction import RuleFiring, RulePredictor
from anyburl.rule import Rule, RuleMetrics, RuleType
from anyburl.rule_evaluation import RuleSet
from tests.rules import multi_grounding_rule_set


@pytest.fixture
def predictor(multi_grounding_graph: HeteroGraph) -> RulePredictor:
    return RulePredictor(multi_grounding_graph, multi_grounding_rule_set())


class TestExplain:
    def test_names_the_firing_chain(self, predictor: RulePredictor) -> None:
        """A score with no provenance cannot be checked; explain must name the body."""
        firings = predictor.explain(0, 2)

        assert len(firings) == 1
        assert firings[0].chain == (
            ("person", "born_in", "city"),
            ("city", "near", "city"),
        )
        assert firings[0].rule_type is RuleType.CYCLIC
        assert firings[0].confidence == pytest.approx(0.5)

    def test_reports_grounding_counts(self, predictor: RulePredictor) -> None:
        """City 2 is reached by two body groundings, city 3 by one."""
        assert predictor.explain(0, 2)[0].groundings == 2
        assert predictor.explain(0, 3)[0].groundings == 1

    def test_returns_nothing_for_unreached_pairs(
        self, predictor: RulePredictor
    ) -> None:
        assert predictor.explain(0, 0) == ()

    def test_contributions_sum_to_the_score(
        self, evaluator_graph: HeteroGraph, cyclic_rule: Rule
    ) -> None:
        """Independent firings must compose by noisy-or into the reported score."""
        results = RuleSet.of([(cyclic_rule, RuleMetrics(1, 0.5, 0.33, 3))])
        predictor = RulePredictor(evaluator_graph, results)

        scores = predictor.score_tails(0)
        for city_id in range(evaluator_graph.node_count("city")):
            firings = predictor.explain(0, city_id)
            complement = 1.0
            for firing in firings:
                complement *= 1.0 - firing.contribution
            assert 1.0 - complement == pytest.approx(scores[city_id].item(), abs=1e-6)


class TestTopTails:
    def test_is_sorted_and_carries_explanations(self, predictor: RulePredictor) -> None:
        predictions = predictor.top_tails(0, limit=5)

        assert [p.tail_id for p in predictions] == [2, 3]
        assert [p.score for p in predictions] == sorted(
            (p.score for p in predictions), reverse=True
        )
        assert all(p.explanations for p in predictions)

    def test_respects_the_limit(self, predictor: RulePredictor) -> None:
        assert len(predictor.top_tails(0, limit=1)) == 1

    def test_rejects_a_non_positive_limit(self, predictor: RulePredictor) -> None:
        with pytest.raises(ConfigurationError, match="limit must be positive"):
            predictor.top_tails(0, limit=0)


class TestRuleFiringDescribe:
    def test_is_readable(self, predictor: RulePredictor) -> None:
        described = predictor.explain(0, 2)[0].describe()

        assert "person_born_in_city -> city_near_city" in described
        assert "groundings=2" in described

    def test_disambiguates_shared_relation_names(self) -> None:
        """Relation names shared across edge types stay distinguishable."""
        firing = RuleFiring(
            chain=(("author", "to", "paper"), ("paper", "to", "author")),
            rule_type=RuleType.CYCLIC,
            confidence=0.5,
            groundings=3,
            contribution=0.5,
        )

        assert "author_to_paper -> paper_to_author" in firing.describe()
