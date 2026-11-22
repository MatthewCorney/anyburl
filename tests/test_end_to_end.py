"""End-to-end tests: split, learn rules, fit an aggregator, rank test triples."""

import math
from collections.abc import Callable, Sequence

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl import (
    AggregationSetup,
    AnyBURL,
    AnyBURLConfig,
    CandidateSampling,
    CandidateScope,
    EdgeTypeTuple,
    EvaluationConfig,
    FeaturizerKind,
    HeteroGraph,
    LinearAggregator,
    LinkPredictionEvaluator,
    SplitConfig,
    Triple,
    ValidationSplit,
    ZeroScorer,
    split_with_validation,
)
from anyburl.exceptions import ConfigurationError, NotFittedError

NUM_PERSONS = 30
NUM_CITIES = 5
SPLIT_FRACTION = 0.2
SEED = 7
LIVES_IN: EdgeTypeTuple = ("person", "lives_in", "city")
BORN_IN: EdgeTypeTuple = ("person", "born_in", "city")
CONFIG = AnyBURLConfig(
    sample_size=20,
    min_walk_length=1,
    max_walk_length=2,
    max_walk_attempts=100,
    target_edge_type=LIVES_IN,
    seed=SEED,
)
SETUPS = [
    pytest.param(
        AggregationSetup(featurizer=kind, sampling=CandidateSampling(scope=scope)),
        id=f"{kind}-{scope}",
    )
    for kind in FeaturizerKind
    for scope in CandidateScope
]

TriplesOf = Callable[[ValidationSplit], tuple[Sequence[Triple], EdgeTypeTuple | None]]


def _born_in_triples(split: ValidationSplit) -> Sequence[Triple]:
    edges = split.fit_data[BORN_IN].edge_index
    return [
        Triple(head, tail, "person", "city", "born_in")
        for head, tail in edges.t().tolist()
    ]


@pytest.fixture(scope="module")
def split() -> ValidationSplit:
    """Split a graph where everyone lives in the city they were born in."""
    persons = torch.arange(NUM_PERSONS)
    cities = persons % NUM_CITIES
    ring = torch.arange(NUM_CITIES)
    data = HeteroData()
    data["person"].num_nodes = NUM_PERSONS
    data["city"].num_nodes = NUM_CITIES
    data[BORN_IN].edge_index = torch.stack([persons, cities])
    data[LIVES_IN].edge_index = torch.stack([persons, cities])
    data["city", "near", "city"].edge_index = torch.stack(
        [ring, (ring + 1) % NUM_CITIES]
    )
    return split_with_validation(
        data,
        SplitConfig(target_edge_type=LIVES_IN, test_fraction=SPLIT_FRACTION),
        validation_fraction=SPLIT_FRACTION,
    )


@pytest.fixture(scope="module")
def pipeline(split: ValidationSplit) -> AnyBURL:
    """Learn rules on the fitting graph."""
    return AnyBURL(CONFIG).fit(split.fit_data)


def _evaluator(
    pipeline: AnyBURL, split: ValidationSplit, *, setup: AggregationSetup
) -> LinkPredictionEvaluator:
    scorer = pipeline.fit_aggregator(split.validation_triples, config=setup)
    return LinkPredictionEvaluator(
        scorer, HeteroGraph(split.train_data), EvaluationConfig()
    )


class TestFitAggregator:
    @pytest.mark.parametrize("setup", SETUPS)
    def test_metrics_on_test_triples_are_in_range(
        self, pipeline: AnyBURL, split: ValidationSplit, setup: AggregationSetup
    ) -> None:
        evaluator = _evaluator(pipeline, split, setup=setup)

        metrics = evaluator.evaluate(split.test_triples)

        assert metrics.num_queries == 2 * len(split.test_triples)
        assert math.isfinite(metrics.mrr)
        assert 0.0 < metrics.mrr <= 1.0
        assert all(0.0 <= hits <= 1.0 for hits in metrics.hits_at_k.values())

    @pytest.mark.parametrize("setup", SETUPS)
    def test_ranks_every_query_from_one(
        self, pipeline: AnyBURL, split: ValidationSplit, setup: AggregationSetup
    ) -> None:
        evaluator = _evaluator(pipeline, split, setup=setup)

        ranks = evaluator.rank_queries(split.test_triples)

        assert len(ranks) == 2 * len(split.test_triples)
        assert all(query.rank >= 1.0 for query in ranks)

    @pytest.mark.parametrize("setup", SETUPS)
    def test_does_at_least_as_well_as_zero_scorer(
        self, pipeline: AnyBURL, split: ValidationSplit, setup: AggregationSetup
    ) -> None:
        known = HeteroGraph(split.train_data)
        zero = LinkPredictionEvaluator(
            ZeroScorer(known, LIVES_IN), known, EvaluationConfig()
        )

        learned = _evaluator(pipeline, split, setup=setup).evaluate(split.test_triples)

        assert learned.mrr >= zero.evaluate(split.test_triples).mrr

    def test_same_seed_gives_same_rules_and_weights(
        self, split: ValidationSplit
    ) -> None:
        first, second = (AnyBURL(CONFIG).fit(split.fit_data) for _ in range(2))

        models = [
            pipeline.fit_aggregator(split.validation_triples).aggregator
            for pipeline in (first, second)
        ]

        assert first.results
        assert first.results == second.results
        assert all(isinstance(model, LinearAggregator) for model in models)
        assert torch.equal(models[0].weights, models[1].weights)

    def test_scores_are_named_by_feature(
        self, pipeline: AnyBURL, split: ValidationSplit
    ) -> None:
        scorer = pipeline.fit_aggregator(split.validation_triples)

        assert len(scorer.feature_names()) == scorer.aggregator.num_features

    def test_raises_before_fit(self, split: ValidationSplit) -> None:
        with pytest.raises(NotFittedError):
            AnyBURL(CONFIG).fit_aggregator(split.validation_triples)

    @pytest.mark.parametrize(
        ("triples_of", "match"),
        [
            pytest.param(
                lambda split: (_born_in_triples(split), None),
                "no rules predict",
                id="relation-without-rules",
            ),
            pytest.param(
                lambda split: (split.validation_triples, BORN_IN),
                "not of edge_type",
                id="edge-type-mismatch",
            ),
            pytest.param(lambda _: ((), None), "exactly one", id="no-triples"),
            pytest.param(
                lambda split: (
                    [*split.validation_triples, *_born_in_triples(split)],
                    None,
                ),
                "exactly one",
                id="mixed-edge-types",
            ),
        ],
    )
    def test_rejects_triples_it_cannot_fit(
        self,
        pipeline: AnyBURL,
        split: ValidationSplit,
        triples_of: TriplesOf,
        match: str,
    ) -> None:
        triples, edge_type = triples_of(split)

        with pytest.raises(ConfigurationError, match=match):
            pipeline.fit_aggregator(triples, edge_type=edge_type)
