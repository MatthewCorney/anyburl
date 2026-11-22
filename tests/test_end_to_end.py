"""End-to-end tests: split, learn rules, fit an aggregator, rank test triples."""

from collections.abc import Callable, Sequence

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl import (
    AggregationSetup,
    AnyBURL,
    AnyBURLConfig,
    EdgeTypeTuple,
    EvaluationConfig,
    FeaturizerKind,
    HeteroGraph,
    LinkPredictionEvaluator,
    SplitConfig,
    Triple,
    ValidationSplit,
    split_with_validation,
)
from anyburl.aggregation import CandidateSampling, CandidateScope, LinearAggregator
from anyburl.exceptions import ConfigurationError, NotFittedError

NUM_PERSONS = 30
NUM_CITIES = 5
SPLIT_FRACTION = 0.2
SEED = 7
EXPECTED_MRR = 0.875
EXPECTED_HITS = {1: 0.75, 3: 1.0, 10: 1.0}
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
    scorer = pipeline.fit_aggregator(split.validation_triples, setup=setup)
    return LinkPredictionEvaluator(
        scorer, HeteroGraph(split.train_data), EvaluationConfig()
    )


class TestFitAggregator:
    @pytest.mark.parametrize("setup", SETUPS)
    def test_ranks_test_triples_as_well_as_the_rules_allow(
        self, pipeline: AnyBURL, split: ValidationSplit, setup: AggregationSetup
    ) -> None:
        """Every setup recovers the fixed predictor's ranking on a learnable graph."""
        learned = _evaluator(pipeline, split, setup=setup).evaluate(split.test_triples)

        assert learned.num_queries == 2 * len(split.test_triples)
        assert learned.mrr == pytest.approx(EXPECTED_MRR)
        assert learned.hits_at_k == pytest.approx(EXPECTED_HITS)

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
