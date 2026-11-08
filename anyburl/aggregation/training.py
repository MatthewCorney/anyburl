"""Assemble ranking queries and sampled candidates for fitting an aggregator.

Each query keeps its correct answer, its strongest rule-scored candidates
(the hard negatives) and a random sample of the rest. Other known answers
are dropped, as in the filtered evaluation setting.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import assert_never

import torch
from torch import Tensor

from ..core import QueryDirection, Triple
from ..exceptions import ConfigurationError
from .features import QueryFeaturizer

__all__ = [
    "CandidateSampling",
    "CandidateScope",
    "KnownAnswers",
    "RankingQuery",
    "TrainingSet",
    "build_training_set",
    "concat_training_sets",
    "ranking_queries",
]

DEFAULT_MAX_RULE_CANDIDATES: int = 500
"""Highest rule-scored negatives kept per query when none is given."""

DEFAULT_NUM_RANDOM_CANDIDATES: int = 100
"""Uniformly sampled negatives kept per query when none is given."""

DEFAULT_SAMPLING_SEED: int = 0
"""Seed for the random negatives when none is given."""

TARGET_POSITION: int = 0
"""Row of each query's correct answer in :attr:`TrainingSet.features`."""


class CandidateScope(StrEnum):
    """Which candidates a learned aggregator ranks.

    Attributes
    ----------
    ALL : str
        Every entity of the candidate type.
    REACHED : str
        Only candidates some rule scores positively; the rest rank below
        them, tied. Queries whose answer no rule reaches teach nothing and
        are left out of training.
    """

    ALL = auto()
    REACHED = auto()


@dataclass(frozen=True, slots=True)
class RankingQuery:
    """One link prediction query with its correct answer.

    Parameters
    ----------
    direction : QueryDirection
        ``TAIL`` ranks tails of head ``query_id``; ``HEAD`` ranks heads of
        tail ``query_id``.
    query_id : int
        The entity being queried.
    target_id : int
        The correct answer.
    """

    direction: QueryDirection
    query_id: int
    target_id: int


def ranking_queries(triples: Iterable[Triple]) -> tuple[RankingQuery, ...]:
    """Return a tail query then a head query for every triple.

    Parameters
    ----------
    triples : Iterable[Triple]
        Held-out triples of one relation.

    Returns
    -------
    tuple[RankingQuery, ...]
        Two queries per triple.
    """
    return tuple(
        query
        for triple in triples
        for query in (
            RankingQuery(QueryDirection.TAIL, triple.head_id, triple.tail_id),
            RankingQuery(QueryDirection.HEAD, triple.tail_id, triple.head_id),
        )
    )


@dataclass(frozen=True, slots=True)
class KnownAnswers:
    """True answers of each query, excluded from its negatives.

    Parameters
    ----------
    tails_of_head : Mapping[int, frozenset[int]]
        Known tails of each head.
    heads_of_tail : Mapping[int, frozenset[int]]
        Known heads of each tail.
    """

    tails_of_head: Mapping[int, frozenset[int]]
    heads_of_tail: Mapping[int, frozenset[int]]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, int]]) -> "KnownAnswers":
        """Index ``(head, tail)`` pairs in both directions.

        Parameters
        ----------
        pairs : Iterable[tuple[int, int]]
            True ``(head_id, tail_id)`` pairs of the target relation.

        Returns
        -------
        KnownAnswers
            The pairs, indexed by head and by tail.
        """
        tails: dict[int, set[int]] = defaultdict(set)
        heads: dict[int, set[int]] = defaultdict(set)
        for head_id, tail_id in pairs:
            tails[head_id].add(tail_id)
            heads[tail_id].add(head_id)
        return cls(
            tails_of_head={key: frozenset(found) for key, found in tails.items()},
            heads_of_tail={key: frozenset(found) for key, found in heads.items()},
        )

    def of(self, query: RankingQuery) -> frozenset[int]:
        """Return every known answer to ``query``, its target included."""
        match query.direction:
            case QueryDirection.TAIL:
                return self.tails_of_head.get(query.query_id, frozenset())
            case QueryDirection.HEAD:
                return self.heads_of_tail.get(query.query_id, frozenset())
            case _ as unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class CandidateSampling:
    """Which candidates represent each query during fitting.

    Parameters
    ----------
    max_rule_candidates : int
        Highest rule-scored negatives kept per query.
    num_random_candidates : int
        Further negatives drawn uniformly from the remaining candidates.
    seed : int
        Seed for the uniform draw.
    scope : CandidateScope
        Which candidates negatives come from. Match the scope the fitted
        model will rank with.

    Raises
    ------
    ConfigurationError
        If either count is negative, or both are zero.
    """

    max_rule_candidates: int = DEFAULT_MAX_RULE_CANDIDATES
    num_random_candidates: int = DEFAULT_NUM_RANDOM_CANDIDATES
    seed: int = DEFAULT_SAMPLING_SEED
    scope: CandidateScope = CandidateScope.ALL

    def __post_init__(self) -> None:
        """Reject counts that leave a query without negatives."""
        if self.max_rule_candidates < 0 or self.num_random_candidates < 0:
            raise ConfigurationError("candidate counts must not be negative")
        if self.max_rule_candidates + self.num_random_candidates == 0:
            raise ConfigurationError("at least one negative per query is required")


@dataclass(frozen=True, slots=True)
class TrainingSet:
    """Padded per-query candidate features, correct answer first.

    Parameters
    ----------
    features : Tensor
        ``(num_queries, max_candidates, num_features)``; row
        :data:`TARGET_POSITION` of each query is its correct answer.
    mask : Tensor
        ``(num_queries, max_candidates)`` bool, ``False`` on padding.
    """

    features: Tensor
    mask: Tensor


def build_training_set(
    featurizer: QueryFeaturizer,
    queries: Sequence[RankingQuery],
    known: KnownAnswers,
    sampling: CandidateSampling,
) -> TrainingSet:
    """Featurize each query and keep its target plus sampled negatives.

    Parameters
    ----------
    featurizer : QueryFeaturizer
        Builds the full candidate feature matrix of a query.
    queries : Sequence[RankingQuery]
        The queries to learn from.
    known : KnownAnswers
        True answers never used as negatives.
    sampling : CandidateSampling
        How many negatives of each kind to keep.

    Returns
    -------
    TrainingSet
        One padded block per kept query; empty when the scope keeps none.

    Raises
    ------
    ConfigurationError
        If ``queries`` is empty.
    """
    if not queries:
        raise ConfigurationError("queries must not be empty")

    generator = torch.Generator().manual_seed(sampling.seed)
    blocks = [
        block
        for query in queries
        if (block := _query_block(featurizer, query, known, sampling, generator))
        is not None
    ]
    if not blocks:
        num_features = len(featurizer.feature_names())
        return TrainingSet(
            features=torch.zeros(0, 0, num_features),
            mask=torch.zeros(0, 0, dtype=torch.bool),
        )
    return _pad(blocks)


def concat_training_sets(training_sets: Sequence[TrainingSet]) -> TrainingSet:
    """Stack training sets with the same features into one.

    Parameters
    ----------
    training_sets : Sequence[TrainingSet]
        Sets whose feature columns mean the same thing.

    Returns
    -------
    TrainingSet
        Every query of every set, padded to the widest.

    Raises
    ------
    ConfigurationError
        If ``training_sets`` hold no query or their feature counts differ.
    """
    if len({ts.features.size(2) for ts in training_sets}) > 1:
        raise ConfigurationError("training sets have different feature counts")
    blocks = [
        ts.features[index, ts.mask[index]]
        for ts in training_sets
        for index in range(ts.features.size(0))
    ]
    if not blocks:
        raise ConfigurationError("training_sets hold no queries")
    return _pad(blocks)


def _query_block(
    featurizer: QueryFeaturizer,
    query: RankingQuery,
    known: KnownAnswers,
    sampling: CandidateSampling,
    generator: torch.Generator,
) -> Tensor | None:
    """Return one query's sampled candidate features, target first.

    ``None`` when the scope leaves the query's answer out.
    """
    match sampling.scope:
        case CandidateScope.ALL:
            candidates = None
            target_row = query.target_id
        case CandidateScope.REACHED:
            candidates = featurizer.reached(query.direction, query.query_id)
            matches = torch.nonzero(candidates == query.target_id)
            if matches.numel() == 0:
                return None
            target_row = int(matches[0, 0].item())
        case _ as unreachable:
            assert_never(unreachable)
    features = featurizer.featurize(
        query.direction, query.query_id, candidates=candidates
    )
    excluded = _known_rows(known.of(query), features.size(0), candidates)
    rows = _sample_rows(
        features[:, featurizer.score_column], target_row, excluded, sampling, generator
    )
    return features[rows]


def _known_rows(
    known: frozenset[int], num_rows: int, candidates: Tensor | None
) -> Tensor:
    """Return a mask of the feature rows holding a known answer."""
    known_ids = torch.tensor(sorted(known), dtype=torch.long)
    if candidates is not None:
        return torch.isin(candidates, known_ids)
    excluded = torch.zeros(num_rows, dtype=torch.bool)
    excluded[known_ids] = True
    return excluded


def _sample_rows(
    rule_scores: Tensor,
    target_row: int,
    excluded: Tensor,
    sampling: CandidateSampling,
    generator: torch.Generator,
) -> Tensor:
    """Return the target, then top rule-scored, then random negative rows."""
    available = ~excluded
    available[target_row] = False

    reached = torch.nonzero(available & (rule_scores > 0), as_tuple=True)[0]
    order = torch.argsort(rule_scores[reached], descending=True, stable=True)
    top = reached[order[: sampling.max_rule_candidates]]
    available[top] = False

    pool = torch.nonzero(available, as_tuple=True)[0]
    draw = torch.randperm(pool.numel(), generator=generator)
    random_rows = pool[draw[: sampling.num_random_candidates]]
    return torch.cat([torch.tensor([target_row]), top, random_rows])


def _pad(blocks: Sequence[Tensor]) -> TrainingSet:
    """Stack variable-length candidate blocks into one zero-padded tensor."""
    max_candidates = max(block.size(0) for block in blocks)
    num_features = blocks[0].size(1)
    features = torch.zeros(len(blocks), max_candidates, num_features)
    mask = torch.zeros(len(blocks), max_candidates, dtype=torch.bool)
    for index, block in enumerate(blocks):
        features[index, : block.size(0)] = block
        mask[index, : block.size(0)] = True
    return TrainingSet(features=features, mask=mask)
