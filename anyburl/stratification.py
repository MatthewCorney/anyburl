"""Link prediction metrics broken down by how well-connected the target is.

An aggregate MRR hides whether a scorer works on rarely seen entities or
only on hubs. Grouping queries by the training degree of their correct
answer exposes that, and makes a popularity prior's contribution visible.
"""

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import assert_never

import torch
from torch import Tensor

from .evaluation import (
    DEFAULT_K_VALUES,
    LinkPredictionMetrics,
    QueryDirection,
    QueryRank,
    summarize_ranks,
)
from .exceptions import ConfigurationError
from .graph import EdgeTypeTuple, HeteroGraph

__all__ = [
    "DegreeBucket",
    "RelationDegrees",
    "StratificationConfig",
    "relation_degrees",
    "stratify_by_degree",
]

DEFAULT_NUM_BUCKETS: int = 4
"""Default number of equal-frequency degree buckets per query direction."""

UNSEEN_DEGREE: int = 0
"""Degree of an entity with no edges of the target relation in training."""


@dataclass(frozen=True, slots=True)
class RelationDegrees:
    """How many edges of one relation each entity takes part in.

    Parameters
    ----------
    head : Tensor
        Out-degree of every candidate head, indexed by local id.
    tail : Tensor
        In-degree of every candidate tail, indexed by local id.
    """

    head: Tensor
    tail: Tensor

    def of_target(self, query: QueryRank) -> int:
        """Return the degree of the entity ``query`` was asked to rank."""
        match query.direction:
            case QueryDirection.TAIL:
                return int(self.tail[query.target_id].item())
            case QueryDirection.HEAD:
                return int(self.head[query.target_id].item())
            case _ as unreachable:
                assert_never(unreachable)


def relation_degrees(graph: HeteroGraph, edge_type: EdgeTypeTuple) -> RelationDegrees:
    """Count each entity's edges of ``edge_type``.

    Parameters
    ----------
    graph : HeteroGraph
        The graph to count in, normally the training graph.
    edge_type : EdgeTypeTuple
        The relation being predicted.

    Returns
    -------
    RelationDegrees
        Head out-degrees and tail in-degrees as float tensors.
    """
    src_type, _, dst_type = edge_type
    edge_index = graph.edge_index(edge_type)
    weights = torch.ones(edge_index.size(1))
    return RelationDegrees(
        head=torch.zeros(graph.node_count(src_type)).scatter_add_(
            0, edge_index[0], weights
        ),
        tail=torch.zeros(graph.node_count(dst_type)).scatter_add_(
            0, edge_index[1], weights
        ),
    )


@dataclass(frozen=True, slots=True)
class StratificationConfig:
    """How to group queries into degree buckets.

    Parameters
    ----------
    num_buckets : int
        Target number of equal-frequency buckets per direction. Fewer are
        produced when many queries share a degree, since one degree is
        never split across buckets.
    k_values : tuple[int, ...]
        Hits@K thresholds.

    Raises
    ------
    ConfigurationError
        If ``num_buckets`` is less than 1.
    """

    num_buckets: int = DEFAULT_NUM_BUCKETS
    k_values: tuple[int, ...] = DEFAULT_K_VALUES

    def __post_init__(self) -> None:
        """Reject a bucket count that cannot hold any queries."""
        if self.num_buckets < 1:
            msg = f"num_buckets must be at least 1, got {self.num_buckets}"
            raise ConfigurationError(msg)


@dataclass(frozen=True, slots=True)
class DegreeBucket:
    """Metrics for the queries whose target degree falls in one range.

    Parameters
    ----------
    direction : QueryDirection
        Which endpoint these queries ranked.
    min_degree : int
        Smallest target degree in the bucket.
    max_degree : int
        Largest target degree in the bucket.
    metrics : LinkPredictionMetrics
        MRR and Hits@K over the bucket's queries.
    """

    direction: QueryDirection
    min_degree: int
    max_degree: int
    metrics: LinkPredictionMetrics


def stratify_by_degree(
    query_ranks: Sequence[QueryRank],
    degrees: RelationDegrees,
    config: StratificationConfig,
) -> tuple[DegreeBucket, ...]:
    """Split query ranks into buckets by the training degree of their target.

    Tail and head queries are bucketed separately because their degree
    distributions differ. Targets unseen in training always get a bucket
    of their own.

    Parameters
    ----------
    query_ranks : Sequence[QueryRank]
        Ranks from :meth:`LinkPredictionEvaluator.rank_queries`.
    degrees : RelationDegrees
        Degrees from the graph the scorer was trained on.
    config : StratificationConfig
        Bucket count and Hits@K thresholds.

    Returns
    -------
    tuple[DegreeBucket, ...]
        Tail buckets then head buckets, each in ascending degree order.
    """
    buckets: list[DegreeBucket] = []
    for direction in QueryDirection:
        scored = [
            (degrees.of_target(query), query.rank)
            for query in query_ranks
            if query.direction is direction
        ]
        buckets.extend(_bucket_direction(direction, scored, config))
    return tuple(buckets)


def _bucket_direction(
    direction: QueryDirection,
    scored: Sequence[tuple[int, float]],
    config: StratificationConfig,
) -> list[DegreeBucket]:
    """Bucket one direction's ``(degree, rank)`` pairs."""
    if not scored:
        return []
    upper_bounds = _equal_frequency_bounds(
        sorted(degree for degree, _ in scored), config.num_buckets
    )
    grouped: list[list[tuple[int, float]]] = [[] for _ in upper_bounds]
    for degree, rank in scored:
        grouped[bisect_left(upper_bounds, degree)].append((degree, rank))
    return [
        DegreeBucket(
            direction=direction,
            min_degree=min(degree for degree, _ in members),
            max_degree=max(degree for degree, _ in members),
            metrics=summarize_ranks([rank for _, rank in members], config.k_values),
        )
        for members in grouped
        if members
    ]


def _equal_frequency_bounds(
    ordered_degrees: Sequence[int], num_buckets: int
) -> list[int]:
    """Return ascending inclusive upper degree bounds for each bucket.

    Parameters
    ----------
    ordered_degrees : Sequence[int]
        Target degrees sorted ascending, non-empty.
    num_buckets : int
        Target number of buckets.

    Returns
    -------
    list[int]
        Distinct upper bounds; the last is the largest degree.
    """
    count = len(ordered_degrees)
    bounds = {
        ordered_degrees[ceil(count * step / num_buckets) - 1]
        for step in range(1, num_buckets + 1)
    }
    if ordered_degrees[0] == UNSEEN_DEGREE:
        bounds.add(UNSEEN_DEGREE)
    return sorted(bounds)
