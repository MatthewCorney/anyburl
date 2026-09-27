"""Hold out target-relation edges so link prediction can be scored honestly.

Learning rules from a graph and then ranking edges of that same graph
measures memorisation, not prediction. :func:`split_target_edges` removes a
fraction of one edge type from the training graph and returns those edges as
test triples.

Many heterogeneous datasets store each relation twice, once in each
direction. Removing an edge only from the forward direction leaves its mirror
in place, and a two-hop body chain walks straight back through it, so the
default is to drop the mirrored inverse as well.
"""

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch_geometric.data import HeteroData

from ._logging import get_logger
from .graph import EdgeTypeTuple
from .sampler import Triple

logger = get_logger(__name__)

DEFAULT_TEST_FRACTION: float = 0.1
"""Share of target edges held out when none is given."""

DEFAULT_SPLIT_SEED: int = 42
"""Random seed used to choose held-out edges when none is given."""

EDGE_INDEX_KEY: str = "edge_index"
"""The one edge-parallel attribute this module knows how to filter."""


class InverseEdgeHandling(StrEnum):
    """What to do with the mirrored inverse of the target edge type.

    Attributes
    ----------
    REMOVE_MIRRORED : str
        Also remove the inverse edges that mirror a held-out edge. The safe
        default: without it a held-out edge stays reachable in one hop and
        the evaluation is not honest. The inverse defaults to
        ``(dst_type, relation, src_type)`` and is a no-op when the graph has
        no such edge type, so a dataset that names its inverse differently
        must say so via ``SplitConfig.inverse_edge_type``.
    KEEP : str
        Leave the reverse edge type untouched, asserting that it does not
        mirror the target relation.
    """

    REMOVE_MIRRORED = "remove_mirrored"
    KEEP = "keep"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Configuration for holding out target-relation edges.

    Parameters
    ----------
    target_edge_type : EdgeTypeTuple
        The ``(src_type, relation, dst_type)`` edges to split.
    test_fraction : float
        Share of those edges to hold out, in ``(0.0, 1.0)``.
    seed : int
        Random seed selecting which edges are held out.
    inverse_handling : InverseEdgeHandling
        Whether to also remove the mirrored reverse edges.
    inverse_edge_type : EdgeTypeTuple | None
        The edge type that mirrors the target. ``None`` (the default) means
        ``(dst_type, relation, src_type)``, which covers datasets that store
        both directions under one relation name. Datasets that name the
        inverse differently --- BIOKG stores the inverse of
        ``protein -is_annotated_to-> phenotype`` as
        ``phenotype -has_annotated-> protein`` --- must name it here, or the
        removal silently does nothing and the split leaks.

    Raises
    ------
    ValueError
        If ``test_fraction`` is not strictly between 0 and 1, or an explicit
        ``inverse_edge_type`` is combined with ``InverseEdgeHandling.KEEP``.
    """

    target_edge_type: EdgeTypeTuple
    test_fraction: float = DEFAULT_TEST_FRACTION
    seed: int = DEFAULT_SPLIT_SEED
    inverse_handling: InverseEdgeHandling = InverseEdgeHandling.REMOVE_MIRRORED
    inverse_edge_type: EdgeTypeTuple | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(
                f"test_fraction must be in (0.0, 1.0), got {self.test_fraction}"
            )
        if (
            self.inverse_edge_type is not None
            and self.inverse_handling is InverseEdgeHandling.KEEP
        ):
            raise ValueError(
                "inverse_edge_type names an edge type to remove, but "
                "inverse_handling is KEEP; these contradict each other"
            )

    def resolve_inverse(self) -> EdgeTypeTuple:
        """Return the edge type treated as the target's inverse.

        Returns
        -------
        EdgeTypeTuple
            The explicit ``inverse_edge_type`` when set, otherwise the
            name-mirrored edge type.
        """
        if self.inverse_edge_type is not None:
            return self.inverse_edge_type
        return mirrored_edge_type(self.target_edge_type)


@dataclass(frozen=True, slots=True)
class TripleSplit:
    """A training graph paired with the triples withheld from it.

    Parameters
    ----------
    train_data : HeteroData
        The graph with held-out edges removed. Safe to fit on.
    test_triples : tuple[Triple, ...]
        The removed edges, as triples to evaluate against.
    """

    train_data: HeteroData
    test_triples: tuple[Triple, ...]


def mirrored_edge_type(edge_type: EdgeTypeTuple) -> EdgeTypeTuple:
    """Return the mirrored edge type of ``edge_type``.

    Parameters
    ----------
    edge_type : EdgeTypeTuple
        A ``(src_type, relation, dst_type)`` triple.

    Returns
    -------
    EdgeTypeTuple
        The same relation with its endpoint types swapped.
    """
    src_type, relation, dst_type = edge_type
    return (dst_type, relation, src_type)


def _edge_parallel_attributes(data: HeteroData, edge_type: EdgeTypeTuple) -> list[str]:
    """Return edge-parallel attributes that filtering would desynchronise.

    Only ``edge_index`` is filtered when edges are held out. Any other
    per-edge tensor would keep its original length and silently stop
    lining up with the edges it describes.

    Parameters
    ----------
    data : HeteroData
        The graph to inspect.
    edge_type : EdgeTypeTuple
        The edge type whose storage to inspect.

    Returns
    -------
    list[str]
        Names of edge-parallel attributes other than ``edge_index``.
    """
    store = data[edge_type]
    stored_names = list(store.keys())
    return [
        name
        for name in stored_names
        if name != EDGE_INDEX_KEY and store.is_edge_attr(name)
    ]


def split_target_edges(data: HeteroData, config: SplitConfig) -> TripleSplit:
    """Hold out a random fraction of one edge type for testing.

    Parameters
    ----------
    data : HeteroData
        The full graph. Not modified; the training graph is a clone.
    config : SplitConfig
        Which edges to hold out and how many.

    Returns
    -------
    TripleSplit
        The training graph and the held-out triples.

    Raises
    ------
    ValueError
        If ``target_edge_type`` is absent from ``data``, the requested
        fraction holds out no edges or all of them, or an affected edge
        type carries edge-parallel attributes beyond ``edge_index``.
    """
    target = config.target_edge_type
    if target not in data.edge_types:
        raise ValueError(
            f"target_edge_type {target!r} not in graph; "
            f"available edge types: {list(data.edge_types)!r}"
        )

    _reject_explicit_inverse_absent(data, config)
    _reject_unfilterable_edge_types(data, config)

    edge_index = data[target].edge_index
    num_edges = int(edge_index.size(1))
    num_test = int(num_edges * config.test_fraction)
    if num_test < 1 or num_test >= num_edges:
        raise ValueError(
            f"test_fraction {config.test_fraction} holds out {num_test} of "
            f"{num_edges} edges; choose a fraction leaving both splits non-empty"
        )

    generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(num_edges, generator=generator)
    test_edges = edge_index[:, permutation[:num_test]]

    train_data = data.clone()
    train_data[target].edge_index = edge_index[:, permutation[num_test:]]

    if config.inverse_handling is InverseEdgeHandling.REMOVE_MIRRORED:
        _remove_mirrored_edges(train_data, config.resolve_inverse(), test_edges)

    src_type, relation, dst_type = target
    test_triples = tuple(
        Triple(
            head_id=head_id,
            tail_id=tail_id,
            head_type=src_type,
            tail_type=dst_type,
            relation=relation,
        )
        for head_id, tail_id in zip(
            test_edges[0].tolist(), test_edges[1].tolist(), strict=True
        )
    )

    logger.info(
        "Split %r: %d train edges, %d test triples",
        target,
        num_edges - num_test,
        num_test,
    )
    return TripleSplit(train_data=train_data, test_triples=test_triples)


def _reject_explicit_inverse_absent(data: HeteroData, config: SplitConfig) -> None:
    """Raise if a named ``inverse_edge_type`` is not in the graph.

    Naming an inverse asserts it exists. The name-mirrored default may
    legitimately be absent, but a typo in an explicit one would silently
    leak the whole test set, so it must fail instead.

    Parameters
    ----------
    data : HeteroData
        The graph to inspect.
    config : SplitConfig
        The split configuration.

    Raises
    ------
    ValueError
        If ``config.inverse_edge_type`` is set but absent from ``data``.
    """
    explicit = config.inverse_edge_type
    if explicit is not None and explicit not in data.edge_types:
        raise ValueError(
            f"inverse_edge_type {explicit!r} not in graph; "
            f"available edge types: {list(data.edge_types)!r}"
        )


def _reject_unfilterable_edge_types(data: HeteroData, config: SplitConfig) -> None:
    """Raise if any edge type about to be filtered carries per-edge data.

    Parameters
    ----------
    data : HeteroData
        The graph to inspect.
    config : SplitConfig
        The split configuration.

    Raises
    ------
    ValueError
        If an affected edge type has edge-parallel attributes beyond
        ``edge_index``.
    """
    target = config.target_edge_type
    affected = [target]
    inverse = config.resolve_inverse()
    if (
        config.inverse_handling is InverseEdgeHandling.REMOVE_MIRRORED
        and inverse != target
        and inverse in data.edge_types
    ):
        affected.append(inverse)

    for edge_type in affected:
        extra = _edge_parallel_attributes(data, edge_type)
        if extra:
            raise ValueError(
                f"edge type {edge_type!r} carries edge-parallel attributes "
                f"{extra!r}; splitting filters only {EDGE_INDEX_KEY!r} and "
                f"would leave them misaligned with the remaining edges"
            )


def _remove_mirrored_edges(
    train_data: HeteroData,
    inverse: EdgeTypeTuple,
    test_edges: torch.Tensor,
) -> None:
    """Drop reverse edges mirroring the held-out ones, in place.

    Parameters
    ----------
    train_data : HeteroData
        The training graph, modified in place.
    inverse : EdgeTypeTuple
        The edge type holding the target's inverse edges.
    test_edges : Tensor
        ``(2, num_test)`` tensor of held-out ``(head, tail)`` pairs.
    """
    if inverse not in train_data.edge_types:
        return

    reverse_index = train_data[inverse].edge_index
    held_out = set(zip(test_edges[0].tolist(), test_edges[1].tolist(), strict=True))
    keep = torch.tensor(
        [
            (tail_id, head_id) not in held_out
            for head_id, tail_id in zip(
                reverse_index[0].tolist(), reverse_index[1].tolist(), strict=True
            )
        ],
        dtype=torch.bool,
    )
    train_data[inverse].edge_index = reverse_index[:, keep]

    logger.info(
        "Removed %d mirrored %r edges",
        int((~keep).sum().item()),
        inverse,
    )
