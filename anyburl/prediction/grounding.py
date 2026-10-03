"""Grounding body chains against the graph, materialised or per query."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, assert_never

import torch
from torch import Tensor

from .._logging import get_logger
from ..graph import HeteroGraph, suppress_sparse_csr_warning
from ..metrics import RuleMetrics
from ..rule import Rule, RuleType

__all__ = [
    "MATERIALISED_CHAIN_BYTES_PER_NNZ",
    "MAX_MATERIALISED_CHAIN_NNZ",
    "MAX_MATERIALISED_TOTAL_NNZ",
    "BodyChainKey",
    "ChainGrounding",
    "GroundingMode",
    "MaterialisedChain",
    "OnDemandChain",
    "body_chain_key",
    "build_groundings",
]

logger = get_logger(__name__)

BodyChainKey = tuple[tuple[str, str, str], ...]
"""Unique identifier for a body chain: tuple of edge-type triples."""

BYTES_PER_MEGABYTE: float = 1e6

MATERIALISED_CHAIN_BYTES_PER_NNZ: int = 24
"""Resident bytes per stored non-zero: index and value, doubled for the transpose."""

MAX_MATERIALISED_CHAIN_NNZ: int = 5_000_000
"""Largest single chain product held in memory, about 120 MB."""

MAX_MATERIALISED_TOTAL_NNZ: int = 10_000_000
"""Total non-zeros held across all materialised chains, about 240 MB.

Chains that do not fit are grounded per query instead, which gives identical
scores at the cost of time rather than memory.
"""


class GroundingMode(StrEnum):
    """How a body chain is grounded against the graph.

    Attributes
    ----------
    MATERIALISED : str
        Multiply the whole chain up front and keep the product and its
        transpose. Fastest when many entities are scored, but dense
        products can exhaust memory on large graphs.
    ON_DEMAND : str
        Propagate a one-hot vector through the body matrices per query.
        Adds no resident memory; suits evaluating a small test set.
    AUTO : str
        Materialise chains whose product size is known to fit the memory
        budget and ground the rest per query.
    """

    MATERIALISED = "materialised"
    ON_DEMAND = "on_demand"
    AUTO = "auto"


class ChainGrounding(Protocol):
    """Supplies one body chain's predictions for a single query entity."""

    def row(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Return candidates reachable from ``query_id``, with grounding counts."""
        ...

    def column(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Return candidates reaching ``query_id``, with grounding counts."""
        ...


@dataclass(frozen=True, slots=True)
class MaterialisedChain:
    """A body chain whose full product and transpose are held in memory.

    Parameters
    ----------
    product : Tensor
        Sparse CSR float tensor of the chain product.
    product_transposed : Tensor
        Its transpose, for reverse-direction queries.
    """

    product: Tensor
    product_transposed: Tensor

    def row(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Slice one row of the product."""
        return _slice_csr_row(self.product, query_id)

    def column(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Slice one row of the transposed product."""
        return _slice_csr_row(self.product_transposed, query_id)


@dataclass(frozen=True, slots=True)
class OnDemandChain:
    """A body chain grounded per query, holding no product of its own.

    Parameters
    ----------
    matrices : tuple[Tensor, ...]
        Body atom adjacency matrices, in chain order.
    """

    matrices: tuple[Tensor, ...]

    def row(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Propagate a one-hot row vector forward through the chain."""
        vector = torch.zeros(1, int(self.matrices[0].shape[0]))
        vector[0, query_id] = 1.0
        with suppress_sparse_csr_warning():
            for matrix in self.matrices:
                vector = vector @ matrix
        return _nonzero_entries(vector.squeeze(0))

    def column(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Propagate a one-hot column vector backward through the chain."""
        vector = torch.zeros(int(self.matrices[-1].shape[1]), 1)
        vector[query_id, 0] = 1.0
        with suppress_sparse_csr_warning():
            for matrix in reversed(self.matrices):
                vector = matrix @ vector
        return _nonzero_entries(vector.squeeze(1))


def body_chain_key(rule: Rule) -> BodyChainKey:
    """Return the edge types of a rule's body, in chain order."""
    return tuple(atom.edge_signature for atom in rule.body)


def build_groundings(
    graph: HeteroGraph,
    results: Sequence[tuple[Rule, RuleMetrics]],
    mode: GroundingMode,
) -> dict[BodyChainKey, ChainGrounding]:
    """Build one grounding per distinct body chain, skipping AC2 rules.

    AC2 rules predict a Cartesian product, so they contribute no
    per-candidate evidence.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    results : Sequence[tuple[Rule, RuleMetrics]]
        Rules with their metrics.
    mode : GroundingMode
        Whether to materialise each chain, ground per query, or decide by
        size.

    Returns
    -------
    dict[BodyChainKey, ChainGrounding]
        Grounding per chain.
    """
    chains = _unique_body_chains(graph, results)
    known_sizes = _known_chain_sizes(results)

    groundings: dict[BodyChainKey, ChainGrounding] = {}
    remaining = MAX_MATERIALISED_TOTAL_NNZ
    with suppress_sparse_csr_warning():
        for key, matrices in chains.items():
            grounding, consumed = _build_grounding(
                matrices, mode, known_sizes.get(key), remaining
            )
            groundings[key] = grounding
            remaining -= consumed

    logger.debug(
        "Built %d %s chain groundings from %d rules (%.0f MB budgeted)",
        len(groundings),
        mode.value,
        len(results),
        (MAX_MATERIALISED_TOTAL_NNZ - remaining)
        * MATERIALISED_CHAIN_BYTES_PER_NNZ
        / BYTES_PER_MEGABYTE,
    )
    return groundings


def _unique_body_chains(
    graph: HeteroGraph,
    results: Sequence[tuple[Rule, RuleMetrics]],
) -> dict[BodyChainKey, tuple[Tensor, ...]]:
    """Collect the body matrices of each distinct non-AC2 chain."""
    chains: dict[BodyChainKey, tuple[Tensor, ...]] = {}
    for rule, _ in results:
        if rule.rule_type is RuleType.AC2:
            continue
        key = body_chain_key(rule)
        if key not in chains:
            chains[key] = tuple(graph.get_csr_matrix(edge_type) for edge_type in key)
    return chains


def _known_chain_sizes(
    results: Sequence[tuple[Rule, RuleMetrics]],
) -> dict[BodyChainKey, int]:
    """Read each chain's product size off its cyclic rules' metrics.

    A cyclic rule's ``num_predictions`` is exactly its chain product's
    non-zero count. Chains seen only through AC1 rules have no known size.
    """
    return {
        body_chain_key(rule): metrics.num_predictions
        for rule, metrics in results
        if rule.rule_type is RuleType.CYCLIC
    }


def _build_grounding(
    matrices: Sequence[Tensor],
    mode: GroundingMode,
    known_nnz: int | None,
    remaining_nnz: int,
) -> tuple[ChainGrounding, int]:
    """Build one chain's grounding and return the budget it consumed.

    Under :attr:`GroundingMode.AUTO` a chain is materialised only when its
    size is known before multiplying and fits both budgets; otherwise it is
    grounded per query.

    Parameters
    ----------
    matrices : Sequence[Tensor]
        Body atom adjacency matrices, in chain order.
    mode : GroundingMode
        The grounding mode.
    known_nnz : int | None
        Exact non-zero count of the product, where known.
    remaining_nnz : int
        Non-zeros still available in the total budget.

    Returns
    -------
    tuple[ChainGrounding, int]
        The grounding and the non-zeros it consumed.
    """
    match mode:
        case GroundingMode.ON_DEMAND:
            return OnDemandChain(matrices=tuple(matrices)), 0
        case GroundingMode.MATERIALISED:
            return _materialise(matrices), 0
        case GroundingMode.AUTO:
            ceiling = min(MAX_MATERIALISED_CHAIN_NNZ, remaining_nnz)
            if known_nnz is None or known_nnz > ceiling:
                return OnDemandChain(matrices=tuple(matrices)), 0
            return _materialise(matrices), known_nnz
        case _ as unreachable:
            assert_never(unreachable)


def _materialise(matrices: Sequence[Tensor]) -> MaterialisedChain:
    """Multiply the chain out and keep the product with its transpose."""
    product = matrices[0]
    for matrix in matrices[1:]:
        product = product @ matrix
    return MaterialisedChain(
        product=product,
        product_transposed=product.t().to_sparse_csr(),
    )


def _slice_csr_row(csr_matrix: Tensor, row_id: int) -> tuple[Tensor, Tensor]:
    """Return the stored column indices and values of one CSR row."""
    crow = csr_matrix.crow_indices()
    start = int(crow[row_id].item())
    end = int(crow[row_id + 1].item())
    return csr_matrix.col_indices()[start:end], csr_matrix.values()[start:end]


def _nonzero_entries(dense: Tensor) -> tuple[Tensor, Tensor]:
    """Return the indices and values of a dense vector's non-zeros."""
    indices = torch.nonzero(dense, as_tuple=True)[0]
    return indices, dense[indices]
