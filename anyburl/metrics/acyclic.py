"""Body lookups and metrics for acyclic rules."""

from dataclasses import dataclass
from typing import assert_never

import numpy as np
from numpy.typing import NDArray

from ..graph import EdgeTypeTuple, HeteroGraph
from ..rule import Atom, Rule, TermKind
from .rule_metrics import RuleMetrics, metrics_from_counts

__all__ = ["AcyclicIndex", "acyclic_constant", "evaluate_acyclic"]


@dataclass(frozen=True, slots=True)
class _Adjacency:
    """Both-direction adjacency of one edge type as flat numpy arrays."""

    row_ptr: NDArray[np.int64]
    targets: NDArray[np.int64]
    col_ptr: NDArray[np.int64]
    sources: NDArray[np.int64]
    has_outgoing: NDArray[np.bool_]
    entities_with_outgoing: NDArray[np.int64]


class AcyclicIndex:
    """Answers which entities satisfy a single-atom acyclic body.

    Adjacency tables are built per edge type on first use.

    Parameters
    ----------
    graph : HeteroGraph
        The graph the bodies are matched against.
    """

    def __init__(self, graph: HeteroGraph) -> None:
        self._graph = graph
        self._tables: dict[EdgeTypeTuple, _Adjacency] = {}

    def satisfying(self, body: Atom) -> NDArray[np.int64]:
        """Return the entities the body's free subject can bind to.

        Parameters
        ----------
        body : Atom
            A body atom whose subject is the rule's free variable.

        Returns
        -------
        NDArray[np.int64]
            Distinct entity ids, ascending.
        """
        table = self._table(body.edge_signature)
        match body.object_.kind:
            case TermKind.CONSTANT:
                return self.sources_into(body.edge_signature, _entity_id(body))
            case TermKind.VARIABLE:
                return table.entities_with_outgoing
            case _ as unreachable:
                assert_never(unreachable)

    def has_outgoing(self, edge_type: EdgeTypeTuple) -> NDArray[np.bool_]:
        """Return a mask of the source entities with at least one edge."""
        return self._table(edge_type).has_outgoing

    def targets_from(self, edge_type: EdgeTypeTuple, source: int) -> NDArray[np.int64]:
        """Return the targets ``source`` reaches by ``edge_type``."""
        table = self._table(edge_type)
        return table.targets[table.row_ptr[source] : table.row_ptr[source + 1]]

    def sources_into(self, edge_type: EdgeTypeTuple, target: int) -> NDArray[np.int64]:
        """Return the distinct sources reaching ``target`` by ``edge_type``."""
        table = self._table(edge_type)
        return np.unique(
            table.sources[table.col_ptr[target] : table.col_ptr[target + 1]]
        )

    def _table(self, edge_type: EdgeTypeTuple) -> _Adjacency:
        """Return the cached adjacency of ``edge_type``, building it if needed."""
        table = self._tables.get(edge_type)
        if table is None:
            table = self._build(edge_type)
            self._tables[edge_type] = table
        return table

    def _build(self, edge_type: EdgeTypeTuple) -> _Adjacency:
        """Build both-direction adjacency of ``edge_type``."""
        src_type, _, dst_type = edge_type
        num_src = self._graph.node_count(src_type)
        num_dst = self._graph.node_count(dst_type)
        sources, targets = (
            t.numpy().astype(np.int64) for t in self._graph.edge_index(edge_type)
        )

        by_source = np.argsort(sources, kind="stable")
        out_degree = np.bincount(sources, minlength=num_src)
        by_target = np.argsort(targets, kind="stable")
        in_degree = np.bincount(targets, minlength=num_dst)
        has_outgoing = out_degree > 0
        return _Adjacency(
            row_ptr=np.concatenate(([0], np.cumsum(out_degree))),
            targets=targets[by_source],
            col_ptr=np.concatenate(([0], np.cumsum(in_degree))),
            sources=sources[by_target],
            has_outgoing=has_outgoing,
            entities_with_outgoing=np.flatnonzero(has_outgoing).astype(np.int64),
        )


def acyclic_constant(rule: Rule) -> tuple[bool, int]:
    """Return whether an acyclic rule's constant is its head object, and its id.

    Parameters
    ----------
    rule : Rule
        An acyclic rule.

    Returns
    -------
    tuple[bool, int]
        ``(is_object_constant, constant_id)``.
    """
    head = rule.head
    is_object_constant = head.object_.kind is TermKind.CONSTANT
    term = head.object_ if is_object_constant else head.subject
    if term.entity_id is None:
        raise AssertionError(f"acyclic rule has no head constant: {rule}")
    return is_object_constant, term.entity_id


def evaluate_acyclic(
    rule: Rule, index: AcyclicIndex, graph: HeteroGraph
) -> RuleMetrics:
    """Count an acyclic rule's predictions and support.

    ``r(X, c) :- b(X, ...)`` predicts ``r(x, c)`` for every ``x`` satisfying
    the body; it is supported where that triple is already in the graph.

    Parameters
    ----------
    rule : Rule
        An acyclic rule.
    index : AcyclicIndex
        Body lookups over ``graph``.
    graph : HeteroGraph
        The graph holding the head relation.

    Returns
    -------
    RuleMetrics
        The rule's metrics.
    """
    head_edge_type = rule.head.edge_signature
    body = rule.body[0]
    is_object_constant, constant_id = acyclic_constant(rule)
    if is_object_constant:
        known = index.sources_into(head_edge_type, constant_id)
    else:
        known = np.unique(index.targets_from(head_edge_type, constant_id))

    match body.object_.kind:
        case TermKind.CONSTANT:
            satisfying = index.satisfying(body)
            num_predictions = int(satisfying.size)
            support = int(np.isin(known, satisfying).sum())
        case TermKind.VARIABLE:
            has_outgoing = index.has_outgoing(body.edge_signature)
            num_predictions = int(has_outgoing.sum())
            support = int(has_outgoing[known].sum())
        case _ as unreachable:
            assert_never(unreachable)
    return metrics_from_counts(
        num_predictions, support, graph.edge_count(head_edge_type)
    )


def _entity_id(body: Atom) -> int:
    """Return the constant id at a body atom's object."""
    entity_id = body.object_.entity_id
    if entity_id is None:
        raise AssertionError(f"body object is not a constant: {body}")
    return entity_id
