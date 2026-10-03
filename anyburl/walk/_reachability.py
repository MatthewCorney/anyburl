"""Per-step candidate edge tables, optionally pruned by type reachability.

For every (target node type, current node type, steps remaining) triple, the
tables list the edge types a walk may take next. When pruned, edge types whose
destination node type cannot reach the target within the remaining steps are
left out. The unpruned lists share the same layout, so one kernel serves both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._numba_graph import NumbaGraphView

MAX_TRACKED_STEPS: int = 8
"""Largest remaining-step count with its own candidate list.

Reachability within *k* steps stops changing once *k* exceeds the graph's
type-level diameter, so longer walks reuse this row without losing paths.
"""


@dataclass(frozen=True, slots=True)
class StepTables:
    """Candidate edge types indexed by target type, node type and steps left.

    Parameters
    ----------
    edges : NDArray[np.int64]
        Flattened candidate edge-type ids for every block.
    offsets : NDArray[np.int64]
        Block start indices into ``edges``, laid out so block
        ``(target, node_type, steps_left)`` begins at
        ``offsets[(target * num_node_types + node_type) * stride + steps_left]``
        where ``stride`` is ``MAX_TRACKED_STEPS + 1``.
    cumulative_weights : NDArray[np.float64]
        Running weight within each block, normalised so the last entry of
        a block is 1.0. A uniformly weighted block is evenly spaced, so
        one kernel serves both weightings.
    num_node_types : int
        Number of node types, needed to decode the block index.
    pruned : bool
        Whether unreachable destinations were removed.
    """

    edges: NDArray[np.int64]
    offsets: NDArray[np.int64]
    cumulative_weights: NDArray[np.float64]
    num_node_types: int
    pruned: bool


def build_step_tables(
    view: NumbaGraphView,
    *,
    prune: bool,
    weights: NDArray[np.float64] | None = None,
) -> StepTables:
    """Build candidate edge tables for every target type and step budget.

    Parameters
    ----------
    view : NumbaGraphView
        The integer-encoded graph.
    prune : bool
        When ``True``, a block holds only edge types whose destination can
        still reach the target node type within the remaining steps. When
        ``False``, every block holds all outgoing edge types, reproducing
        uniform selection exactly.
    weights : NDArray[np.float64] | None
        Per-edge-type selection weight. ``None`` weights every candidate
        equally.

    Returns
    -------
    StepTables
        Tables ready to index from a walk kernel.
    """
    num_node_types = len(view.node_type_names)
    successors = _successor_types(view, num_node_types)
    stride = MAX_TRACKED_STEPS + 1

    edges: list[int] = []
    offsets: list[int] = [0]
    cumulative: list[float] = []
    for target in range(num_node_types):
        within = _reach_within(successors, target, num_node_types)
        for node_type in range(num_node_types):
            outgoing = _outgoing_edges(view, node_type)
            for steps_left in range(stride):
                block = _candidates(view, outgoing, within, steps_left, prune=prune)
                edges.extend(block)
                cumulative.extend(_cumulative_weights(block, weights))
                offsets.append(len(edges))

    return StepTables(
        edges=np.array(edges, dtype=np.int64),
        offsets=np.array(offsets, dtype=np.int64),
        cumulative_weights=np.array(cumulative, dtype=np.float64),
        num_node_types=num_node_types,
        pruned=prune,
    )


def _cumulative_weights(
    block: list[int],
    weights: NDArray[np.float64] | None,
) -> list[float]:
    """Return a block's running weights, normalised to end at 1.0."""
    if not block:
        return []
    raw = [1.0 if weights is None else float(weights[edge]) for edge in block]
    total = sum(raw)
    if total <= 0.0:
        raw = [1.0] * len(block)
        total = float(len(block))
    running = 0.0
    cumulative: list[float] = []
    for value in raw:
        running += value
        cumulative.append(running / total)
    cumulative[-1] = 1.0
    return cumulative


def _candidates(
    view: NumbaGraphView,
    outgoing: list[int],
    within: NDArray[np.bool_],
    steps_left: int,
    *,
    prune: bool,
) -> list[int]:
    """Return the edge types worth taking with ``steps_left`` remaining."""
    if not prune:
        return list(outgoing)
    if steps_left < 1:
        return []
    return [
        edge
        for edge in outgoing
        if within[steps_left - 1, int(view.edge_dst_type[edge])]
    ]


def _successor_types(view: NumbaGraphView, num_node_types: int) -> list[set[int]]:
    """Return the node types directly reachable from each node type."""
    successors: list[set[int]] = [set() for _ in range(num_node_types)]
    for node_type in range(num_node_types):
        for edge in _outgoing_edges(view, node_type):
            successors[node_type].add(int(view.edge_dst_type[edge]))
    return successors


def _outgoing_edges(view: NumbaGraphView, node_type: int) -> list[int]:
    """Return the edge-type ids leaving a node type."""
    start = int(view.node_out_offsets[node_type])
    stop = int(view.node_out_offsets[node_type + 1])
    return [int(edge) for edge in view.node_out_edges[start:stop]]


def _reach_within(
    successors: list[set[int]],
    target: int,
    num_node_types: int,
) -> NDArray[np.bool_]:
    """Return which node types reach ``target`` within *k* steps.

    "Within" rather than "exactly" is deliberate: a walk with three steps
    left may still succeed in one, so pruning on an exact step count would
    discard viable continuations.

    Parameters
    ----------
    successors : list[set[int]]
        Directly reachable node types per node type.
    target : int
        The node-type id the walk must end on.
    num_node_types : int
        Total node types.

    Returns
    -------
    NDArray[np.bool_]
        Array of shape ``(MAX_TRACKED_STEPS + 1, num_node_types)`` where
        entry ``[k, t]`` is ``True`` when ``t`` reaches ``target`` in at
        most ``k`` steps.
    """
    within = np.zeros((MAX_TRACKED_STEPS + 1, num_node_types), dtype=np.bool_)
    within[0, target] = True
    for steps in range(1, MAX_TRACKED_STEPS + 1):
        within[steps] = within[steps - 1]
        for node_type in range(num_node_types):
            if within[steps, node_type]:
                continue
            within[steps, node_type] = any(
                within[steps - 1, destination] for destination in successors[node_type]
            )
    return within
