"""Domain types shared by every stage: edge types, body chains, triples, queries."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from torch import Tensor

__all__ = ["BodyChain", "EdgeTypeTuple", "EntityScorer", "QueryDirection", "Triple"]

EdgeTypeTuple = tuple[str, str, str]
"""(source_node_type, relation, destination_node_type)."""

BodyChain = tuple[EdgeTypeTuple, ...]
"""Edge types of a rule body in traversal order; shared by rules with the same chain."""


@dataclass(frozen=True, slots=True)
class Triple:
    """A single edge of the knowledge graph, as a target or test triple.

    Parameters
    ----------
    head_id : int
        Source entity index.
    tail_id : int
        Destination entity index.
    head_type : str
        Node type of the source entity.
    tail_type : str
        Node type of the destination entity.
    relation : str
        The relation connecting head to tail.
    """

    head_id: int
    tail_id: int
    head_type: str
    tail_type: str
    relation: str

    @property
    def edge_type(self) -> EdgeTypeTuple:
        """Return ``(head_type, relation, tail_type)``."""
        return (self.head_type, self.relation, self.tail_type)


class QueryDirection(StrEnum):
    """Which endpoint of a triple a link prediction query ranks.

    Attributes
    ----------
    TAIL : str
        Rank candidate tails given the head.
    HEAD : str
        Rank candidate heads given the tail.
    """

    TAIL = "tail"
    HEAD = "head"


class EntityScorer(Protocol):
    """Scores candidate entities for link prediction queries of one relation.

    Satisfied by :class:`~anyburl.prediction.RulePredictor`, the scorers in
    :mod:`anyburl.baselines` and :class:`~anyburl.aggregation.AggregatedScorer`.
    """

    def score_tails(self, head_id: int) -> Tensor:
        """Return scores over all candidate tails for ``head_id``."""
        ...

    def score_heads(self, tail_id: int) -> Tensor:
        """Return scores over all candidate heads for ``tail_id``."""
        ...
