"""Grounding body chains against the graph, one query entity at a time."""

from dataclasses import dataclass

from torch import Tensor

from .._logging import get_logger
from ..chain import PathCounter
from ..core import BodyChain
from ..graph import HeteroGraph
from ..rule import Rule, RuleType
from ..rule_evaluation import RuleSet

__all__ = ["ChainGrounding", "body_chain_key", "build_groundings"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ChainGrounding:
    """One body chain's predictions for a single query entity.

    Parameters
    ----------
    chain : BodyChain
        Body edge types in chain order.
    counter : PathCounter
        Counts the chain's paths on the graph.
    """

    chain: BodyChain
    counter: PathCounter

    def row(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Return candidates reachable from ``query_id``, with grounding counts."""
        return self.counter.from_source(self.chain, query_id)

    def column(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Return candidates reaching ``query_id``, with grounding counts."""
        return self.counter.into_target(self.chain, query_id)


def body_chain_key(rule: Rule) -> BodyChain:
    """Return the edge types of a rule's body, in chain order."""
    return tuple(atom.edge_signature for atom in rule.body)


def build_groundings(
    graph: HeteroGraph,
    results: RuleSet,
) -> dict[BodyChain, ChainGrounding]:
    """Build one grounding per distinct body chain, skipping acyclic rules.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    results : RuleSet
        Rules with their metrics.

    Returns
    -------
    dict[BodyChain, ChainGrounding]
        Grounding per chain.

    Raises
    ------
    GraphSchemaError
        If a chain uses an edge type missing from, or empty in, ``graph``.
    """
    counter = graph.cached(PathCounter)
    groundings = {
        key: ChainGrounding(chain=key, counter=counter)
        for key in dict.fromkeys(
            body_chain_key(rule)
            for rule, _ in results
            if rule.rule_type is not RuleType.ACYCLIC
        )
    }
    for key in groundings:
        counter.check(key)
    logger.debug(
        "Built %d chain groundings from %d rules", len(groundings), len(results)
    )
    return groundings
