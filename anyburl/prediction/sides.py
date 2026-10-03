"""Per-direction views of the evidence each body chain contributes."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import assert_never

import torch
from torch import Tensor

from ..metrics import RuleMetrics, aggregate_confidence
from ..rule import Rule, RuleType, TermKind
from .grounding import BodyChainKey, ChainGrounding, body_chain_key
from .scoring import ScoringStrategy, evidence_weights

__all__ = ["ChainSide", "apply_side", "build_chain_sides"]

ChainFetch = Callable[[int], tuple[Tensor, Tensor]]
"""Returns a query entity's candidates and their grounding counts."""


@dataclass(frozen=True, slots=True)
class ChainSide:
    """One body chain's rules, seen from one query direction.

    Scoring tails for a head and heads for a tail are mirror images: each
    fetches the chain's candidates for the query entity, applies the
    confidence of rules that hold for that query, then applies the
    confidences of rules grounded on individual candidates.

    Parameters
    ----------
    chain : BodyChainKey
        The body chain these rules share.
    rule_type : RuleType
        ``CYCLIC`` or ``AC1``.
    fetch : ChainFetch
        The grounding's ``row`` (scoring tails) or ``column`` (scoring heads).
    shared_confidence : float | None
        Aggregated confidence applying to every query, for cyclic rules.
    query_confidences : Mapping[int, float]
        Aggregated confidence of AC1 rules grounded on each query entity.
    candidate_confidences : Mapping[int, float]
        Aggregated confidence of AC1 rules grounded on each candidate.
    """

    chain: BodyChainKey
    rule_type: RuleType
    fetch: ChainFetch
    shared_confidence: float | None = None
    query_confidences: Mapping[int, float] = field(default_factory=dict)
    candidate_confidences: Mapping[int, float] = field(default_factory=dict)
    candidate_ids: Tensor = field(init=False)
    candidate_values: Tensor = field(init=False)

    def __post_init__(self) -> None:
        """Index the candidate confidences as tensors for vectorised scoring."""
        object.__setattr__(
            self,
            "candidate_ids",
            torch.tensor(list(self.candidate_confidences), dtype=torch.long),
        )
        object.__setattr__(
            self,
            "candidate_values",
            torch.tensor(list(self.candidate_confidences.values()), dtype=torch.float),
        )

    def confidence_for(self, query_id: int) -> float | None:
        """Return the confidence applying to every candidate of ``query_id``."""
        if self.shared_confidence is not None:
            return self.shared_confidence
        return self.query_confidences.get(query_id)


def apply_side(
    complement: Tensor,
    side: ChainSide,
    query_id: int,
    strategy: ScoringStrategy,
) -> None:
    """Fold one chain side's evidence into a noisy-or complement, in place.

    Parameters
    ----------
    complement : Tensor
        Running ``prod(1 - contribution)`` over all candidates.
    side : ChainSide
        The chain side to apply.
    query_id : int
        The entity being queried.
    strategy : ScoringStrategy
        How grounding counts weigh on each contribution.
    """
    candidates, groundings = side.fetch(query_id)
    if candidates.numel() == 0:
        return
    weights = evidence_weights(groundings, strategy)

    confidence = side.confidence_for(query_id)
    if confidence is not None:
        complement[candidates] *= (1.0 - confidence) ** weights

    if side.candidate_ids.numel() == 0:
        return
    reachable = torch.isin(side.candidate_ids, candidates)
    if not bool(reachable.any()):
        return
    matching_ids = side.candidate_ids[reachable]
    dense_weights = torch.zeros(complement.numel())
    dense_weights[candidates] = weights
    complement[matching_ids] *= (1.0 - side.candidate_values[reachable]) ** (
        dense_weights[matching_ids]
    )


@dataclass(slots=True)
class _ChainConfidences:
    """Rule confidences collected per body chain, before aggregation."""

    cyclic: dict[BodyChainKey, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    subject_grounded: dict[BodyChainKey, dict[int, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    object_grounded: dict[BodyChainKey, dict[int, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )


def build_chain_sides(
    results: Sequence[tuple[Rule, RuleMetrics]],
    groundings: Mapping[BodyChainKey, ChainGrounding],
) -> tuple[list[ChainSide], list[ChainSide]]:
    """Group rules by body chain into tail-scoring and head-scoring sides.

    Parameters
    ----------
    results : Sequence[tuple[Rule, RuleMetrics]]
        Rules with their metrics. AC2 rules are ignored.
    groundings : Mapping[BodyChainKey, ChainGrounding]
        Grounding per body chain.

    Returns
    -------
    tuple[list[ChainSide], list[ChainSide]]
        Sides for scoring tails given a head, and heads given a tail, in
        the same chain order. Cyclic chains come first.
    """
    collected = _collect_confidences(results)
    tail_sides: list[ChainSide] = []
    head_sides: list[ChainSide] = []

    for key, confidences in collected.cyclic.items():
        shared = aggregate_confidence(confidences)
        for sides, fetch in (
            (tail_sides, groundings[key].row),
            (head_sides, groundings[key].column),
        ):
            sides.append(
                ChainSide(key, RuleType.CYCLIC, fetch, shared_confidence=shared)
            )

    for key in dict.fromkeys([*collected.subject_grounded, *collected.object_grounded]):
        by_subject = _aggregate(collected.subject_grounded.get(key, {}))
        by_object = _aggregate(collected.object_grounded.get(key, {}))
        tail_sides.append(
            ChainSide(
                key,
                RuleType.AC1,
                groundings[key].row,
                query_confidences=by_subject,
                candidate_confidences=by_object,
            )
        )
        head_sides.append(
            ChainSide(
                key,
                RuleType.AC1,
                groundings[key].column,
                query_confidences=by_object,
                candidate_confidences=by_subject,
            )
        )
    return tail_sides, head_sides


def _collect_confidences(
    results: Sequence[tuple[Rule, RuleMetrics]],
) -> _ChainConfidences:
    """Collect cyclic and AC1 rule confidences per body chain."""
    collected = _ChainConfidences()
    for rule, metrics in results:
        key = body_chain_key(rule)
        match rule.rule_type:
            case RuleType.CYCLIC:
                collected.cyclic[key].append(metrics.confidence)
            case RuleType.AC1:
                head = rule.head
                if head.subject.kind is TermKind.CONSTANT:
                    grounded, term = collected.subject_grounded, head.subject
                else:
                    grounded, term = collected.object_grounded, head.object_
                if term.entity_id is not None:
                    grounded[key][term.entity_id].append(metrics.confidence)
            case RuleType.AC2:
                continue
            case _ as unreachable:
                assert_never(unreachable)
    return collected


def _aggregate(confidences: Mapping[int, list[float]]) -> dict[int, float]:
    """Noisy-or the confidences of each grounded entity."""
    return {
        entity_id: aggregate_confidence(values)
        for entity_id, values in confidences.items()
    }
