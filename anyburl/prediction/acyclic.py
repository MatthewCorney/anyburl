"""Firing acyclic rules for link prediction queries."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ..graph import EdgeTypeTuple
from ..metrics import RuleMetrics
from ..metrics.acyclic import AcyclicIndex, acyclic_constant
from ..rule import Atom, Rule, RuleType
from .scoring import corrected_confidence

__all__ = ["AcyclicScorer", "Firings"]

Firings = tuple[Tensor, Tensor]
"""Parallel ``(candidate_ids, confidences)``, one entry per rule firing."""

_Targets = list[tuple[int, float]]


def _as_firings(targets: _Targets) -> Firings:
    """Convert ``(candidate, confidence)`` pairs into firing tensors."""
    candidates, confidences = zip(*targets, strict=True)
    return (
        torch.tensor(candidates, dtype=torch.long),
        torch.tensor(confidences, dtype=torch.float),
    )


def _concat(parts: Sequence[Firings]) -> Firings:
    """Concatenate firings, empty when there are none."""
    if not parts:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


@dataclass(slots=True)
class _RuleSet:
    """Acyclic rules whose head constant sits at the same position."""

    open_rules: dict[EdgeTypeTuple, _Targets] = field(
        default_factory=lambda: defaultdict(list)
    )
    grounded_rules: dict[EdgeTypeTuple, dict[int, _Targets]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    by_constant: dict[int, list[tuple[Atom, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, constant_id: int, body: Atom, confidence: float) -> None:
        """Index one rule by its body and by its head constant."""
        edge_type = body.edge_signature
        neighbor = body.object_.entity_id
        if neighbor is None:
            self.open_rules[edge_type].append((constant_id, confidence))
        else:
            self.grounded_rules[edge_type][neighbor].append((constant_id, confidence))
        self.by_constant[constant_id].append((body, confidence))

    def freeze(self) -> "_FrozenRuleSet":
        """Return the rules with their predicted constants as tensors."""
        return _FrozenRuleSet(
            open_rules={
                edge_type: _as_firings(targets)
                for edge_type, targets in self.open_rules.items()
            },
            grounded_rules={
                edge_type: {
                    neighbor: _as_firings(targets)
                    for neighbor, targets in by_neighbor.items()
                }
                for edge_type, by_neighbor in self.grounded_rules.items()
            },
            by_constant=dict(self.by_constant),
        )


@dataclass(frozen=True, slots=True)
class _FrozenRuleSet:
    """A :class:`_RuleSet` ready to fire."""

    open_rules: dict[EdgeTypeTuple, Firings]
    grounded_rules: dict[EdgeTypeTuple, dict[int, Firings]]
    by_constant: dict[int, list[tuple[Atom, float]]]

    def free_firings(self, entity: int, index: AcyclicIndex) -> list[Firings]:
        """Fire rules whose body ``entity`` satisfies; each predicts its constant."""
        fired = [
            firings
            for edge_type, firings in self.open_rules.items()
            if bool(index.has_outgoing(edge_type)[entity])
        ]
        for edge_type, by_neighbor in self.grounded_rules.items():
            for neighbor in np.unique(index.targets_from(edge_type, entity)).tolist():
                firings = by_neighbor.get(neighbor)
                if firings is not None:
                    fired.append(firings)
        return fired

    def constant_firings(self, entity: int, index: AcyclicIndex) -> list[Firings]:
        """Fire rules with constant ``entity``; each predicts its body's matches."""
        fired: list[Firings] = []
        for body, confidence in self.by_constant.get(entity, []):
            candidates = torch.from_numpy(index.satisfying(body))
            fired.append((candidates, torch.full((candidates.numel(),), confidence)))
        return fired


class AcyclicScorer:
    """Fires acyclic rules for tail and head queries.

    Parameters
    ----------
    index : AcyclicIndex
        Body lookups over the graph being predicted on.
    results : Sequence[tuple[Rule, RuleMetrics]]
        Rules with their metrics; rules of other types are ignored.
    unseen_negatives : int
        Assumed incorrect predictions added to each rule's denominator.
    """

    def __init__(
        self,
        index: AcyclicIndex,
        results: Sequence[tuple[Rule, RuleMetrics]],
        unseen_negatives: int,
    ) -> None:
        self._index = index
        object_constant, subject_constant = _RuleSet(), _RuleSet()
        for rule, metrics in results:
            if rule.rule_type is not RuleType.ACYCLIC:
                continue
            is_object_constant, constant_id = acyclic_constant(rule)
            rule_set = object_constant if is_object_constant else subject_constant
            rule_set.add(
                constant_id,
                rule.body[0],
                corrected_confidence(metrics, unseen_negatives),
            )
        self._object_constant = object_constant.freeze()
        self._subject_constant = subject_constant.freeze()

    def tail_firings(self, head_id: int) -> Firings:
        """Return the firings scoring candidate tails of ``head_id``."""
        return _concat(
            self._object_constant.free_firings(head_id, self._index)
            + self._subject_constant.constant_firings(head_id, self._index)
        )

    def head_firings(self, tail_id: int) -> Firings:
        """Return the firings scoring candidate heads of ``tail_id``."""
        return _concat(
            self._subject_constant.free_firings(tail_id, self._index)
            + self._object_constant.constant_firings(tail_id, self._index)
        )
