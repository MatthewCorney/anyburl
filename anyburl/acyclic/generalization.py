"""Generalization of one-step acyclic walks into acyclic rules."""

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from ..core import EdgeTypeTuple
from ..rule.model import Atom, Rule, RuleType, Term

__all__ = ["AcyclicStep", "WalkAnchor", "generalize_acyclic"]

SUBJECT_VARIABLE: str = "X"
OBJECT_VARIABLE: str = "Y"
UNBOUND_VARIABLE: str = "A"


class WalkAnchor(StrEnum):
    """Which endpoint of a target triple an acyclic walk leaves from.

    Attributes
    ----------
    HEAD : str
        The walk leaves the head; the tail becomes the rule's constant.
    TAIL : str
        The walk leaves the tail; the head becomes the rule's constant.
    """

    HEAD = "head"
    TAIL = "tail"


@dataclass(frozen=True, slots=True)
class AcyclicStep:
    """One edge taken from an endpoint of a target triple.

    Parameters
    ----------
    target_edge_type : EdgeTypeTuple
        The target triple's ``(head_type, relation, tail_type)``.
    anchor : WalkAnchor
        The endpoint the step leaves from.
    constant_id : int
        The triple's other endpoint, which becomes the head constant.
    body_edge_type : EdgeTypeTuple
        The edge type taken; its source type is the anchor's node type.
    neighbor_id : int
        The entity the step reaches.
    """

    target_edge_type: EdgeTypeTuple
    anchor: WalkAnchor
    constant_id: int
    body_edge_type: EdgeTypeTuple
    neighbor_id: int


def generalize_acyclic(step: AcyclicStep) -> list[Rule]:
    """Generalize a step into its grounded and open acyclic rules.

    Leaving head ``h`` of ``r(h, t)`` by ``b`` to ``d`` yields
    ``r(X, t) :- b(X, d)`` and ``r(X, t) :- b(X, A)``; leaving the tail
    yields the mirror image with the head constant.

    Parameters
    ----------
    step : AcyclicStep
        The step to generalize.

    Returns
    -------
    list[Rule]
        The non-tautological rules, grounded first.
    """
    head_type, relation, tail_type = step.target_edge_type
    _, body_relation, neighbor_type = step.body_edge_type

    match step.anchor:
        case WalkAnchor.HEAD:
            free = Term.variable(SUBJECT_VARIABLE, node_type=head_type)
            head = Atom(
                relation=relation,
                subject=free,
                object_=Term.constant(step.constant_id, node_type=tail_type),
            )
        case WalkAnchor.TAIL:
            free = Term.variable(OBJECT_VARIABLE, node_type=tail_type)
            head = Atom(
                relation=relation,
                subject=Term.constant(step.constant_id, node_type=head_type),
                object_=free,
            )
        case _ as unreachable:
            assert_never(unreachable)

    ends = (
        Term.constant(step.neighbor_id, node_type=neighbor_type),
        Term.variable(UNBOUND_VARIABLE, node_type=neighbor_type),
    )
    rules = [
        Rule(
            head=head,
            body=(Atom(relation=body_relation, subject=free, object_=end),),
            rule_type=RuleType.ACYCLIC,
        )
        for end in ends
    ]
    return [rule for rule in rules if not rule.is_tautological]
