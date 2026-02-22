"""Rule and metrics factory helpers for benchmark fixtures and tests."""

from anyburl.metrics import RuleMetrics
from anyburl.rule import Atom, Rule, RuleType, Term


def make_cyclic_rule() -> Rule:
    """Build: r_ac(X:A, Y:C) :- r_ab(X:A, Z0:B), r_bc(Z0:B, Y:C)."""
    head = Atom(
        relation="r_ac",
        subject=Term.variable("X", node_type="A"),
        object_=Term.variable("Y", node_type="C"),
    )
    body = (
        Atom(
            relation="r_ab",
            subject=Term.variable("X", node_type="A"),
            object_=Term.variable("Z0", node_type="B"),
        ),
        Atom(
            relation="r_bc",
            subject=Term.variable("Z0", node_type="B"),
            object_=Term.variable("Y", node_type="C"),
        ),
    )
    return Rule(head=head, body=body, rule_type=RuleType.CYCLIC)


def make_ac1_rule() -> Rule:
    """Build: r_ac(A:0, Y:C) :- r_ab(X:A, Z0:B), r_bc(Z0:B, Y:C)."""
    head = Atom(
        relation="r_ac",
        subject=Term.constant(0, node_type="A"),
        object_=Term.variable("Y", node_type="C"),
    )
    body = (
        Atom(
            relation="r_ab",
            subject=Term.variable("X", node_type="A"),
            object_=Term.variable("Z0", node_type="B"),
        ),
        Atom(
            relation="r_bc",
            subject=Term.variable("Z0", node_type="B"),
            object_=Term.variable("Y", node_type="C"),
        ),
    )
    return Rule(head=head, body=body, rule_type=RuleType.AC1)


def make_ac2_rule() -> Rule:
    """Build: r_ac(X:A, Y:C) :- r_ab(X:A, Z0:B)  -- Y:C absent from body."""
    head = Atom(
        relation="r_ac",
        subject=Term.variable("X", node_type="A"),
        object_=Term.variable("Y", node_type="C"),
    )
    body = (
        Atom(
            relation="r_ab",
            subject=Term.variable("X", node_type="A"),
            object_=Term.variable("Z0", node_type="B"),
        ),
    )
    return Rule(head=head, body=body, rule_type=RuleType.AC2)


def make_metrics(confidence: float = 0.5) -> RuleMetrics:
    """Create a RuleMetrics instance with the given confidence."""
    return RuleMetrics(
        support=10,
        confidence=confidence,
        head_coverage=0.1,
        num_predictions=20,
    )
