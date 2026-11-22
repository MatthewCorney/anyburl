"""Rule and metrics builders shared across tests.

Every rule predicts ``lives_in(person, city)``; the two-atom rules use the
body ``born_in(X, Z0), near(Z0, Y)``.
"""

from anyburl.rule import Atom, Rule, RuleMetrics, RuleType, Term

BORN_IN_NEAR_BODY: tuple[Atom, ...] = (
    Atom(
        relation="born_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Z0", node_type="city"),
    ),
    Atom(
        relation="near",
        subject=Term.variable("Z0", node_type="city"),
        object_=Term.variable("Y", node_type="city"),
    ),
)


def lives_in(subject: Term, object_: Term) -> Atom:
    """Return a ``lives_in`` head atom."""
    return Atom(relation="lives_in", subject=subject, object_=object_)


def cyclic_born_in_near() -> Rule:
    """Return ``lives_in(X, Y) :- born_in(X, Z0), near(Z0, Y)``."""
    head = lives_in(
        Term.variable("X", node_type="person"), Term.variable("Y", node_type="city")
    )
    return Rule(head=head, body=BORN_IN_NEAR_BODY, rule_type=RuleType.CYCLIC)


def subject_grounded(person_id: int) -> Rule:
    """Return ``lives_in(person:<id>, Y) :- born_in(X, Z0), near(Z0, Y)``."""
    head = lives_in(
        Term.constant(person_id, node_type="person"),
        Term.variable("Y", node_type="city"),
    )
    return Rule(head=head, body=BORN_IN_NEAR_BODY, rule_type=RuleType.AC1)


def object_grounded(city_id: int) -> Rule:
    """Return ``lives_in(X, city:<id>) :- born_in(X, Z0), near(Z0, Y)``."""
    head = lives_in(
        Term.variable("X", node_type="person"),
        Term.constant(city_id, node_type="city"),
    )
    return Rule(head=head, body=BORN_IN_NEAR_BODY, rule_type=RuleType.AC1)


def ac2_born_in() -> Rule:
    """Return ``lives_in(X, Y) :- born_in(X, Z0)``, where ``Y`` is unbound."""
    head = lives_in(
        Term.variable("X", node_type="person"), Term.variable("Y", node_type="city")
    )
    return Rule(head=head, body=BORN_IN_NEAR_BODY[:1], rule_type=RuleType.AC2)


def metrics_with(confidence: float) -> RuleMetrics:
    """Return metrics whose only meaningful field is ``confidence``."""
    return RuleMetrics(
        support=1, confidence=confidence, head_coverage=0.25, num_predictions=4
    )


def object_constant_rule(city_id: int, born_in_city: int | None) -> Rule:
    """Return ``lives_in(X, city) :- born_in(X, d)``, or ``born_in(X, A)``."""
    end = (
        Term.variable("A", node_type="city")
        if born_in_city is None
        else Term.constant(born_in_city, node_type="city")
    )
    return Rule(
        head=Atom(
            relation="lives_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.constant(city_id, node_type="city"),
        ),
        body=(
            Atom(
                relation="born_in",
                subject=Term.variable("X", node_type="person"),
                object_=end,
            ),
        ),
        rule_type=RuleType.ACYCLIC,
    )


def subject_constant_near_rule(person_id: int) -> Rule:
    """Return ``lives_in(person, Y) :- near(Y, A)``."""
    return Rule(
        head=Atom(
            relation="lives_in",
            subject=Term.constant(person_id, node_type="person"),
            object_=Term.variable("Y", node_type="city"),
        ),
        body=(
            Atom(
                relation="near",
                subject=Term.variable("Y", node_type="city"),
                object_=Term.variable("A", node_type="city"),
            ),
        ),
        rule_type=RuleType.ACYCLIC,
    )
