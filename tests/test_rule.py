"""Tests for Horn rule representation, configuration, and generalization."""

import pytest

from anyburl.rule import (
    Atom,
    Rule,
    RuleConfig,
    RuleGeneralizer,
    RuleType,
    Term,
    TermKind,
)


def test_term_variable_kind() -> None:
    term = Term.variable("X", node_type="person")
    assert term.kind is TermKind.VARIABLE
    assert term.name == "X"
    assert term.entity_id is None
    assert term.node_type == "person"


def test_term_constant_kind() -> None:
    term = Term.constant(42, node_type="city")
    assert term.kind is TermKind.CONSTANT
    assert term.entity_id == 42
    assert term.name is None
    assert term.node_type == "city"


def test_term_variable_str() -> None:
    assert str(Term.variable("X", node_type="person")) == "X"


def test_term_constant_str() -> None:
    assert str(Term.constant(3, node_type="city")) == "city:3"


def test_term_variable_with_entity_id_raises() -> None:
    with pytest.raises(ValueError, match="must not have an entity_id"):
        Term(kind=TermKind.VARIABLE, node_type="person", name="X", entity_id=0)


def test_term_constant_with_name_raises() -> None:
    with pytest.raises(ValueError, match="must not have a name"):
        Term(kind=TermKind.CONSTANT, node_type="person", entity_id=0, name="X")


def test_atom_edge_signature() -> None:
    atom = Atom(
        relation="born_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    assert atom.edge_signature == ("person", "born_in", "city")


def test_atom_str() -> None:
    atom = Atom(
        relation="born_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    assert str(atom) == "person_born_in_city(X, Y)"


def test_rule_length() -> None:
    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
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
    rule = Rule(head=head, body=body, rule_type=RuleType.CYCLIC)
    assert rule.length == 2


def test_rule_variables() -> None:
    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
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
    rule = Rule(head=head, body=body, rule_type=RuleType.CYCLIC)
    assert rule.variables == frozenset({"X", "Y", "Z0"})


def test_rule_constants() -> None:
    head = Atom(
        relation="lives_in",
        subject=Term.constant(0, node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
        Atom(
            relation="born_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.variable("Z0", node_type="city"),
        ),
    )
    rule = Rule(head=head, body=body, rule_type=RuleType.AC1)
    assert rule.constants == frozenset({(0, "person")})


def test_rule_str_contains_key_parts() -> None:
    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
        Atom(
            relation="born_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.variable("Z0", node_type="city"),
        ),
    )
    rule = Rule(head=head, body=body, rule_type=RuleType.CYCLIC)
    rule_str = str(rule)
    assert "lives_in" in rule_str
    assert "born_in" in rule_str
    assert ":-" in rule_str


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_support": 0}, "min_support must be positive"),
        ({"min_confidence": -0.1}, "min_confidence must be in"),
        ({"min_confidence": 1.1}, "min_confidence must be in"),
        ({"min_head_coverage": -0.1}, "min_head_coverage must be in"),
    ],
)
def test_rule_config_invalid(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RuleConfig(**kwargs)  # type: ignore[arg-type]


def test_rule_config_valid() -> None:
    config = RuleConfig(min_support=2, min_confidence=0.1, min_head_coverage=0.1)
    assert config.min_support == 2
    assert config.min_confidence == 0.1
    assert config.min_head_coverage == 0.1


def test_classify_rule_cyclic() -> None:
    head = Atom(
        relation="r",
        subject=Term.variable("X", node_type="A"),
        object_=Term.variable("Y", node_type="B"),
    )
    body = (
        Atom(
            relation="r1",
            subject=Term.variable("X", node_type="A"),
            object_=Term.variable("Z", node_type="C"),
        ),
        Atom(
            relation="r2",
            subject=Term.variable("Z", node_type="C"),
            object_=Term.variable("Y", node_type="B"),
        ),
    )
    assert RuleGeneralizer.classify_rule(head, body) == RuleType.CYCLIC


def test_classify_rule_ac1_subject_constant() -> None:
    head = Atom(
        relation="r",
        subject=Term.constant(0, node_type="A"),
        object_=Term.variable("Y", node_type="B"),
    )
    body = (
        Atom(
            relation="r1",
            subject=Term.variable("X", node_type="A"),
            object_=Term.variable("Y", node_type="B"),
        ),
    )
    assert RuleGeneralizer.classify_rule(head, body) == RuleType.AC1


def test_classify_rule_ac2_disconnected_head_var() -> None:
    head = Atom(
        relation="r",
        subject=Term.variable("X", node_type="A"),
        object_=Term.variable("Y", node_type="B"),
    )
    body = (
        Atom(
            relation="r1",
            subject=Term.variable("X", node_type="A"),
            object_=Term.variable("Z", node_type="C"),
        ),
    )
    assert RuleGeneralizer.classify_rule(head, body) == RuleType.AC2


def test_generalize_empty_path_raises() -> None:
    config = RuleConfig()
    generalizer = RuleGeneralizer(config)
    with pytest.raises(ValueError, match="at least one step"):
        generalizer.generalize([], target_relation="r", head_type="A", tail_type="B")


def test_generalize_2step_path_produces_three_rules() -> None:
    config = RuleConfig()
    generalizer = RuleGeneralizer(config)
    path = [(0, "person", "born_in"), (1, "city", "near"), (2, "city", "")]

    rules = generalizer.generalize(
        path,
        target_relation="lives_in",
        head_type="person",
        tail_type="city",
    )

    assert len(rules) == 3
    rule_types = {r.rule_type for r in rules}
    assert RuleType.CYCLIC in rule_types
    assert RuleType.AC1 in rule_types


def test_generalize_1step_same_relation_filters_tautological() -> None:
    """1-step path with target relation yields a tautological cyclic rule, filtered."""
    config = RuleConfig()
    generalizer = RuleGeneralizer(config)
    path = [(0, "person", "lives_in"), (1, "city", "")]

    rules = generalizer.generalize(
        path,
        target_relation="lives_in",
        head_type="person",
        tail_type="city",
    )

    assert all(r.rule_type == RuleType.AC1 for r in rules)
    assert len(rules) == 2
