"""Horn rule representation: terms, atoms and rules."""

from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import assert_never

from ..exceptions import InvalidRuleError

__all__ = [
    "Atom",
    "PathStep",
    "Rule",
    "RuleType",
    "Term",
    "TermKind",
    "classify_rule",
]

ACYCLIC_BODY_LENGTH: int = 1

PathStep = tuple[int, str, str]
"""One walk step: ``(entity_id, node_type, relation_to_next_entity)``.

The final step of a path holds the tail entity and an empty relation.
"""


class RuleType(StrEnum):
    """Classification of learned Horn rules by structural type.

    ``AC1`` and ``CYCLIC`` rules have all-variable body chains of any length.
    ``ACYCLIC`` covers AnyBURL's AC1 and AC2 rules, which have one body atom.
    Rules with a head variable absent from the body are not learned.

    Attributes
    ----------
    AC1 : str
        One head term is a constant; the other is a variable. The body is a
        chain of variables, as in ``r(X, c) :- b1(X, Z0), b2(Z0, Y)``.
    CYCLIC : str
        Both head variables appear in the body, closing the chain, as in
        ``r(X, Y) :- b1(X, Z0), b2(Z0, Y)``.
    ACYCLIC : str
        One head term is a constant and the body is a single atom leaving
        the free head variable, ending at a constant (``r(X, c) :- b(X, d)``,
        AnyBURL's AC1) or at an unbound variable (``r(X, c) :- b(X, A)``,
        AnyBURL's AC2).
    """

    AC1 = "ac1"
    CYCLIC = "cyclic"
    ACYCLIC = "acyclic"


class TermKind(Enum):
    """Whether a term in an atom is a variable or a grounded constant.

    Attributes
    ----------
    VARIABLE : int
        A placeholder that can bind to any entity during rule application.
    CONSTANT : int
        A specific, fixed entity from the knowledge graph.
    """

    VARIABLE = auto()
    CONSTANT = auto()


@dataclass(frozen=True, slots=True)
class Term:
    """A term in a logical atom — either a variable or a grounded constant.

    In a heterogeneous graph, the same integer entity ID can refer to
    different entities under different node types. The ``node_type`` field
    disambiguates this.

    Parameters
    ----------
    kind : TermKind
        Whether this term is a variable or a constant.
    node_type : str
        The node type this term refers to.
    name : str | None
        Variable name when ``kind`` is ``VARIABLE``, ``None`` otherwise.
    entity_id : int | None
        Grounded entity ID when ``kind`` is ``CONSTANT``, ``None`` otherwise.

    Raises
    ------
    InvalidRuleError
        If the combination of kind, name, and entity_id is invalid.
    """

    kind: TermKind
    node_type: str
    name: str | None = None
    entity_id: int | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one of name or entity_id is set."""
        match self.kind:
            case TermKind.VARIABLE:
                if self.name is None:
                    raise InvalidRuleError("VARIABLE term must have a name")
                if self.entity_id is not None:
                    raise InvalidRuleError("VARIABLE term must not have an entity_id")
            case TermKind.CONSTANT:
                if self.entity_id is None:
                    raise InvalidRuleError("CONSTANT term must have an entity_id")
                if self.name is not None:
                    raise InvalidRuleError("CONSTANT term must not have a name")
            case _ as unreachable:
                assert_never(unreachable)

    @staticmethod
    def variable(name: str, *, node_type: str) -> "Term":
        """Create a variable term.

        Parameters
        ----------
        name : str
            The variable name (e.g. "X", "Y", "Z0").
        node_type : str
            The node type this variable ranges over.

        Returns
        -------
        Term
            A term with ``kind=VARIABLE``.
        """
        return Term(kind=TermKind.VARIABLE, node_type=node_type, name=name)

    @staticmethod
    def constant(entity_id: int, *, node_type: str) -> "Term":
        """Create a constant (grounded) term.

        Parameters
        ----------
        entity_id : int
            The entity index in the knowledge graph.
        node_type : str
            The node type of this entity.

        Returns
        -------
        Term
            A term with ``kind=CONSTANT``.
        """
        return Term(kind=TermKind.CONSTANT, node_type=node_type, entity_id=entity_id)

    def __str__(self) -> str:
        """Return human-readable representation."""
        match self.kind:
            case TermKind.VARIABLE:
                return str(self.name)
            case TermKind.CONSTANT:
                return f"{self.node_type}:{self.entity_id}"
            case _ as unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class Atom:
    """A logical atom: a relation applied to a subject and object term.

    Represents predicates like ``born_in(X, Y)`` or ``lives_in(X, london)``.

    Parameters
    ----------
    relation : str
        The relation/predicate name.
    subject : Term
        The first argument (source entity side).
    object_ : Term
        The second argument (target entity side).
    """

    relation: str
    subject: Term
    object_: Term

    @property
    def edge_signature(self) -> tuple[str, str, str]:
        """Return the typed edge signature of this atom.

        Returns
        -------
        tuple[str, str, str]
            ``(subject_node_type, relation, object_node_type)`` triple that
            uniquely identifies the edge type in a heterogeneous graph.
        """
        return (self.subject.node_type, self.relation, self.object_.node_type)

    @property
    def variable_names(self) -> frozenset[str]:
        """Return the names of this atom's variable terms."""
        return frozenset(
            str(term.name)
            for term in (self.subject, self.object_)
            if term.kind is TermKind.VARIABLE
        )

    def __str__(self) -> str:
        """Return a representation like ``person_born_in_city(X, Y)``.

        Node types are included because one relation name may be shared by
        several edge types.
        """
        src_type = self.subject.node_type
        dst_type = self.object_.node_type
        return f"{src_type}_{self.relation}_{dst_type}({self.subject}, {self.object_})"


def classify_rule(head: Atom, body: tuple[Atom, ...]) -> RuleType | None:
    """Return the chain-rule type of ``head :- body``.

    Parameters
    ----------
    head : Atom
        The rule head atom.
    body : tuple[Atom, ...]
        The rule body atoms.

    Returns
    -------
    RuleType | None
        ``None`` if a head variable is absent from the body; otherwise AC1
        if a head term is a constant and CYCLIC if not.
    """
    body_variables = frozenset().union(*(atom.variable_names for atom in body))
    if not head.variable_names <= body_variables:
        return None
    if TermKind.CONSTANT in (head.subject.kind, head.object_.kind):
        return RuleType.AC1
    return RuleType.CYCLIC


def _admissible_rule_types(head: Atom, body: tuple[Atom, ...]) -> frozenset[RuleType]:
    """Return the rule types the structure ``head :- body`` admits.

    A single-atom AC1 structure may also be an acyclic rule, the only kind
    whose body holds constants.
    """
    chain_type = classify_rule(head, body)
    is_single_atom = len(body) == ACYCLIC_BODY_LENGTH
    has_body_constant = any(
        TermKind.CONSTANT in (atom.subject.kind, atom.object_.kind) for atom in body
    )
    match chain_type:
        case RuleType.AC1 if is_single_atom and has_body_constant:
            return frozenset({RuleType.ACYCLIC})
        case RuleType.AC1 if is_single_atom:
            return frozenset({RuleType.AC1, RuleType.ACYCLIC})
        case None:
            return frozenset()
        case _ if has_body_constant:
            return frozenset()
        case _:
            return frozenset({chain_type})


@dataclass(frozen=True, slots=True)
class Rule:
    """A Horn rule learned from the knowledge graph.

    A rule has the form ``head :- body_1, body_2, ..., body_n`` where
    the head and each body atom are :class:`Atom` instances.

    Bodies are linear chains: each intermediate variable joins two
    consecutive atoms, which is what evaluation and prediction assume.

    Parameters
    ----------
    head : Atom
        The consequent atom (what the rule predicts).
    body : tuple[Atom, ...]
        The antecedent atoms (conjunctive conditions).
    rule_type : RuleType
        The structural classification (AC1, CYCLIC or ACYCLIC).

    Raises
    ------
    InvalidRuleError
        If ``rule_type`` does not match the rule's structure.
    """

    head: Atom
    body: tuple[Atom, ...]
    rule_type: RuleType

    def __post_init__(self) -> None:
        """Reject a rule type the head and body do not admit."""
        if self.rule_type not in _admissible_rule_types(self.head, self.body):
            raise InvalidRuleError(
                f"Rule {self} does not have the structure of a "
                f"{self.rule_type.value} rule"
            )

    @property
    def length(self) -> int:
        """Return the number of atoms in the rule body."""
        return len(self.body)

    @property
    def variables(self) -> frozenset[str]:
        """Return all variable names appearing in the rule."""
        return frozenset().union(
            *(atom.variable_names for atom in (self.head, *self.body))
        )

    @property
    def is_tautological(self) -> bool:
        """Return whether the rule can only predict edges its body matches.

        ``p(X, Y) :- p(X, Y)`` repeats its head in its body, while
        ``p(X, Y) :- p(X, Z0)`` does not. An AC1 rule is grounded from its
        constant along its body's edge types, so ``p(c, Y) :- p(X, Y)``
        predicts exactly the known ``p`` edges of ``c`` and is tautological
        too.
        """
        if any(atom == self.head for atom in self.body):
            return True
        return (
            self.rule_type is RuleType.AC1
            and self.length == ACYCLIC_BODY_LENGTH
            and self.body[0].edge_signature == self.head.edge_signature
        )

    @property
    def constants(self) -> frozenset[tuple[int, str]]:
        """Return all constant (entity_id, node_type) pairs in the rule."""
        result: set[tuple[int, str]] = set()
        for atom in (self.head, *self.body):
            for term in (atom.subject, atom.object_):
                if term.kind is TermKind.CONSTANT and term.entity_id is not None:
                    result.add((term.entity_id, term.node_type))
        return frozenset(result)

    def __str__(self) -> str:
        """Return human-readable rule string.

        Example: ``person_lives_in_city(X, Y) :- person_born_in_city(X, Z0),
        city_near_city(Z0, Y)``
        """
        body_str = ", ".join(str(atom) for atom in self.body)
        return f"{self.head} :- {body_str}"
