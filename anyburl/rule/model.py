"""Horn rule representation: terms, atoms and rules."""

from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import assert_never

from ..exceptions import InvalidRuleError

__all__ = ["Atom", "PathStep", "Rule", "RuleType", "Term", "TermKind"]

PathStep = tuple[int, str, str]
"""One walk step: ``(entity_id, node_type, relation_to_next_entity)``.

The final step of a path holds the tail entity and an empty relation.
"""


class RuleType(StrEnum):
    """Classification of learned Horn rules by structural type.

    Attributes
    ----------
    AC1 : str
        Acyclic rule with one free variable. One head variable is grounded
        as a constant; the other appears in the body.
    AC2 : str
        Acyclic rule with two free variables. One head variable does not
        appear in the body at all.
    CYCLIC : str
        Cyclic rule. Both head variables appear in the body, forming a
        closed variable chain.
    """

    AC1 = "ac1"
    AC2 = "ac2"
    CYCLIC = "cyclic"


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
        The structural classification (AC1, AC2, or CYCLIC).
    """

    head: Atom
    body: tuple[Atom, ...]
    rule_type: RuleType

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
        """Return whether a body atom is identical to the head atom.

        ``p(X, Y) :- p(X, Y)`` is tautological; ``p(X, Y) :- p(X, Z0)`` is
        not, because its body uses different variables.
        """
        return any(atom == self.head for atom in self.body)

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

        Example: ``born_in(X, Y) :- lives_in(X, Z0), located_in(Z0, Y)``
        """
        body_str = ", ".join(str(atom) for atom in self.body)
        return f"{self.head} :- {body_str}"
