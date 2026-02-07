"""Horn rule representation, configuration, and generalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anyburl.enums import RuleType, TermKind
from anyburl.exceptions import RuleGeneralizationError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = ["Atom", "Rule", "RuleConfig", "RuleGeneralizer", "Term"]

# Variable name constants for rule generalization
SUBJECT_VARIABLE: str = "X"
OBJECT_VARIABLE: str = "Y"
INTERMEDIATE_VARIABLE_PREFIX: str = "Z"

# Default thresholds for rule quality
DEFAULT_MIN_SUPPORT: int = 2
DEFAULT_MIN_CONFIDENCE: float = 0.01
DEFAULT_MIN_HEAD_COVERAGE: float = 0.01


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
    ValueError
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
                    raise ValueError("VARIABLE term must have a name")
                if self.entity_id is not None:
                    raise ValueError("VARIABLE term must not have an entity_id")
            case TermKind.CONSTANT:
                if self.entity_id is None:
                    raise ValueError("CONSTANT term must have an entity_id")
                if self.name is not None:
                    raise ValueError("CONSTANT term must not have a name")

    @staticmethod
    def variable(name: str, *, node_type: str) -> Term:
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
    def constant(entity_id: int, *, node_type: str) -> Term:
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
        The second argument (target entity side). Named with trailing
        underscore to avoid shadowing the Python builtin.
    """

    relation: str
    subject: Term
    object_: Term

    def __str__(self) -> str:
        """Return human-readable representation like ``born_in(X, Y)``."""
        return f"{self.relation}({self.subject}, {self.object_})"


@dataclass(frozen=True, slots=True)
class Rule:
    """A Horn rule learned from the knowledge graph.

    A rule has the form ``head :- body_1, body_2, ..., body_n`` where
    the head and each body atom are :class:`Atom` instances.

    The body is a tuple of atoms whose topology is determined by shared
    variables. This naturally supports both linear chain rules and
    branching rules without requiring a special data structure.

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
        names: set[str] = set()
        for atom in (self.head, *self.body):
            if atom.subject.kind is TermKind.VARIABLE:
                names.add(str(atom.subject.name))
            if atom.object_.kind is TermKind.VARIABLE:
                names.add(str(atom.object_.name))
        return frozenset(names)

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


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Configuration for rule generalization and filtering.

    Parameters
    ----------
    min_support : int
        Minimum support threshold for a rule to be retained.
    min_confidence : float
        Minimum confidence threshold in [0.0, 1.0].
    min_head_coverage : float
        Minimum head coverage threshold in [0.0, 1.0].

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.min_support < 1:
            raise ValueError(f"min_support must be positive, got {self.min_support}")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}"
            )
        if not (0.0 <= self.min_head_coverage <= 1.0):
            raise ValueError(
                f"min_head_coverage must be in [0.0, 1.0], got {self.min_head_coverage}"
            )


# Path step: (entity_id, node_type, relation_traversed)
# The last step's relation connects to the tail entity of the target triple.
PathStep = tuple[int, str, str]


@dataclass(frozen=True, slots=True)
class _GeneralizationContext:
    """Internal context shared across rule generalization methods."""

    variable_map: dict[tuple[int, str], str]
    target_relation: str
    head_type: str
    tail_type: str


class RuleGeneralizer:
    """Generalizes concrete walk paths into variable-based Horn rules.

    Given a path (a sequence of concrete entity-relation steps), the
    generalizer replaces entities with variables to produce AC1, AC2,
    and cyclic rules.

    Parameters
    ----------
    config : RuleConfig
        Configuration controlling generalization behavior.
    """

    def __init__(self, config: RuleConfig) -> None:
        self._config = config

    def generalize(
        self,
        path: Sequence[PathStep],
        *,
        target_relation: str,
        head_type: str,
        tail_type: str,
    ) -> Sequence[Rule]:
        """Generalize a walk path into one or more Horn rules.

        A single path can produce multiple rules of different types
        (cyclic, AC1, AC2). The path represents a walk from the head
        entity to the tail entity of a target triple.

        Parameters
        ----------
        path : Sequence[PathStep]
            Concrete walk steps as ``(entity_id, node_type, relation)``
            tuples. The first step starts at the head entity; the walk
            ends at the tail entity (not included in the path).
        target_relation : str
            The relation of the target triple being learned.
        head_type : str
            Node type of the head entity in the target triple.
        tail_type : str
            Node type of the tail entity in the target triple.

        Returns
        -------
        Sequence[Rule]
            Generalized rules. May be empty if the path is too short.

        Raises
        ------
        RuleGeneralizationError
            If the path structure is fundamentally invalid.
        """
        if len(path) < 1:
            raise RuleGeneralizationError("Path must contain at least one step")

        ctx = _GeneralizationContext(
            variable_map=self._assign_variables(path, head_type=head_type),
            target_relation=target_relation,
            head_type=head_type,
            tail_type=tail_type,
        )
        rules: list[Rule] = []

        rules.append(self._create_cyclic_rule(path, ctx))

        for ground_subject in (True, False):
            rules.append(
                self._create_ac1_rule(path, ctx, ground_subject=ground_subject)
            )

        return rules

    @staticmethod
    def classify_rule(head: Atom, body: tuple[Atom, ...]) -> RuleType:
        """Determine the structural type of a rule.

        Parameters
        ----------
        head : Atom
            The rule head atom.
        body : tuple[Atom, ...]
            The rule body atoms.

        Returns
        -------
        RuleType
            AC1 if one head term is a constant, AC2 if a head variable
            is absent from the body, CYCLIC if both head variables
            appear in the body.
        """
        head_subject_is_const = head.subject.kind is TermKind.CONSTANT
        head_object_is_const = head.object_.kind is TermKind.CONSTANT

        if head_subject_is_const or head_object_is_const:
            return RuleType.AC1

        body_variables: set[str] = set()
        for atom in body:
            if atom.subject.kind is TermKind.VARIABLE:
                body_variables.add(str(atom.subject.name))
            if atom.object_.kind is TermKind.VARIABLE:
                body_variables.add(str(atom.object_.name))

        head_vars_in_body = sum(
            1
            for term in (head.subject, head.object_)
            if term.kind is TermKind.VARIABLE and term.name in body_variables
        )

        if head_vars_in_body >= 2:  # noqa: PLR2004
            return RuleType.CYCLIC
        return RuleType.AC2

    def _assign_variables(
        self,
        path: Sequence[PathStep],
        *,
        head_type: str,
    ) -> dict[tuple[int, str], str]:
        """Assign variable names to entities in a walk path.

        The head entity gets ``X``, the tail entity gets ``Y``, and
        intermediate entities get ``Z0``, ``Z1``, etc. Repeated
        entities receive the same variable.

        Parameters
        ----------
        path : Sequence[PathStep]
            The walk path steps.
        head_type : str
            Node type of the head entity.

        Returns
        -------
        dict[tuple[int, str], str]
            Mapping from ``(entity_id, node_type)`` to variable name.
        """
        variable_map: dict[tuple[int, str], str] = {}
        intermediate_counter = 0

        head_entity_id, _, _ = path[0]
        head_key = (head_entity_id, head_type)
        variable_map[head_key] = SUBJECT_VARIABLE

        tail_entity_id, tail_node_type, _ = path[-1]
        tail_key = (tail_entity_id, tail_node_type)
        if tail_key not in variable_map:
            variable_map[tail_key] = OBJECT_VARIABLE

        for entity_id, node_type, _ in path[1:-1]:
            key = (entity_id, node_type)
            if key not in variable_map:
                variable_map[key] = (
                    f"{INTERMEDIATE_VARIABLE_PREFIX}{intermediate_counter}"
                )
                intermediate_counter += 1

        return variable_map

    def _create_cyclic_rule(
        self,
        path: Sequence[PathStep],
        ctx: _GeneralizationContext,
    ) -> Rule:
        """Create a cyclic rule where both head variables appear in the body.

        Parameters
        ----------
        path : Sequence[PathStep]
            The walk path steps.
        ctx : _GeneralizationContext
            Shared generalization context.

        Returns
        -------
        Rule
            A cyclic rule.
        """
        head_atom = Atom(
            relation=ctx.target_relation,
            subject=Term.variable(SUBJECT_VARIABLE, node_type=ctx.head_type),
            object_=Term.variable(OBJECT_VARIABLE, node_type=ctx.tail_type),
        )

        body = self._build_body_atoms(path, ctx)
        rule_type = self.classify_rule(head_atom, body)

        return Rule(head=head_atom, body=body, rule_type=rule_type)

    def _create_ac1_rule(
        self,
        path: Sequence[PathStep],
        ctx: _GeneralizationContext,
        *,
        ground_subject: bool,
    ) -> Rule:
        """Create an AC1 rule with one constant in the head.

        Parameters
        ----------
        path : Sequence[PathStep]
            The walk path steps.
        ctx : _GeneralizationContext
            Shared generalization context.
        ground_subject : bool
            If ``True``, ground the subject (head entity) as a constant.
            If ``False``, ground the object (tail entity).

        Returns
        -------
        Rule
            An AC1 rule.
        """
        head_entity_id = path[0][0]
        tail_entity_id = path[-1][0]

        if ground_subject:
            subject = Term.constant(head_entity_id, node_type=ctx.head_type)
            object_ = Term.variable(OBJECT_VARIABLE, node_type=ctx.tail_type)
        else:
            subject = Term.variable(SUBJECT_VARIABLE, node_type=ctx.head_type)
            object_ = Term.constant(tail_entity_id, node_type=ctx.tail_type)

        head_atom = Atom(
            relation=ctx.target_relation,
            subject=subject,
            object_=object_,
        )

        body = self._build_body_atoms(path, ctx)
        rule_type = self.classify_rule(head_atom, body)

        return Rule(head=head_atom, body=body, rule_type=rule_type)

    def _build_body_atoms(
        self,
        path: Sequence[PathStep],
        ctx: _GeneralizationContext,
    ) -> tuple[Atom, ...]:
        """Build body atoms from consecutive path steps.

        Each consecutive pair of steps produces one body atom connecting
        the two entities via the relation from the first step.

        Parameters
        ----------
        path : Sequence[PathStep]
            The walk path steps.
        ctx : _GeneralizationContext
            Shared generalization context (provides variable_map).

        Returns
        -------
        tuple[Atom, ...]
            The body atoms.
        """
        atoms: list[Atom] = []
        for i in range(len(path) - 1):
            src_id, src_type, relation = path[i]
            dst_id, dst_type, _ = path[i + 1]

            src_var = ctx.variable_map[(src_id, src_type)]
            dst_var = ctx.variable_map[(dst_id, dst_type)]

            atom = Atom(
                relation=relation,
                subject=Term.variable(src_var, node_type=src_type),
                object_=Term.variable(dst_var, node_type=dst_type),
            )
            atoms.append(atom)

        return tuple(atoms)
