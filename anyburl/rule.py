"""Horn rule representation, configuration, and generalization."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from typing import assert_never

from .exceptions import ConfigurationError, InvalidRuleError

SUBJECT_VARIABLE: str = "X"
OBJECT_VARIABLE: str = "Y"
INTERMEDIATE_VARIABLE_PREFIX: str = "Z"

DEFAULT_MIN_SUPPORT: int = 2
DEFAULT_MIN_CONFIDENCE: float = 0.01
DEFAULT_MIN_HEAD_COVERAGE: float = 0.01

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


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    """Quality floors a rule must clear to be retained.

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
    ConfigurationError
        If any parameter is out of its valid range.
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.min_support < 1:
            raise ConfigurationError(
                f"min_support must be positive, got {self.min_support}"
            )
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ConfigurationError(
                f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}"
            )
        if not (0.0 <= self.min_head_coverage <= 1.0):
            raise ConfigurationError(
                f"min_head_coverage must be in [0.0, 1.0], got {self.min_head_coverage}"
            )


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Configuration for rule generalization and filtering.

    The top-level floors apply to every rule type unless ``per_type``
    overrides them. Head coverage in particular is not comparable across
    types: an AC1 rule is pinned to one entity, so its support cannot exceed
    that entity's degree, and it usually needs a much lower floor than a
    cyclic rule.

    Parameters
    ----------
    min_support : int
        Default minimum support threshold.
    min_confidence : float
        Default minimum confidence threshold in [0.0, 1.0].
    min_head_coverage : float
        Default minimum head coverage threshold in [0.0, 1.0].
    per_type : Mapping[RuleType, RuleThresholds]
        Floors that replace the defaults for the named rule types.

    Raises
    ------
    ConfigurationError
        If any default parameter is out of its valid range.
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_head_coverage: float = DEFAULT_MIN_HEAD_COVERAGE
    per_type: Mapping[RuleType, RuleThresholds] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self.default_thresholds()

    def default_thresholds(self) -> RuleThresholds:
        """Return the floors applied to rule types without an override."""
        return RuleThresholds(
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            min_head_coverage=self.min_head_coverage,
        )

    def thresholds_for(self, rule_type: RuleType) -> RuleThresholds:
        """Return the floors a rule of this type must clear.

        Parameters
        ----------
        rule_type : RuleType
            The rule type being filtered.

        Returns
        -------
        RuleThresholds
            The override for this type, else the defaults.
        """
        override = self.per_type.get(rule_type)
        if override is not None:
            return override
        return self.default_thresholds()


@dataclass(frozen=True, slots=True)
class _GeneralizationContext:
    """Internal context shared across rule generalization methods."""

    variable_map: dict[int, str]
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
            tuples. The first step describes the head entity and the
            relation used to leave it. The final step is a sentinel
            containing the tail entity with an empty relation string.
            Path length equals the number of body atoms plus one.
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
        InvalidRuleError
            If the path is empty.
        """
        if not path:
            raise InvalidRuleError("Path must contain at least one step")

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

        return [r for r in rules if not r.is_tautological]

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
        if TermKind.CONSTANT in (head.subject.kind, head.object_.kind):
            return RuleType.AC1

        body_variables = frozenset().union(*(atom.variable_names for atom in body))
        if head.variable_names <= body_variables:
            return RuleType.CYCLIC
        return RuleType.AC2

    def _assign_variables(
        self,
        path: Sequence[PathStep],
        *,
        head_type: str,
    ) -> dict[int, str]:
        """Assign a canonical variable name to each position in a walk path.

        Variables are keyed by **position**, not by entity identity: the
        first step gets ``X``, the last gets ``Y``, and every intermediate
        position gets a fresh ``Z0``, ``Z1``, ... even where the walk
        revisits an entity.

        Parameters
        ----------
        path : Sequence[PathStep]
            The walk path steps.
        head_type : str
            Node type of the head entity.

        Returns
        -------
        dict[int, str]
            Mapping from path position to variable name.
        """
        last_position = len(path) - 1
        variable_map: dict[int, str] = {0: SUBJECT_VARIABLE}

        if last_position > 0:
            tail_entity_id, tail_node_type, _ = path[last_position]
            returns_to_head = (
                tail_entity_id == path[0][0] and tail_node_type == head_type
            )
            variable_map[last_position] = (
                SUBJECT_VARIABLE if returns_to_head else OBJECT_VARIABLE
            )

        for position in range(1, last_position):
            variable_map[position] = f"{INTERMEDIATE_VARIABLE_PREFIX}{position - 1}"

        return variable_map

    def _create_cyclic_rule(
        self,
        path: Sequence[PathStep],
        ctx: _GeneralizationContext,
    ) -> Rule:
        """Create a fully-variable-head rule from the walk path.

        The rule is ``CYCLIC`` when both head variables appear in the body,
        and ``AC2`` when the walk returns to its head entity, leaving ``Y``
        absent from the body.

        Parameters
        ----------
        path : Sequence[PathStep]
            The walk path steps.
        ctx : _GeneralizationContext
            Shared generalization context.

        Returns
        -------
        Rule
            A rule with variable head; type determined by
            :meth:`classify_rule`.
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
            _, src_type, relation = path[i]
            _, dst_type, _ = path[i + 1]

            src_var = ctx.variable_map[i]
            dst_var = ctx.variable_map[i + 1]

            atom = Atom(
                relation=relation,
                subject=Term.variable(src_var, node_type=src_type),
                object_=Term.variable(dst_var, node_type=dst_type),
            )
            atoms.append(atom)

        return tuple(atoms)
