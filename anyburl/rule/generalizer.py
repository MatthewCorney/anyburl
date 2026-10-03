"""Generalization of concrete walk paths into Horn rules."""

from collections.abc import Sequence
from dataclasses import dataclass

from ..exceptions import InvalidRuleError
from .config import RuleConfig
from .model import Atom, PathStep, Rule, RuleType, Term, TermKind

__all__ = ["RuleGeneralizer"]

SUBJECT_VARIABLE: str = "X"
OBJECT_VARIABLE: str = "Y"
INTERMEDIATE_VARIABLE_PREFIX: str = "Z"


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
