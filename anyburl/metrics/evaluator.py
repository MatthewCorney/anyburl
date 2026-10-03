"""Rule quality evaluation by counting body-chain groundings."""

from collections import defaultdict
from collections.abc import Sequence
from typing import assert_never

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .._logging import get_logger
from ..chain import ChainScanner
from ..exceptions import GraphSchemaError
from ..graph import EdgeTypeTuple, HeteroGraph
from ..rule import Atom, Rule, RuleConfig, RuleType, TermKind
from .coverage_warning import HeadCoverageWarning
from .rule_metrics import NO_PREDICTIONS, RuleMetrics, metrics_from_counts

__all__ = ["BodySignature", "RuleEvaluator"]

BodySignature = tuple[EdgeTypeTuple, ...]
"""Ordered edge types of a rule body; shared by rules with the same chain."""

logger = get_logger(__name__)


class RuleEvaluator:
    """Evaluates rule quality by counting body-chain groundings.

    Computes support, confidence, and head coverage by walking each rule's
    body chain over the graph and comparing what it reaches against the
    head relation's known triples.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to evaluate rules against.
    config : RuleConfig
        Evaluation configuration with quality thresholds.
    """

    def __init__(self, graph: HeteroGraph, config: RuleConfig) -> None:
        self._graph = graph
        self._config = config
        self._edge_type_set: frozenset[EdgeTypeTuple] = frozenset(graph.edge_types)
        self._coverage_warning = HeadCoverageWarning(config)
        self._scanner = ChainScanner(graph)

    def evaluate(self, rule: Rule) -> RuleMetrics:
        """Compute quality metrics for a single rule.

        Parameters
        ----------
        rule : Rule
            The rule to evaluate.

        Returns
        -------
        RuleMetrics
            The computed metrics.
        """
        match rule.rule_type:
            case RuleType.CYCLIC:
                return self._evaluate_cyclic(rule)
            case RuleType.AC1:
                return self._evaluate_ac1(rule)
            case RuleType.AC2:
                return self._evaluate_ac2(rule)
            case _ as unreachable:
                assert_never(unreachable)

    def evaluate_batch(
        self,
        rules: Sequence[Rule],
        *,
        max_results: int | None = None,
    ) -> list[tuple[Rule, RuleMetrics]]:
        """Evaluate multiple rules, returning those that pass thresholds.

        Parameters
        ----------
        rules : Sequence[Rule]
            The rules to evaluate.
        max_results : int | None
            Stop after collecting this many passing rules. ``None``
            evaluates all rules.

        Returns
        -------
        list[tuple[Rule, RuleMetrics]]
            Passing rules paired with their metrics, in input order.
        """
        metrics_by_rule = self._compute_all_metrics(rules)

        results: list[tuple[Rule, RuleMetrics]] = []
        for rule in rules:
            if self._passes(rule, metrics_by_rule[rule]):
                results.append((rule, metrics_by_rule[rule]))
                if max_results is not None and len(results) >= max_results:
                    break
        logger.debug(
            "Evaluated %d rules, %d passed thresholds", len(rules), len(results)
        )
        self._coverage_warning.check(rules, metrics_by_rule)
        return results

    def _passes(self, rule: Rule, metrics: RuleMetrics) -> bool:
        """Return whether ``metrics`` clear the floors for ``rule``'s type."""
        thresholds = self._config.thresholds_for(rule.rule_type)
        return metrics.passes_thresholds(
            min_support=thresholds.min_support,
            min_confidence=thresholds.min_confidence,
            min_head_coverage=thresholds.min_head_coverage,
        )

    def _compute_all_metrics(
        self,
        rules: Sequence[Rule],
    ) -> dict[Rule, RuleMetrics]:
        """Compute metrics for every rule, grouping AC1 rules by body chain.

        AC1 rules sharing a body chain and grounding side are scanned in one
        kernel call. Other rule types are evaluated individually.

        Parameters
        ----------
        rules : Sequence[Rule]
            The rules to evaluate.

        Returns
        -------
        dict[Rule, RuleMetrics]
            Metrics keyed by rule.
        """
        metrics_by_rule: dict[Rule, RuleMetrics] = {}
        ac1_rules = [r for r in rules if r.rule_type is RuleType.AC1]
        other_rules = [r for r in rules if r.rule_type is not RuleType.AC1]

        for rule in tqdm(other_rules, desc="Evaluating rules", disable=not other_rules):
            if rule not in metrics_by_rule:
                metrics_by_rule[rule] = self.evaluate(rule)

        self._evaluate_ac1_groups(ac1_rules, metrics_by_rule)
        return metrics_by_rule

    def _evaluate_ac1_groups(
        self,
        rules: Sequence[Rule],
        out: dict[Rule, RuleMetrics],
    ) -> None:
        """Group AC1 rules by body chain and grounding side, then evaluate.

        Parameters
        ----------
        rules : Sequence[Rule]
            The AC1 rules to evaluate.
        out : dict[Rule, RuleMetrics]
            Destination mapping, updated in place.
        """
        groups: dict[tuple[BodySignature, bool], list[Rule]] = defaultdict(list)
        for rule in rules:
            groups[(self._body_signature(rule), _is_subject_grounded(rule))].append(
                rule
            )

        for (_, is_subject_grounded), group in tqdm(
            groups.items(), desc="Evaluating AC1 groups", disable=not groups
        ):
            self._evaluate_ac1_group(
                group, is_subject_grounded=is_subject_grounded, out=out
            )

    def _evaluate_ac1_group(
        self,
        rules: list[Rule],
        *,
        is_subject_grounded: bool,
        out: dict[Rule, RuleMetrics],
    ) -> None:
        """Evaluate one AC1 group sharing a body chain and grounding side.

        Every grounded entity in the group is scanned in a single kernel
        call per head relation.

        Parameters
        ----------
        rules : list[Rule]
            Rules in the group (non-empty).
        is_subject_grounded : bool
            Whether the grounded head term is the subject.
        out : dict[Rule, RuleMetrics]
            Destination mapping, updated in place.
        """
        pending: dict[EdgeTypeTuple, dict[int, list[Rule]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rule in rules:
            term = rule.head.subject if is_subject_grounded else rule.head.object_
            if term.entity_id is None:
                out[rule] = NO_PREDICTIONS
                continue
            pending[self._edge_type_of(rule.head)][term.entity_id].append(rule)

        signature = self._body_signature(rules[0])
        for head_et, by_entity in pending.items():
            entities = sorted(by_entity)
            predictions, support = self._scan_ac1(
                signature,
                head_et,
                np.array(entities, dtype=np.int64),
                is_subject_grounded=is_subject_grounded,
            )
            total_head_triples = self._graph.edge_count(head_et)
            for position, entity_id in enumerate(entities):
                metrics = metrics_from_counts(
                    int(predictions[position]),
                    int(support[position]),
                    total_head_triples,
                )
                for rule in by_entity[entity_id]:
                    out[rule] = metrics

    def _scan_ac1(
        self,
        signature: BodySignature,
        head_et: EdgeTypeTuple,
        entities: NDArray[np.int64],
        *,
        is_subject_grounded: bool,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Scan from each grounded entity, backwards for object-grounded rules."""
        if is_subject_grounded:
            return self._scanner.scan_rows(signature, head_et, entities)
        return self._scanner.scan_reversed_rows(signature, head_et, entities)

    def _evaluate_cyclic(self, rule: Rule) -> RuleMetrics:
        """Evaluate a cyclic rule by counting groundings over every source."""
        head_et = self._edge_type_of(rule.head)
        num_predictions, support = self._scanner.scan_all_rows(
            self._body_signature(rule), head_et
        )
        return metrics_from_counts(
            num_predictions, support, self._graph.edge_count(head_et)
        )

    def _evaluate_ac1(self, rule: Rule) -> RuleMetrics:
        """Evaluate a single AC1 rule through the grouped path.

        A subject-grounded rule ``h(person:0, Y) :- b1(X, Z0), ...`` is
        grounded from ``person:0``; an object-grounded rule is grounded
        backwards from its constant tail.
        """
        out: dict[Rule, RuleMetrics] = {}
        self._evaluate_ac1_group(
            [rule], is_subject_grounded=_is_subject_grounded(rule), out=out
        )
        return out[rule]

    def _evaluate_ac2(self, rule: Rule) -> RuleMetrics:
        """Evaluate an AC2 rule, where one head variable is absent from the body.

        Predictions are every reachable binding of the connected head
        variable paired with every entity of the disconnected one's type.
        """
        head = rule.head
        body_variables = frozenset().union(*(atom.variable_names for atom in rule.body))
        is_subject_connected = head.subject.name in body_variables
        if is_subject_connected == (head.object_.name in body_variables):
            logger.debug("AC2 rule has unexpected variable structure: %s", rule)
            return NO_PREDICTIONS

        signature = self._body_signature(rule)
        head_et = self._edge_type_of(head)
        head_sources, head_targets = self._graph.edge_index(head_et)
        if is_subject_connected:
            connected = self._scanner.reachable_sources(signature)
            known_connected = head_sources
            disconnected_type = head.object_.node_type
        else:
            connected = self._scanner.reachable_targets(signature)
            known_connected = head_targets
            disconnected_type = head.subject.node_type

        num_predictions = int(connected.sum()) * self._graph.node_count(
            disconnected_type
        )
        support = int(connected[known_connected.numpy()].sum())
        return metrics_from_counts(
            num_predictions, support, self._graph.edge_count(head_et)
        )

    def _body_signature(self, rule: Rule) -> BodySignature:
        """Return the ordered edge types of a rule's body chain."""
        return tuple(self._edge_type_of(atom) for atom in rule.body)

    def _edge_type_of(self, atom: Atom) -> EdgeTypeTuple:
        """Return the graph edge type an atom refers to.

        Raises
        ------
        GraphSchemaError
            If the graph has no matching edge type.
        """
        edge_type = atom.edge_signature
        if edge_type not in self._edge_type_set:
            raise GraphSchemaError(f"No edge type matches atom: {edge_type!r}")
        return edge_type


def _is_subject_grounded(rule: Rule) -> bool:
    """Return whether an AC1 rule's constant is its head subject."""
    return rule.head.subject.kind is TermKind.CONSTANT
