"""Rule quality evaluation: support, confidence, and head coverage."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

import numpy as np
from tqdm import tqdm

from ._chain_scan import ChainScanner
from ._logging import get_logger
from .graph import EdgeTypeTuple, HeteroGraph
from .rule import Atom, Rule, RuleConfig, RuleType, TermKind

BodySignature = tuple[EdgeTypeTuple, ...]
"""Ordered edge types of a rule body; shared by rules with the same chain."""

logger = get_logger(__name__)


ZERO_CONFIDENCE: float = 0.0
ZERO_HEAD_COVERAGE: float = 0.0


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    """Computed quality metrics for a single rule.

    Parameters
    ----------
    support : int
        Number of known triples correctly predicted by the rule.
    confidence : float
        ``support / (support + incorrect_predictions)``.
        In ``[0.0, 1.0]``. Higher is better.
    head_coverage : float
        ``support / total_triples_with_head_relation``.
        In ``[0.0, 1.0]``. Higher is better.
    num_predictions : int
        Total predictions made by the rule (correct + incorrect).
    """

    support: int
    confidence: float
    head_coverage: float
    num_predictions: int

    @property
    def is_trivial(self) -> bool:
        """Return ``True`` if the rule makes zero predictions."""
        return self.num_predictions == 0

    def passes_thresholds(
        self,
        *,
        min_support: int,
        min_confidence: float,
        min_head_coverage: float,
    ) -> bool:
        """Check whether this rule meets all quality thresholds.

        Parameters
        ----------
        min_support : int
            Minimum required support.
        min_confidence : float
            Minimum required confidence.
        min_head_coverage : float
            Minimum required head coverage.

        Returns
        -------
        bool
            ``True`` if all thresholds are met.
        """
        return (
            self.support >= min_support
            and self.confidence >= min_confidence
            and self.head_coverage >= min_head_coverage
        )


def aggregate_confidence(confidences: Sequence[float]) -> float:
    """Aggregate confidences from multiple rules via the noisy-or formula.

    When several rules predict the same triple, their individual
    confidences are combined as::

        conf_agg = 1 - prod(1 - c_i)

    This treats each rule as an independent "chance" of the triple
    being true.

    Parameters
    ----------
    confidences : Sequence[float]
        Individual rule confidences, each in ``[0.0, 1.0]``.

    Returns
    -------
    float
        Aggregated confidence in ``[0.0, 1.0]``.
        Returns ``0.0`` for an empty sequence.
    """
    result = 1.0
    for c in confidences:
        result *= 1.0 - c
    return 1.0 - result


def _ac1_metrics(
    num_predictions: int,
    support: int,
    total_head_triples: int,
) -> RuleMetrics:
    """Build AC1 metrics from one grounded entity's counts.

    Parameters
    ----------
    num_predictions : int
        Distinct entities the body chain reaches from the constant.
    support : int
        How many of those the head relation already connects.
    total_head_triples : int
        Triples carrying the head relation, for head coverage.

    Returns
    -------
    RuleMetrics
        Computed metrics.
    """
    if num_predictions == 0:
        return RuleMetrics(
            support=0,
            confidence=ZERO_CONFIDENCE,
            head_coverage=ZERO_HEAD_COVERAGE,
            num_predictions=0,
        )
    return RuleMetrics(
        support=support,
        confidence=support / num_predictions,
        head_coverage=(
            support / total_head_triples
            if total_head_triples > 0
            else ZERO_HEAD_COVERAGE
        ),
        num_predictions=num_predictions,
    )


class RuleEvaluator:
    """Evaluates rule quality using sparse CSR matmul against a graph.

    Computes support, confidence, and head coverage for rules by
    multiplying body atom adjacency matrices and comparing against
    the head relation's known triples.

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
        self._warned_types: set[RuleType] = set()
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
            (the default) evaluates all rules.

        Returns
        -------
        list[tuple[Rule, RuleMetrics]]
            Rules paired with their metrics, filtered to only those
            passing the configured thresholds.
        """
        metrics_by_rule = self._compute_all_metrics(rules)

        results: list[tuple[Rule, RuleMetrics]] = []
        for rule in rules:
            metrics = metrics_by_rule.get(rule)
            if metrics is None:
                continue
            thresholds = self._config.thresholds_for(rule.rule_type)
            if metrics.passes_thresholds(
                min_support=thresholds.min_support,
                min_confidence=thresholds.min_confidence,
                min_head_coverage=thresholds.min_head_coverage,
            ):
                results.append((rule, metrics))
                if max_results is not None and len(results) >= max_results:
                    break
        logger.debug(
            "Evaluated %d rules, %d passed thresholds",
            len(rules),
            len(results),
        )
        self._warn_on_eliminated_types(rules, metrics_by_rule)
        return results

    def _warn_on_eliminated_types(
        self,
        rules: Sequence[Rule],
        metrics_by_rule: dict[Rule, RuleMetrics],
    ) -> None:
        """Warn when head coverage alone wipes out an entire rule type.

        Head coverage is ``support`` over *all* head triples, so a rule
        pinned to one entity cannot reach the value a cyclic rule reaches;
        judging both against one floor deletes the pinned ones silently.
        The signal is specific: rules of a type that clear support and
        confidence, yet every one fails head coverage. Rules that are
        simply poor fail the other floors too and are not reported.

        Parameters
        ----------
        rules : Sequence[Rule]
            The rules that were evaluated.
        metrics_by_rule : dict[Rule, RuleMetrics]
            Their computed metrics.
        """
        by_type: dict[RuleType, list[RuleMetrics]] = defaultdict(list)
        for rule in rules:
            evaluated = metrics_by_rule.get(rule)
            if evaluated is not None:
                by_type[rule.rule_type].append(evaluated)

        for rule_type, metrics in by_type.items():
            if rule_type in self._warned_types:
                continue
            thresholds = self._config.thresholds_for(rule_type)
            if thresholds.min_head_coverage <= 0.0:
                continue
            otherwise_eligible = [
                m
                for m in metrics
                if m.support >= thresholds.min_support
                and m.confidence >= thresholds.min_confidence
            ]
            if not otherwise_eligible:
                continue
            if any(
                m.head_coverage >= thresholds.min_head_coverage
                for m in otherwise_eligible
            ):
                continue
            self._warned_types.add(rule_type)
            logger.warning(
                "All %d %s rules that met support and confidence were removed "
                "by min_head_coverage=%.5f; the best any of them reached was "
                "%.5f. Head coverage is not comparable across rule types -- "
                "give this one its own floor via RuleConfig.per_type.",
                len(otherwise_eligible),
                rule_type.value,
                thresholds.min_head_coverage,
                max(m.head_coverage for m in otherwise_eligible),
            )

    def _compute_all_metrics(
        self,
        rules: Sequence[Rule],
    ) -> dict[Rule, RuleMetrics]:
        """Compute metrics for every rule, grouping AC1 rules by body chain.

        AC1 rules that share a body chain and grounding side reuse a single
        chain-product matrix, avoiding a redundant sparse matmul per rule.
        Other rule types are evaluated individually.

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
            is_subject_grounded = rule.head.subject.kind is TermKind.CONSTANT
            groups[(self._body_signature(rule), is_subject_grounded)].append(rule)

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
        call, so the group costs one pass over its constants rather than a
        chain product per block of them.

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
            entity_id = self._ac1_entity_id(
                rule, is_subject_grounded=is_subject_grounded
            )
            if entity_id is None:
                out[rule] = self.evaluate(rule)
                continue
            pending[self._find_head_edge_type(rule)][entity_id].append(rule)

        signature = self._body_signature(rules[0])
        for head_et, by_entity in pending.items():
            self._score_ac1_entities(
                signature,
                head_et,
                by_entity,
                is_subject_grounded=is_subject_grounded,
                out=out,
            )

    def _score_ac1_entities(
        self,
        signature: BodySignature,
        head_et: EdgeTypeTuple,
        by_entity: dict[int, list[Rule]],
        *,
        is_subject_grounded: bool,
        out: dict[Rule, RuleMetrics],
    ) -> None:
        """Score every grounded entity sharing a chain and head relation.

        Parameters
        ----------
        signature : BodySignature
            The shared body chain, in forward order.
        head_et : EdgeTypeTuple
            The head relation these rules predict.
        by_entity : dict[int, list[Rule]]
            Rules awaiting metrics, keyed by their grounded entity.
        is_subject_grounded : bool
            Whether the grounded head term is the subject. Object-grounded
            rules pin the chain's tail, so they are scanned in reverse.
        out : dict[Rule, RuleMetrics]
            Destination mapping, updated in place.
        """
        entities = sorted(by_entity)
        rows = np.array(entities, dtype=np.int64)
        if is_subject_grounded:
            predictions, support = self._scanner.scan_rows(signature, head_et, rows)
        else:
            predictions, support = self._scanner.scan_reversed_rows(
                signature, head_et, rows
            )

        total_head_triples = self._graph.edge_count(head_et)
        for position, entity_id in enumerate(entities):
            metrics = _ac1_metrics(
                int(predictions[position]),
                int(support[position]),
                total_head_triples,
            )
            for rule in by_entity[entity_id]:
                out[rule] = metrics

    def _body_signature(self, rule: Rule) -> BodySignature:
        """Return the ordered body edge types identifying the chain product."""
        return tuple(self._resolve_body_atom_edge_type(atom) for atom in rule.body)

    @staticmethod
    def _ac1_entity_id(rule: Rule, *, is_subject_grounded: bool) -> int | None:
        """Return the grounded head entity id (subject or object)."""
        term = rule.head.subject if is_subject_grounded else rule.head.object_
        return term.entity_id

    def _evaluate_cyclic(self, rule: Rule) -> RuleMetrics:
        """Evaluate a cyclic rule by counting groundings per source row.

        Delegates to :class:`~anyburl._chain_scan.ChainScanner`, which walks
        the body chain without materialising its product --- the difference
        between ~1.1 GB and ~0 on BIOKG's densest chain.

        Parameters
        ----------
        rule : Rule
            A cyclic rule where both head variables appear in the body.

        Returns
        -------
        RuleMetrics
            Computed metrics.
        """
        head_et = self._find_head_edge_type(rule)
        num_predictions, support = self._scanner.scan_all_rows(
            self._body_signature(rule), head_et
        )
        if num_predictions == 0:
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        confidence = support / num_predictions
        total_head_triples = self._graph.edge_count(head_et)
        head_coverage = (
            support / total_head_triples
            if total_head_triples > 0
            else ZERO_HEAD_COVERAGE
        )

        return RuleMetrics(
            support=support,
            confidence=confidence,
            head_coverage=head_coverage,
            num_predictions=num_predictions,
        )

    def _evaluate_ac1(self, rule: Rule) -> RuleMetrics:
        """Evaluate a single AC1 rule with one grounded head entity.

        **Evaluation semantics**: for a subject-grounded rule
        ``h(person:0, Y) :- b1(X, Z0), ...``, X is a free variable in the
        stored body, but the evaluator pins X to ``person:0`` by grounding
        the chain from that entity. For an object-grounded rule
        ``h(X, city:0) :- ..., bk(Z, Y)``, Y is pinned to ``city:0`` and the
        chain is walked backwards. This matches the original AnyBURL
        semantics, where the constant appears in both the head and the
        anchoring body position.

        Takes the same path as :meth:`_evaluate_ac1_group`, so single-rule
        and batch evaluation cannot drift apart.

        Parameters
        ----------
        rule : Rule
            An AC1 rule with one constant in the head.

        Returns
        -------
        RuleMetrics
            Computed metrics.
        """
        is_subject_grounded = rule.head.subject.kind is TermKind.CONSTANT
        entity_id = self._ac1_entity_id(rule, is_subject_grounded=is_subject_grounded)
        if entity_id is None:
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        head_et = self._find_head_edge_type(rule)
        signature = self._body_signature(rule)
        rows = np.array([entity_id], dtype=np.int64)
        if is_subject_grounded:
            predictions, support = self._scanner.scan_rows(signature, head_et, rows)
        else:
            predictions, support = self._scanner.scan_reversed_rows(
                signature, head_et, rows
            )

        return _ac1_metrics(
            int(predictions[0]),
            int(support[0]),
            self._graph.edge_count(head_et),
        )

    def _evaluate_ac2(self, rule: Rule) -> RuleMetrics:
        """Evaluate an AC2 rule (one head variable absent from body).

        The body chain determines bindings for the *connected* head
        variable. The *disconnected* variable is unconstrained, so
        predictions are the Cartesian product of connected bindings
        with all entities of the disconnected variable's type.

        Only a boolean mask over the connected side is needed, so the chain
        is walked rather than multiplied out --- this was the evaluator's
        last unbounded product.

        Parameters
        ----------
        rule : Rule
            An AC2 rule.

        Returns
        -------
        RuleMetrics
            Computed metrics (typically low confidence).
        """
        head = rule.head

        body_variable_names: set[str | None] = set()
        for atom in rule.body:
            if atom.subject.kind is TermKind.VARIABLE:
                body_variable_names.add(atom.subject.name)
            if atom.object_.kind is TermKind.VARIABLE:
                body_variable_names.add(atom.object_.name)

        is_subject_connected = (
            head.subject.kind is TermKind.VARIABLE
            and head.subject.name in body_variable_names
        )
        is_object_connected = (
            head.object_.kind is TermKind.VARIABLE
            and head.object_.name in body_variable_names
        )

        if is_subject_connected == is_object_connected:
            logger.debug(
                "AC2 rule has unexpected variable structure "
                "(both or neither head variable in body): %s",
                rule,
            )
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        signature = self._body_signature(rule)
        if is_subject_connected:
            connected_mask = self._scanner.reachable_sources(signature)
            disconnected_type = head.object_.node_type
        else:
            connected_mask = self._scanner.reachable_targets(signature)
            disconnected_type = head.subject.node_type

        connected_count = int(connected_mask.sum())
        if connected_count == 0:
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        disconnected_count = self._graph.node_count(disconnected_type)
        num_predictions = connected_count * disconnected_count

        head_et = self._find_head_edge_type(rule)
        head_ei = self._graph.edge_index(head_et)

        known_connected = (head_ei[0] if is_subject_connected else head_ei[1]).numpy()
        support = int(connected_mask[known_connected].sum())

        confidence = support / num_predictions
        total_head_triples = self._graph.edge_count(head_et)
        head_coverage = (
            support / total_head_triples
            if total_head_triples > 0
            else ZERO_HEAD_COVERAGE
        )

        return RuleMetrics(
            support=support,
            confidence=confidence,
            head_coverage=head_coverage,
            num_predictions=num_predictions,
        )

    def _find_head_edge_type(self, rule: Rule) -> EdgeTypeTuple:
        """Find the graph edge type matching the rule head.

        Parameters
        ----------
        rule : Rule
            The rule whose head to match.

        Returns
        -------
        EdgeTypeTuple
            The matching ``(src_type, relation, dst_type)`` tuple.

        Raises
        ------
        ValueError
            If no matching edge type is found.
        """
        head = rule.head
        src_type = head.subject.node_type
        dst_type = head.object_.node_type
        relation = head.relation

        target: EdgeTypeTuple = (src_type, relation, dst_type)
        if target in self._edge_type_set:
            return target

        raise ValueError(f"No edge type matches rule head: {target!r}")

    def _resolve_body_atom_edge_type(self, atom: Atom) -> EdgeTypeTuple:
        """Resolve a body atom to a graph edge type.

        Parameters
        ----------
        atom : Atom
            The body atom to resolve.

        Returns
        -------
        EdgeTypeTuple
            The matching edge type.

        Raises
        ------
        ValueError
            If no matching edge type is found.
        """
        src_type = atom.subject.node_type
        dst_type = atom.object_.node_type
        relation = atom.relation

        target: EdgeTypeTuple = (src_type, relation, dst_type)
        if target in self._edge_type_set:
            return target

        raise ValueError(f"No edge type matches body atom: {target!r}")
