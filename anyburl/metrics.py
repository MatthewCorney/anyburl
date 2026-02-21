"""Rule quality evaluation: support, confidence, and head coverage."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

import torch
from torch import Tensor
from tqdm import tqdm

from ._logging import get_logger
from .graph import EdgeTypeTuple, HeteroGraph
from .rule import Atom, Rule, RuleConfig, RuleType, TermKind

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
        results: list[tuple[Rule, RuleMetrics]] = []
        for rule in tqdm(rules, desc="Evaluating Rules"):
            metrics = self.evaluate(rule)
            if metrics.passes_thresholds(
                min_support=self._config.min_support,
                min_confidence=self._config.min_confidence,
                min_head_coverage=self._config.min_head_coverage,
            ):
                results.append((rule, metrics))
                if max_results is not None and len(results) >= max_results:
                    break
        logger.debug(
            "Evaluated %d rules, %d passed thresholds",
            len(rules),
            len(results),
        )
        return results

    def _evaluate_cyclic(self, rule: Rule) -> RuleMetrics:
        """Evaluate a cyclic rule via sparse chain matmul.

        Parameters
        ----------
        rule : Rule
            A cyclic rule where both head variables appear in the body.

        Returns
        -------
        RuleMetrics
            Computed metrics.
        """
        chain = self._build_body_chain_matrices(rule)
        prediction_matrix = self._chain_multiply(chain)

        # TODO: .to_dense() is the bottleneck for larger graphs.
        # Should use sparse intersection for scalability.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            prediction_dense = prediction_matrix.to_dense()
        prediction_mask = prediction_dense > 0
        num_predictions = int(prediction_mask.sum().item())

        if num_predictions == 0:
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        head_et = self._find_head_edge_type(rule)
        head_matrix = self._graph.get_csr_matrix(head_et)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            head_dense = head_matrix.to_dense() > 0

        support = int((prediction_mask & head_dense).sum().item())
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
        """Evaluate an AC1 rule with one grounded head entity.

        Builds a one-hot vector for the constant entity and multiplies
        through the body chain to find predictions.

        Parameters
        ----------
        rule : Rule
            An AC1 rule with one constant in the head.

        Returns
        -------
        RuleMetrics
            Computed metrics.
        """
        head = rule.head
        chain = self._build_body_chain_matrices(rule)

        is_subject_grounded = head.subject.kind is TermKind.CONSTANT
        if is_subject_grounded:
            entity_id = head.subject.entity_id
            num_nodes = self._graph.node_count(head.subject.node_type)
        else:
            entity_id = head.object_.entity_id
            num_nodes = self._graph.node_count(head.object_.node_type)

        if entity_id is None:
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        one_hot = torch.zeros(num_nodes, dtype=torch.float32)
        one_hot[entity_id] = 1.0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            if is_subject_grounded:
                predictions = one_hot.unsqueeze(0)
                for matrix in chain:
                    predictions = predictions @ matrix
                predictions = predictions.squeeze(0)
            else:
                predictions = one_hot.unsqueeze(1)
                for matrix in reversed(chain):
                    predictions = matrix @ predictions
                predictions = predictions.squeeze(1)

        predicted_mask = predictions > 0
        num_predictions = int(predicted_mask.sum().item())

        if num_predictions == 0:
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        head_et = self._find_head_edge_type(rule)
        head_ei = self._graph.edge_index(head_et)

        if is_subject_grounded:
            known_mask = head_ei[0] == entity_id
            known_targets = head_ei[1, known_mask]
        else:
            known_mask = head_ei[1] == entity_id
            known_targets = head_ei[0, known_mask]

        predicted_indices = torch.where(predicted_mask)[0]
        support = int(torch.isin(predicted_indices, known_targets).sum().item())

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

    def _evaluate_ac2(self, rule: Rule) -> RuleMetrics:
        """Evaluate an AC2 rule (one head variable absent from body).

        The body chain determines bindings for the *connected* head
        variable. The *disconnected* variable is unconstrained, so
        predictions are the Cartesian product of connected bindings
        with all entities of the disconnected variable's type.

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

        chain = self._build_body_chain_matrices(rule)
        prediction_matrix = self._chain_multiply(chain)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            dense = prediction_matrix.to_dense()

        if is_subject_connected:
            connected_mask = (dense > 0).any(dim=1)
            disconnected_type = head.object_.node_type
        else:
            connected_mask = (dense > 0).any(dim=0)
            disconnected_type = head.subject.node_type

        connected_count = int(connected_mask.sum().item())
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

        known_connected_entities = head_ei[0] if is_subject_connected else head_ei[1]
        support = int(connected_mask[known_connected_entities].sum().item())

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
        if target in dict.fromkeys(self._graph.edge_types):
            return target

        raise ValueError(f"No edge type matches rule head: {target!r}")

    def _build_body_chain_matrices(self, rule: Rule) -> list[Tensor]:
        """Build CSR matrices for each body atom in chain order.

        Parameters
        ----------
        rule : Rule
            The rule whose body to convert to matrices.

        Returns
        -------
        list[Tensor]
            Sparse CSR float tensors, one per body atom.

        Raises
        ------
        ValueError
            If a body atom's relation doesn't match any edge type.
        """
        matrices: list[Tensor] = []
        for atom in rule.body:
            et = self._resolve_body_atom_edge_type(atom)
            matrices.append(self._graph.get_csr_matrix(et))
        return matrices

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
        if target in dict.fromkeys(self._graph.edge_types):
            return target

        raise ValueError(f"No edge type matches body atom: {target!r}")

    @staticmethod
    def _chain_multiply(matrices: list[Tensor]) -> Tensor:
        """Multiply a chain of sparse matrices left to right.

        Parameters
        ----------
        matrices : list[Tensor]
            Sparse CSR float tensors to multiply.

        Returns
        -------
        Tensor
            The product matrix.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            result = matrices[0]
            for matrix in matrices[1:]:
                result = result @ matrix
        return result
