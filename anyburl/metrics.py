"""Rule quality evaluation: support, confidence, and head coverage."""

import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

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


def _csr_nnz(matrix: Tensor) -> int:
    """Return the stored non-zero count of a CSR tensor."""
    return matrix.col_indices().numel()


def _csr_any_row(matrix: Tensor) -> Tensor:
    """Bool mask: True for rows that contain at least one non-zero entry."""
    crow = matrix.crow_indices()
    return (crow[1:] - crow[:-1]) > 0


def _csr_any_col(matrix: Tensor) -> Tensor:
    """Bool mask: True for columns that appear in at least one row."""
    num_cols = matrix.shape[1]
    col_idx = matrix.col_indices()
    mask = torch.zeros(num_cols, dtype=torch.bool)
    if col_idx.numel() > 0:
        mask.scatter_(0, col_idx.to(torch.long), True)
    return mask


DENSE_MASK_MAX_ELEMENTS: int = 200_000_000
"""Row*col ceiling for the dense-mask intersection path (~200 MB as bool)."""


def _csr_linear_indices(matrix: Tensor) -> Tensor:
    """Return the flattened ``row * ncols + col`` index of each non-zero."""
    crow = matrix.crow_indices()
    col = matrix.col_indices()
    row_nnz = (crow[1:] - crow[:-1]).to(torch.long)
    rows = torch.repeat_interleave(
        torch.arange(crow.numel() - 1, dtype=torch.long, device=col.device),
        row_nnz,
    )
    return rows * matrix.shape[1] + col.to(torch.long)


def _dense_mask_intersection_count(mat_a: Tensor, mat_b: Tensor) -> int:
    """Count shared non-zeros via a dense boolean mask of the sparser matrix.

    Building a ``row*col`` mask from the matrix with fewer non-zeros and
    gathering the other matrix's coordinates is far faster than a set
    intersection when one matrix is near-dense. Bounded by
    :data:`DENSE_MASK_MAX_ELEMENTS`.
    """
    small, large = (
        (mat_a, mat_b) if _csr_nnz(mat_a) <= _csr_nnz(mat_b) else (mat_b, mat_a)
    )
    n_rows, n_cols = mat_a.shape
    mask = torch.zeros(n_rows * n_cols, dtype=torch.bool)
    mask[_csr_linear_indices(small)] = True
    return int(mask[_csr_linear_indices(large)].sum().item())


def _linear_isin_intersection_count(mat_a: Tensor, mat_b: Tensor) -> int:
    """Count shared non-zeros via ``isin`` on flattened indices (O(nnz) memory)."""
    a_linear = _csr_linear_indices(mat_a)
    b_linear = _csr_linear_indices(mat_b)
    query, table = (
        (a_linear, b_linear)
        if a_linear.numel() <= b_linear.numel()
        else (b_linear, a_linear)
    )
    return int(torch.isin(query, table).sum().item())


def _csr_intersection_count(mat_a: Tensor, mat_b: Tensor) -> int:
    """Count (row, col) pairs present as non-zeros in both CSR tensors.

    Uses a dense boolean mask when the ``row*col`` grid is small enough
    (:data:`DENSE_MASK_MAX_ELEMENTS`), otherwise a memory-bounded ``isin``
    on flattened indices.
    """
    if mat_a.shape != mat_b.shape:
        raise ValueError(
            f"Shape mismatch in intersection count: {mat_a.shape} vs {mat_b.shape}"
        )
    n_rows, n_cols = mat_a.shape
    if n_rows * n_cols <= DENSE_MASK_MAX_ELEMENTS:
        return _dense_mask_intersection_count(mat_a, mat_b)
    return _linear_isin_intersection_count(mat_a, mat_b)


@dataclass(frozen=True, slots=True)
class _CsrArrays:
    """Row-offset and column-index arrays of a CSR matrix as NumPy arrays.

    Extracting the CSR structure once lets grouped evaluation slice many
    rows in pure NumPy without per-rule torch dispatch.

    Parameters
    ----------
    crow : np.ndarray
        Compressed row offsets (length ``num_rows + 1``).
    col : np.ndarray
        Column indices for each stored non-zero.
    """

    crow: np.ndarray
    col: np.ndarray

    @classmethod
    def from_csr(cls, matrix: Tensor) -> "_CsrArrays":
        """Build from a sparse CSR tensor."""
        return cls(
            crow=matrix.crow_indices().numpy(),
            col=matrix.col_indices().numpy(),
        )

    def row(self, index: int) -> np.ndarray:
        """Return the stored column indices of row ``index``."""
        start = int(self.crow[index])
        end = int(self.crow[index + 1])
        return self.col[start:end]


def _ac1_metrics(
    product: _CsrArrays,
    head: _CsrArrays,
    entity_id: int,
    total_head_triples: int,
) -> RuleMetrics:
    """Compute AC1 metrics for one grounded entity from CSR rows.

    Row ``entity_id`` of ``product`` holds the rule's predictions (the
    forward chain for subject-grounded rules, the transposed chain for
    object-grounded ones). Row ``entity_id`` of ``head`` holds the known
    targets. This mirrors :meth:`RuleEvaluator._evaluate_ac1` exactly.

    Parameters
    ----------
    product : _CsrArrays
        Chain-product CSR rows (predictions).
    head : _CsrArrays
        Head-relation CSR rows (known targets).
    entity_id : int
        The grounded head entity id.
    total_head_triples : int
        Total triples with the head relation, for head coverage.

    Returns
    -------
    RuleMetrics
        Computed metrics.
    """
    predicted = product.row(entity_id)
    num_predictions = int(predicted.size)
    if num_predictions == 0:
        return RuleMetrics(
            support=0,
            confidence=ZERO_CONFIDENCE,
            head_coverage=ZERO_HEAD_COVERAGE,
            num_predictions=0,
        )

    known = head.row(entity_id)
    support = int(np.isin(predicted, known).sum())
    confidence = support / num_predictions
    head_coverage = (
        support / total_head_triples if total_head_triples > 0 else ZERO_HEAD_COVERAGE
    )
    return RuleMetrics(
        support=support,
        confidence=confidence,
        head_coverage=head_coverage,
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
            metrics = metrics_by_rule[rule]
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

        Parameters
        ----------
        rules : list[Rule]
            Rules in the group (non-empty).
        is_subject_grounded : bool
            Whether the grounded head term is the subject.
        out : dict[Rule, RuleMetrics]
            Destination mapping, updated in place.
        """
        product = self._ac1_product(rules[0], is_subject_grounded=is_subject_grounded)
        head_cache: dict[EdgeTypeTuple, tuple[_CsrArrays, int]] = {}
        entity_cache: dict[tuple[EdgeTypeTuple, int], RuleMetrics] = {}

        for rule in rules:
            entity_id = self._ac1_entity_id(
                rule, is_subject_grounded=is_subject_grounded
            )
            if entity_id is None:
                out[rule] = self.evaluate(rule)
                continue

            head_et = self._find_head_edge_type(rule)
            cache_key = (head_et, entity_id)
            metrics = entity_cache.get(cache_key)
            if metrics is None:
                head, total_head = self._ac1_head(
                    head_et, head_cache, is_subject_grounded=is_subject_grounded
                )
                metrics = _ac1_metrics(product, head, entity_id, total_head)
                entity_cache[cache_key] = metrics
            out[rule] = metrics

    def _ac1_product(self, rule: Rule, *, is_subject_grounded: bool) -> _CsrArrays:
        """Build the chain-product CSR rows for an AC1 group.

        For subject-grounded rules this is the forward body-chain product;
        for object-grounded rules it is the transposed product, built from
        transposed body matrices so the large product is never transposed.

        Parameters
        ----------
        rule : Rule
            A representative rule from the group.
        is_subject_grounded : bool
            Whether the grounded head term is the subject.

        Returns
        -------
        _CsrArrays
            CSR rows of the (possibly transposed) chain product.
        """
        body = self._build_body_chain_matrices(rule)
        if is_subject_grounded:
            matrix = self._chain_multiply(body)
        else:
            matrix = self._chain_multiply(
                [self._transpose_csr(m) for m in reversed(body)]
            )
        return _CsrArrays.from_csr(matrix)

    def _ac1_head(
        self,
        head_et: EdgeTypeTuple,
        cache: dict[EdgeTypeTuple, tuple[_CsrArrays, int]],
        *,
        is_subject_grounded: bool,
    ) -> tuple[_CsrArrays, int]:
        """Return CSR rows of the head relation and its total triple count.

        Object-grounded rules need known *sources*, so the head matrix is
        transposed. Results are cached by edge type.

        Parameters
        ----------
        head_et : EdgeTypeTuple
            The head edge type.
        cache : dict[EdgeTypeTuple, tuple[_CsrArrays, int]]
            Per-edge-type cache, updated in place.
        is_subject_grounded : bool
            Whether the grounded head term is the subject.

        Returns
        -------
        tuple[_CsrArrays, int]
            Head CSR rows and total head triple count.
        """
        cached = cache.get(head_et)
        if cached is not None:
            return cached

        matrix = self._graph.get_csr_matrix(head_et)
        if not is_subject_grounded:
            matrix = self._transpose_csr(matrix)
        value = (_CsrArrays.from_csr(matrix), self._graph.edge_count(head_et))
        cache[head_et] = value
        return value

    def _body_signature(self, rule: Rule) -> BodySignature:
        """Return the ordered body edge types identifying the chain product."""
        return tuple(self._resolve_body_atom_edge_type(atom) for atom in rule.body)

    @staticmethod
    def _ac1_entity_id(rule: Rule, *, is_subject_grounded: bool) -> int | None:
        """Return the grounded head entity id (subject or object)."""
        term = rule.head.subject if is_subject_grounded else rule.head.object_
        return term.entity_id

    @staticmethod
    def _transpose_csr(matrix: Tensor) -> Tensor:
        """Return the transpose of a sparse CSR tensor as CSR."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            transposed: Tensor = matrix.to_sparse_coo().t().to_sparse_csr()  # type: ignore[no-untyped-call]
            return transposed

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

        num_predictions = _csr_nnz(prediction_matrix)

        if num_predictions == 0:
            return RuleMetrics(
                support=0,
                confidence=ZERO_CONFIDENCE,
                head_coverage=ZERO_HEAD_COVERAGE,
                num_predictions=0,
            )

        head_et = self._find_head_edge_type(rule)
        head_matrix = self._graph.get_csr_matrix(head_et)
        support = _csr_intersection_count(prediction_matrix, head_matrix)
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

        **Evaluation semantics**: For a subject-grounded rule
        ``h(person:0, Y) :- b1(X, Z0), ...``, X is a free variable in
        the stored body, but the evaluator pins X to ``person:0`` by
        starting the forward chain from that entity's one-hot vector.
        For an object-grounded rule ``h(X, city:0) :- ..., bk(Z, Y)``,
        Y is pinned to ``city:0`` via backward chain propagation.

        This matches the original AnyBURL semantics where the constant
        appears in both head and the anchoring body position.

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

        if is_subject_connected:
            connected_mask = _csr_any_row(prediction_matrix)
            disconnected_type = head.object_.node_type
        else:
            connected_mask = _csr_any_col(prediction_matrix)
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
        if target in self._edge_type_set:
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
        if target in self._edge_type_set:
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
