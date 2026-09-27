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

from ._chain_scan import ChainScanner
from ._logging import get_logger
from .graph import EdgeTypeTuple, HeteroGraph
from .rule import Atom, Rule, RuleConfig, RuleType, TermKind

BodySignature = tuple[EdgeTypeTuple, ...]
"""Ordered edge types of a rule body; shared by rules with the same chain."""

logger = get_logger(__name__)


class ChainBudgetExceededError(RuntimeError):
    """Raised when a body chain predicts more pairs than the budget allows.

    Carries what was known when the scan stopped, so a caller can report
    why a rule went unevaluated rather than silently dropping it.

    Parameters
    ----------
    predictions : int
        Predictions counted before abandoning the chain.
    budget : int
        The configured ceiling.
    """

    def __init__(self, predictions: int, budget: int) -> None:
        super().__init__(
            f"body chain predicts at least {predictions} pairs, over the "
            f"max_chain_predictions budget of {budget}"
        )
        self.predictions = predictions
        self.budget = budget


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

MAX_FOLDED_TAIL_NNZ: int = 20_000_000
"""Largest suffix product worth precomputing and reusing across row blocks.

A chain is multiplied as ``rows @ (rest of chain)`` when the right-hand side
is this small, because the suffix is then computed once instead of once per
block. On BIOKG's worst chain the suffix holds 8.2M non-zeros against the
93.2M of the full product, which cuts peak memory by 17% and is slightly
faster. Above this the suffix would be dearer to keep than to recompute.
"""

ROW_BLOCK_SIZE: int = 1024
"""Source rows multiplied through a chain at once.

A whole chain product can be far larger than memory on a dense graph --- a
single BIOKG chain is estimated at 3.9 GB --- while support and prediction
counts are plain sums over disjoint row blocks. Multiplying a block at a
time bounds peak memory without approximating anything.
"""


def _select_csr_rows(matrix: Tensor, start: int, stop: int) -> Tensor:
    """Return rows ``[start, stop)`` of a CSR tensor as its own CSR tensor.

    Parameters
    ----------
    matrix : Tensor
        A sparse CSR tensor.
    start, stop : int
        Half-open row range.

    Returns
    -------
    Tensor
        A CSR tensor with ``stop - start`` rows and the same column count.
    """
    crow = matrix.crow_indices()
    first = int(crow[start].item())
    last = int(crow[stop].item())
    return torch.sparse_csr_tensor(
        crow[start : stop + 1] - first,
        matrix.col_indices()[first:last],
        matrix.values()[first:last],
        size=(stop - start, matrix.shape[1]),
    )


def _fold_chain_tail(chain: Sequence[Tensor]) -> Tensor | None:
    """Precompute a chain's suffix when it is cheaper to keep than to repeat.

    Matrix chain association changes the size of the intermediates
    dramatically. Multiplying a block of source rows left to right rebuilds
    a large intermediate for every block; folding the suffix once and
    multiplying ``rows @ suffix`` reuses one small matrix throughout.

    Parameters
    ----------
    chain : Sequence[Tensor]
        Body atom matrices in chain order.

    Returns
    -------
    Tensor | None
        The product of ``chain[1:]``, or ``None`` when the chain is too
        short to benefit or the suffix grew past
        :data:`MAX_FOLDED_TAIL_NNZ`.
    """
    if len(chain) < 3:  # noqa: PLR2004
        return None
    tail = chain[-1]
    for matrix in reversed(chain[1:-1]):
        tail = matrix @ tail
        if _csr_nnz(tail) > MAX_FOLDED_TAIL_NNZ:
            return None
    return tail


def _gather_csr_rows(matrix: Tensor, row_ids: Sequence[int]) -> Tensor:
    """Return the named rows of a CSR tensor, in the order given.

    Parameters
    ----------
    matrix : Tensor
        A sparse CSR tensor.
    row_ids : Sequence[int]
        Row indices to gather.

    Returns
    -------
    Tensor
        A CSR tensor with one row per entry of ``row_ids``.
    """
    crow = matrix.crow_indices()
    col = matrix.col_indices()
    val = matrix.values()
    offsets = [0]
    cols: list[Tensor] = []
    vals: list[Tensor] = []
    for row_id in row_ids:
        first = int(crow[row_id].item())
        last = int(crow[row_id + 1].item())
        cols.append(col[first:last])
        vals.append(val[first:last])
        offsets.append(offsets[-1] + last - first)
    return torch.sparse_csr_tensor(
        torch.tensor(offsets, dtype=crow.dtype),
        torch.cat(cols) if cols else col[:0],
        torch.cat(vals) if vals else val[:0],
        size=(len(row_ids), matrix.shape[1]),
    )


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


@dataclass(frozen=True, slots=True)
class _AC1GroupContext:
    """State shared while scoring one AC1 group's blocks of entities.

    Parameters
    ----------
    rules_by_entity : dict[int, list[Rule]]
        Rules awaiting metrics, keyed by grounded entity.
    head_cache : dict[EdgeTypeTuple, tuple[_CsrArrays, int]]
        Head-relation rows cached across blocks, updated in place.
    is_subject_grounded : bool
        Whether the grounded head term is the subject.
    out : dict[Rule, RuleMetrics]
        Destination mapping, updated in place.
    """

    rules_by_entity: dict[int, list[Rule]]
    head_cache: dict[EdgeTypeTuple, tuple[_CsrArrays, int]]
    is_subject_grounded: bool
    out: dict[Rule, RuleMetrics]


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
        self._over_budget = 0
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
            "Evaluated %d rules, %d passed thresholds, %d over budget",
            len(rules),
            len(results),
            self._over_budget,
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
        self._over_budget = 0

        ac1_rules = [r for r in rules if r.rule_type is RuleType.AC1]
        other_rules = [r for r in rules if r.rule_type is not RuleType.AC1]

        for rule in tqdm(other_rules, desc="Evaluating rules", disable=not other_rules):
            if rule not in metrics_by_rule:
                try:
                    metrics_by_rule[rule] = self.evaluate(rule)
                except ChainBudgetExceededError as exceeded:
                    self._report_over_budget(rule, exceeded)

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
            try:
                self._evaluate_ac1_group(
                    group, is_subject_grounded=is_subject_grounded, out=out
                )
            except ChainBudgetExceededError as exceeded:
                self._report_over_budget(group[0], exceeded, group_size=len(group))

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
        self._check_budget(num_predictions)

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

    def _scan_chain_blocks(
        self,
        chain: list[Tensor],
        head_matrix: Tensor,
    ) -> tuple[int, int]:
        """Count predictions and support by multiplying the chain in row blocks.

        Equivalent to multiplying the whole chain and comparing against the
        head relation, but peak memory is bounded by one block's product
        rather than the full one.

        Parameters
        ----------
        chain : list[Tensor]
            Body atom matrices in chain order.
        head_matrix : Tensor
            The head relation's adjacency matrix.

        Returns
        -------
        tuple[int, int]
            ``(num_predictions, support)`` summed over all blocks.

        Raises
        ------
        ChainBudgetExceededError
            If the running prediction count passes
            ``RuleConfig.max_chain_predictions``.
        """
        num_rows = int(chain[0].shape[0])
        num_predictions = 0
        support = 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            tail = _fold_chain_tail(chain)
            remainder = chain[1:] if tail is None else (tail,)
            for start in range(0, num_rows, ROW_BLOCK_SIZE):
                stop = min(start + ROW_BLOCK_SIZE, num_rows)
                block = _select_csr_rows(chain[0], start, stop)
                for matrix in remainder:
                    block = block @ matrix
                block_nnz = _csr_nnz(block)
                if block_nnz == 0:
                    continue
                num_predictions += block_nnz
                self._check_budget(num_predictions)
                support += _csr_intersection_count(
                    block, _select_csr_rows(head_matrix, start, stop)
                )

        return num_predictions, support

    def _report_over_budget(
        self,
        rule: Rule,
        exceeded: ChainBudgetExceededError,
        *,
        group_size: int = 1,
    ) -> None:
        """Log that a chain was abandoned, naming it so the gap is visible.

        Parameters
        ----------
        rule : Rule
            A rule using the abandoned chain.
        exceeded : ChainBudgetExceededError
            The raised budget error.
        group_size : int
            How many rules shared the abandoned chain.
        """
        self._over_budget += group_size
        chain = " -> ".join(atom.relation for atom in rule.body)
        logger.warning(
            "Skipped %d %s rule(s) on chain %s: at least %d predictions, over "
            "the max_chain_predictions budget of %d. These rules are "
            "unevaluated, not rejected.",
            group_size,
            rule.rule_type.value,
            chain,
            exceeded.predictions,
            exceeded.budget,
        )

    def _check_budget(self, predictions: int) -> None:
        """Abandon the chain if it has already outgrown the budget.

        Parameters
        ----------
        predictions : int
            Predictions counted so far.

        Raises
        ------
        ChainBudgetExceededError
            If a budget is configured and ``predictions`` exceeds it.
        """
        budget = self._config.max_chain_predictions
        if budget is not None and predictions > budget:
            raise ChainBudgetExceededError(predictions, budget)

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
