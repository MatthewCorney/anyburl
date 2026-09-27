"""Rule prediction: ground rules against the graph and aggregate scores.

Groups rules by body chain to avoid redundant sparse matmul computation.
On DBLP there are only ~36 unique chains across thousands of rules, so
this yields ~100x fewer matmul calls compared to per-rule grounding.

Prediction iterates per head entity, calling ``score_tails()`` which
uses vectorized ``torch.isin`` operations on pre-computed tensors.

The stored values of a chain product are the number of body groundings
reaching each candidate. :class:`ScoringStrategy` decides whether that
count weighs on the score or is ignored.
"""

import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, assert_never

import torch
from torch import Tensor
from tqdm import tqdm

from ._logging import get_logger
from .graph import EdgeTypeTuple, HeteroGraph
from .metrics import RuleMetrics, aggregate_confidence
from .rule import Rule, RuleType, TermKind

logger = get_logger(__name__)

BodyChainKey = tuple[tuple[str, str, str], ...]
"""Unique identifier for a body chain: tuple of edge-type triples."""

GROUNDING_WEIGHT_OFFSET: float = 1.0
"""Offset in ``log2(offset + groundings)``, set so one grounding weighs 1.0."""


class ScoringStrategy(StrEnum):
    """How a rule's confidence is weighted against a candidate.

    Attributes
    ----------
    NOISY_OR : str
        Each firing rule contributes its confidence once, however many
        body groundings support the candidate. Scores are coarse: a
        candidate either is reachable via a chain or is not.
    PATH_WEIGHTED : str
        A rule's confidence is weighted by ``log2(1 + groundings)``, so
        a candidate reached by many body groundings outranks one reached
        by a single grounding. A single grounding reproduces
        :attr:`NOISY_OR` exactly.
    """

    NOISY_OR = "noisy_or"
    PATH_WEIGHTED = "path_weighted"


def _evidence_weights(groundings: Tensor, strategy: ScoringStrategy) -> Tensor:
    """Convert body grounding counts into per-candidate evidence weights.

    Parameters
    ----------
    groundings : Tensor
        Stored chain-product values: the number of body groundings
        supporting each candidate.
    strategy : ScoringStrategy
        The weighting scheme.

    Returns
    -------
    Tensor
        Weights parallel to ``groundings``.
    """
    match strategy:
        case ScoringStrategy.NOISY_OR:
            return torch.ones_like(groundings)
        case ScoringStrategy.PATH_WEIGHTED:
            return torch.log2(GROUNDING_WEIGHT_OFFSET + groundings)
        case _ as unreachable:
            assert_never(unreachable)


class GroundingMode(StrEnum):
    """How a body chain is grounded against the graph.

    Attributes
    ----------
    MATERIALISED : str
        Multiply the whole chain up front and keep the product and its
        transpose. Queries are then CSR row slices, which is fastest when
        many entities are scored, but the products are dense enough on
        large graphs to exhaust memory --- BIOKG's 48 chains would need
        tens of gigabytes.
    ON_DEMAND : str
        Keep only the body matrices and propagate a one-hot vector through
        them per query. Costs one chain multiply per query but adds no
        resident memory, so it suits evaluating a test set, where only a
        small fraction of rows is ever touched.
    """

    MATERIALISED = "materialised"
    ON_DEMAND = "on_demand"


class ChainGrounding(Protocol):
    """Supplies one body chain's predictions for a single query entity."""

    def row(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Return candidates reachable from ``query_id``, with grounding counts."""
        ...

    def column(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Return candidates reaching ``query_id``, with grounding counts."""
        ...


@dataclass(frozen=True, slots=True)
class _MaterialisedChain:
    """A body chain whose full product and transpose are held in memory.

    Parameters
    ----------
    product : Tensor
        Sparse CSR float tensor, computed once via chain matmul.
    product_transposed : Tensor
        Transposed product for reverse-direction queries.
    """

    product: Tensor
    product_transposed: Tensor

    def row(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Slice one row of the product."""
        return _slice_csr_row(self.product, query_id)

    def column(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Slice one row of the transposed product."""
        return _slice_csr_row(self.product_transposed, query_id)


@dataclass(frozen=True, slots=True)
class _OnDemandChain:
    """A body chain grounded per query, holding no product of its own.

    The matrices are the graph's own cached CSR tensors, so an instance
    adds no resident memory beyond the intermediate vector of one query.

    Parameters
    ----------
    matrices : tuple[Tensor, ...]
        Body atom adjacency matrices, in chain order.
    """

    matrices: tuple[Tensor, ...]

    def row(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Propagate a one-hot row vector forward through the chain."""
        vector = torch.zeros(1, int(self.matrices[0].shape[0]))
        vector[0, query_id] = 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            for matrix in self.matrices:
                vector = vector @ matrix
        return _nonzero_entries(vector.squeeze(0))

    def column(self, query_id: int) -> tuple[Tensor, Tensor]:
        """Propagate a one-hot column vector backward through the chain."""
        vector = torch.zeros(int(self.matrices[-1].shape[1]), 1)
        vector[query_id, 0] = 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            for matrix in reversed(self.matrices):
                vector = matrix @ vector
        return _nonzero_entries(vector.squeeze(1))


def _slice_csr_row(csr_matrix: Tensor, row_id: int) -> tuple[Tensor, Tensor]:
    """Return the stored column indices and values of one CSR row."""
    crow = csr_matrix.crow_indices()
    start = int(crow[row_id].item())
    end = int(crow[row_id + 1].item())
    return csr_matrix.col_indices()[start:end], csr_matrix.values()[start:end]


def _nonzero_entries(dense: Tensor) -> tuple[Tensor, Tensor]:
    """Return the indices and values of a dense vector's non-zeros."""
    indices = torch.nonzero(dense, as_tuple=True)[0]
    return indices, dense[indices]


@dataclass(frozen=True, slots=True)
class _CyclicChainGroup:
    """All cyclic rules sharing a body chain, pre-aggregated.

    Parameters
    ----------
    chain : BodyChainKey
        The body chain these rules share.
    grounding : ChainGrounding
        The shared chain grounding.
    aggregated_confidence : float
        Noisy-or of all rule confidences in this group.
    """

    chain: BodyChainKey
    grounding: ChainGrounding
    aggregated_confidence: float


@dataclass(frozen=True, slots=True)
class _GroundedSide:
    """One query direction's view of an AC1 group.

    Scoring tails and scoring heads are mirror images: each reads a row
    of one product matrix, applies the confidence grounded on the query
    side, then applies the confidences grounded on the candidate side.
    This view names those four pieces so a single routine serves both.

    Parameters
    ----------
    fetch : Callable[[int], tuple[Tensor, Tensor]]
        Returns this direction's candidates and grounding counts for a
        query entity --- the grounding's ``row`` or ``column``.
    query_confidences : dict[int, float]
        Confidence per query entity, for rules grounded on the query side.
    candidate_ids : Tensor
        1-D int tensor of entity IDs grounded on the candidate side.
    candidate_confidences : Tensor
        1-D float tensor parallel to ``candidate_ids``.
    """

    fetch: Callable[[int], tuple[Tensor, Tensor]]
    query_confidences: dict[int, float]
    candidate_ids: Tensor
    candidate_confidences: Tensor


@dataclass(frozen=True, slots=True)
class _AC1ChainGroup:
    """All AC1 rules sharing a body chain, indexed by grounded entity.

    Parameters
    ----------
    chain : BodyChainKey
        The body chain these rules share.
    grounding : ChainGrounding
        The shared chain grounding.
    subject_grounded : dict[int, float]
        Entity ID to aggregated confidence for subject-grounded rules.
    object_grounded : dict[int, float]
        Entity ID to aggregated confidence for object-grounded rules.
    subject_entity_ids : Tensor
        1-D int tensor of subject-grounded entity IDs.
    subject_confidences : Tensor
        1-D float tensor of corresponding confidences.
    object_entity_ids : Tensor
        1-D int tensor of object-grounded entity IDs.
    object_confidences : Tensor
        1-D float tensor of corresponding confidences.
    """

    chain: BodyChainKey
    grounding: ChainGrounding
    subject_grounded: dict[int, float]
    object_grounded: dict[int, float]
    subject_entity_ids: Tensor
    subject_confidences: Tensor
    object_entity_ids: Tensor
    object_confidences: Tensor

    @property
    def tail_side(self) -> _GroundedSide:
        """Return the view used when scoring tails for a given head."""
        return _GroundedSide(
            fetch=self.grounding.row,
            query_confidences=self.subject_grounded,
            candidate_ids=self.object_entity_ids,
            candidate_confidences=self.object_confidences,
        )

    @property
    def head_side(self) -> _GroundedSide:
        """Return the view used when scoring heads for a given tail."""
        return _GroundedSide(
            fetch=self.grounding.column,
            query_confidences=self.object_grounded,
            candidate_ids=self.subject_entity_ids,
            candidate_confidences=self.subject_confidences,
        )


@dataclass(frozen=True, slots=True)
class RuleFiring:
    """One body chain's contribution to a single predicted pair.

    A score on its own cannot be checked or argued with. Naming the chain
    that fired, how many groundings supported it and what it contributed
    turns a number back into a claim --- "these proteins interact and that
    one carries the phenotype" rather than "0.29".

    Parameters
    ----------
    chain : BodyChainKey
        Edge types of the body that fired.
    rule_type : RuleType
        Structural type of the rules in the firing group.
    confidence : float
        Aggregated confidence of the rules sharing this chain.
    groundings : int
        Number of body groundings connecting this pair.
    contribution : float
        Probability mass this firing alone contributes to the score,
        ``1 - (1 - confidence) ** weight``.
    """

    chain: BodyChainKey
    rule_type: RuleType
    confidence: float
    groundings: int
    contribution: float

    def describe(self) -> str:
        """Return a one-line human-readable account of this firing.

        Edge types are spelled with both endpoints, because a graph may
        reuse one relation name across many types --- DBLP calls every
        relation ``to``, so bare names would render every chain alike.
        """
        body = " -> ".join(
            f"{src}_{relation}_{dst}" for src, relation, dst in self.chain
        )
        return (
            f"{body} [{self.rule_type.value}] "
            f"conf={self.confidence:.4f} groundings={self.groundings} "
            f"contribution={self.contribution:.4f}"
        )


@dataclass(frozen=True, slots=True)
class Prediction:
    """A predicted triple with an aggregated quality score.

    Parameters
    ----------
    head_id : int
        Source entity index in the knowledge graph.
    tail_id : int
        Destination entity index in the knowledge graph.
    head_type : str
        Source node type.
    tail_type : str
        Destination node type.
    relation : str
        Predicted relation name.
    score : float
        Noisy-or aggregated confidence across all firing rules, each
        weighted per the predictor's :class:`ScoringStrategy`.
    explanations : tuple[RuleFiring, ...]
        Which rules produced the score, strongest first. Empty unless the
        prediction came from a method that attaches provenance ---
        :meth:`RulePredictor.predict` leaves it empty because attaching it
        to every candidate pair would dwarf the scores themselves.
    """

    head_id: int
    tail_id: int
    head_type: str
    tail_type: str
    relation: str
    score: float
    explanations: tuple["RuleFiring", ...] = ()


class RulePredictor:
    """Grounds rules against a graph and aggregates predictions.

    Pre-computes product matrices for each unique body chain at init
    time, then uses them for fast ``predict()``, ``score_tails()``,
    and ``score_heads()`` queries.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to ground rules against.
    results : list[tuple[Rule, RuleMetrics]]
        Rules paired with their evaluated metrics. All rules must
        predict the same head relation.
    scoring_strategy : ScoringStrategy
        How rule confidences are weighted against a candidate.
    grounding_mode : GroundingMode
        Whether to materialise chain products up front or ground each
        chain per query. Materialised is fastest when scoring many
        entities; on-demand adds no resident memory and suits large
        graphs or evaluating a small test set.

    Raises
    ------
    ValueError
        If rules predict different head relations or results is empty.
    """

    def __init__(
        self,
        graph: HeteroGraph,
        results: list[tuple[Rule, RuleMetrics]],
        *,
        scoring_strategy: ScoringStrategy = ScoringStrategy.PATH_WEIGHTED,
        grounding_mode: GroundingMode = GroundingMode.MATERIALISED,
    ) -> None:
        if not results:
            msg = "results must not be empty"
            raise ValueError(msg)

        head_relations = {rule.head.edge_signature for rule, _ in results}
        if len(head_relations) > 1:
            msg = (
                f"All rules must predict the same head relation, "
                f"got {len(head_relations)} distinct: {head_relations}"
            )
            raise ValueError(msg)

        t_init = time.perf_counter()

        self._graph = graph
        self._scoring_strategy = scoring_strategy

        first_rule = results[0][0]
        self._head_type = first_rule.head.subject.node_type
        self._tail_type = first_rule.head.object_.node_type
        self._relation = first_rule.head.relation
        self._head_edge_type: EdgeTypeTuple = first_rule.head.edge_signature

        self._num_heads = graph.node_count(self._head_type)
        self._num_tails = graph.node_count(self._tail_type)

        t_chain = time.perf_counter()
        chain_products = self._build_groundings(graph, results, grounding_mode)
        t_chain_done = time.perf_counter()

        self._cyclic_groups, self._ac1_groups = self._build_groups(
            results, chain_products
        )
        t_groups_done = time.perf_counter()

        logger.info(
            "RulePredictor init: chains=%.3fs groups=%.3fs total=%.3fs",
            t_chain_done - t_chain,
            t_groups_done - t_chain_done,
            t_groups_done - t_init,
        )

    def predict(
        self, *, filter_known: bool = False, min_score: float = 0.0
    ) -> list[Prediction]:
        """Ground rules and aggregate predictions per head entity.

        Iterates over head entities, calling :meth:`score_tails` for each
        to leverage vectorized tensor operations.

        Parameters
        ----------
        filter_known : bool
            If ``True``, exclude predictions corresponding to edges
            already present in the graph.
        min_score : float
            Minimum score threshold for predictions. Pairs with scores
            at or below this value are excluded.

        Returns
        -------
        list[Prediction]
            Predictions sorted by score descending.
        """
        t_start = time.perf_counter()

        known_tails: dict[int, set[int]] = {}
        if filter_known:
            known_tails = self._build_known_tail_sets()

        predictions: list[Prediction] = []

        for head_id in tqdm(range(self._num_heads), desc="Predicting"):
            scores = self.score_tails(head_id)
            nonzero_mask = scores > min_score
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]

            known = known_tails.get(head_id, set())

            for tail_id_t in nonzero_indices:
                tail_id = int(tail_id_t.item())
                if tail_id in known:
                    continue
                predictions.append(
                    Prediction(
                        head_id=head_id,
                        tail_id=tail_id,
                        head_type=self._head_type,
                        tail_type=self._tail_type,
                        relation=self._relation,
                        score=float(scores[tail_id].item()),
                    )
                )

        predictions.sort(key=lambda p: p.score, reverse=True)

        t_done = time.perf_counter()
        logger.info(
            "predict(): total=%.3fs predictions=%d",
            t_done - t_start,
            len(predictions),
        )
        return predictions

    def score_tails(self, head_id: int) -> Tensor:
        """Compute noisy-or scores for all candidate tails given a head.

        Parameters
        ----------
        head_id : int
            The source entity index.

        Returns
        -------
        Tensor
            1-D float tensor of shape ``(num_tails,)`` with scores.
        """
        t_start = time.perf_counter()
        complement = torch.ones(self._num_tails)

        for cyc_group in self._cyclic_groups:
            cols, weights = self._row_evidence(cyc_group.grounding.row, head_id)
            if cols.numel() > 0:
                complement[cols] *= (1.0 - cyc_group.aggregated_confidence) ** weights

        for ac1_group in self._ac1_groups:
            self._apply_ac1_side(complement, ac1_group.tail_side, head_id)

        logger.debug(
            "score_tails(head=%d): %.3fs", head_id, time.perf_counter() - t_start
        )
        return 1.0 - complement

    def score_heads(self, tail_id: int) -> Tensor:
        """Compute noisy-or scores for all candidate heads given a tail.

        Parameters
        ----------
        tail_id : int
            The destination entity index.

        Returns
        -------
        Tensor
            1-D float tensor of shape ``(num_heads,)`` with scores.
        """
        t_start = time.perf_counter()
        complement = torch.ones(self._num_heads)

        for cyc_group in self._cyclic_groups:
            rows, weights = self._row_evidence(cyc_group.grounding.column, tail_id)
            if rows.numel() > 0:
                complement[rows] *= (1.0 - cyc_group.aggregated_confidence) ** weights

        for ac1_group in self._ac1_groups:
            self._apply_ac1_side(complement, ac1_group.head_side, tail_id)

        logger.debug(
            "score_heads(tail=%d): %.3fs", tail_id, time.perf_counter() - t_start
        )
        return 1.0 - complement

    def explain(self, head_id: int, tail_id: int) -> tuple[RuleFiring, ...]:
        """Report which rules produce the score for one predicted pair.

        Parameters
        ----------
        head_id : int
            The source entity.
        tail_id : int
            The destination entity.

        Returns
        -------
        tuple[RuleFiring, ...]
            One entry per firing body chain, strongest contribution first.
            Empty when no rule connects the pair.
        """
        firings: list[RuleFiring] = []

        for cyc_group in self._cyclic_groups:
            weight = _weight_for_candidate(
                cyc_group.grounding.row(head_id), tail_id, self._scoring_strategy
            )
            if weight is None:
                continue
            groundings, evidence = weight
            firings.append(
                _firing(
                    cyc_group.chain,
                    RuleType.CYCLIC,
                    cyc_group.aggregated_confidence,
                    groundings,
                    evidence,
                )
            )

        for ac1_group in self._ac1_groups:
            weight = _weight_for_candidate(
                ac1_group.grounding.row(head_id), tail_id, self._scoring_strategy
            )
            if weight is None:
                continue
            groundings, evidence = weight
            for confidence in (
                ac1_group.subject_grounded.get(head_id),
                ac1_group.object_grounded.get(tail_id),
            ):
                if confidence is not None:
                    firings.append(
                        _firing(
                            ac1_group.chain,
                            RuleType.AC1,
                            confidence,
                            groundings,
                            evidence,
                        )
                    )

        firings.sort(key=lambda firing: firing.contribution, reverse=True)
        return tuple(firings)

    def top_tails(self, head_id: int, *, limit: int = 10) -> list[Prediction]:
        """Return the best-scoring tails for one head, with explanations.

        This is the query :meth:`predict` cannot answer cheaply: it scores
        one entity and keeps only the best candidates, so provenance can be
        attached without materialising every pair in the graph.

        Parameters
        ----------
        head_id : int
            The source entity.
        limit : int
            Maximum number of predictions to return.

        Returns
        -------
        list[Prediction]
            Predictions sorted by score descending, each carrying the rule
            firings that produced it. Candidates scoring zero are omitted.

        Raises
        ------
        ValueError
            If ``limit`` is not positive.
        """
        if limit < 1:
            raise ValueError(f"limit must be positive, got {limit}")

        scores = self.score_tails(head_id)
        ranked = torch.argsort(scores, descending=True)[:limit]

        predictions: list[Prediction] = []
        for tail_index in ranked.tolist():
            score = float(scores[tail_index].item())
            if score <= 0.0:
                break
            predictions.append(
                Prediction(
                    head_id=head_id,
                    tail_id=tail_index,
                    head_type=self._head_type,
                    tail_type=self._tail_type,
                    relation=self._relation,
                    score=score,
                    explanations=self.explain(head_id, tail_index),
                )
            )
        return predictions

    def _build_known_tail_sets(self) -> dict[int, set[int]]:
        """Build per-head sets of known tail IDs for the head relation.

        Returns
        -------
        dict[int, set[int]]
            Mapping from head_id to the set of known tail_ids.
        """
        ei = self._graph.edge_index(self._head_edge_type)
        result: dict[int, set[int]] = defaultdict(set)
        for h, t in zip(ei[0].tolist(), ei[1].tolist(), strict=True):
            result[h].add(t)
        return dict(result)

    def _row_evidence(
        self,
        fetch: Callable[[int], tuple[Tensor, Tensor]],
        query_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Fetch one query's candidates and turn counts into evidence weights.

        The fetched values are the number of body groundings reaching each
        candidate, which :func:`_evidence_weights` turns into a weight
        under the configured strategy.

        Parameters
        ----------
        fetch : Callable[[int], tuple[Tensor, Tensor]]
            A grounding's ``row`` or ``column``.
        query_id : int
            The entity being queried.

        Returns
        -------
        tuple[Tensor, Tensor]
            Candidate indices and parallel evidence weights (may be empty).
        """
        candidates, groundings = fetch(query_id)
        return candidates, _evidence_weights(groundings, self._scoring_strategy)

    def _apply_ac1_side(
        self,
        complement: Tensor,
        side: _GroundedSide,
        query_id: int,
    ) -> None:
        """Fold one AC1 group's contribution into a noisy-or complement.

        Parameters
        ----------
        complement : Tensor
            Running noisy-or complement over candidates, updated in place.
        side : _GroundedSide
            The direction-specific view of the group.
        query_id : int
            The entity being queried (a head for tails, a tail for heads).
        """
        columns, weights = self._row_evidence(side.fetch, query_id)
        if columns.numel() == 0:
            return

        query_confidence = side.query_confidences.get(query_id)
        if query_confidence is not None:
            complement[columns] *= (1.0 - query_confidence) ** weights

        if side.candidate_ids.numel() == 0:
            return

        reachable = torch.isin(side.candidate_ids, columns)
        if not bool(reachable.any()):
            return

        matching_ids = side.candidate_ids[reachable]
        dense_weights = torch.zeros(complement.numel())
        dense_weights[columns] = weights
        complement[matching_ids] *= (
            1.0 - side.candidate_confidences[reachable]
        ) ** dense_weights[matching_ids]

    @staticmethod
    def _build_groundings(
        graph: HeteroGraph,
        results: list[tuple[Rule, RuleMetrics]],
        mode: GroundingMode,
    ) -> dict[BodyChainKey, ChainGrounding]:
        """Build one grounding per unique body chain.

        Parameters
        ----------
        graph : HeteroGraph
            The knowledge graph.
        results : list[tuple[Rule, RuleMetrics]]
            All rules to process.
        mode : GroundingMode
            Whether to materialise each chain product or ground per query.

        Returns
        -------
        dict[BodyChainKey, ChainGrounding]
            Mapping from chain key to its grounding.
        """
        unique_chains = _unique_body_chains(graph, results)

        groundings: dict[BodyChainKey, ChainGrounding] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            for chain_key, matrices in unique_chains.items():
                groundings[chain_key] = _build_grounding(matrices, mode)

        logger.debug(
            "Built %d %s chain groundings from %d rules",
            len(groundings),
            mode.value,
            len(results),
        )
        return groundings

    @staticmethod
    def _build_groups(
        results: list[tuple[Rule, RuleMetrics]],
        chain_products: dict[BodyChainKey, ChainGrounding],
    ) -> tuple[list[_CyclicChainGroup], list[_AC1ChainGroup]]:
        """Group rules by type and body chain, pre-aggregating confidences.

        Parameters
        ----------
        results : list[tuple[Rule, RuleMetrics]]
            All rules with metrics.
        chain_products : dict[BodyChainKey, ChainGrounding]
            Per-chain groundings.

        Returns
        -------
        tuple[list[_CyclicChainGroup], list[_AC1ChainGroup]]
            Cyclic groups and AC1 groups.
        """
        cyclic_confs: dict[BodyChainKey, list[float]] = defaultdict(list)
        ac1_subj: dict[BodyChainKey, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        ac1_obj: dict[BodyChainKey, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for rule, metrics in results:
            match rule.rule_type:
                case RuleType.CYCLIC:
                    key = _body_chain_key(rule)
                    cyclic_confs[key].append(metrics.confidence)
                case RuleType.AC1:
                    key = _body_chain_key(rule)
                    head = rule.head
                    is_subject_grounded = head.subject.kind is TermKind.CONSTANT
                    if is_subject_grounded:
                        entity_id = head.subject.entity_id
                        if entity_id is not None:
                            ac1_subj[key][entity_id].append(metrics.confidence)
                    else:
                        entity_id = head.object_.entity_id
                        if entity_id is not None:
                            ac1_obj[key][entity_id].append(metrics.confidence)
                case RuleType.AC2:
                    pass
                case _ as unreachable:
                    assert_never(unreachable)

        cyclic_groups: list[_CyclicChainGroup] = []
        for key, confs in cyclic_confs.items():
            cyclic_groups.append(
                _CyclicChainGroup(
                    chain=key,
                    grounding=chain_products[key],
                    aggregated_confidence=aggregate_confidence(confs),
                )
            )

        all_ac1_keys = set(ac1_subj.keys()) | set(ac1_obj.keys())
        ac1_groups: list[_AC1ChainGroup] = []
        for key in all_ac1_keys:
            subj_aggregated = {
                eid: aggregate_confidence(confs) for eid, confs in ac1_subj[key].items()
            }
            obj_aggregated = {
                eid: aggregate_confidence(confs) for eid, confs in ac1_obj[key].items()
            }

            subj_ids = list(subj_aggregated.keys())
            subj_confs = list(subj_aggregated.values())
            obj_ids = list(obj_aggregated.keys())
            obj_confs = list(obj_aggregated.values())

            ac1_groups.append(
                _AC1ChainGroup(
                    chain=key,
                    grounding=chain_products[key],
                    subject_grounded=subj_aggregated,
                    object_grounded=obj_aggregated,
                    subject_entity_ids=torch.tensor(subj_ids, dtype=torch.long),
                    subject_confidences=torch.tensor(subj_confs, dtype=torch.float),
                    object_entity_ids=torch.tensor(obj_ids, dtype=torch.long),
                    object_confidences=torch.tensor(obj_confs, dtype=torch.float),
                )
            )

        return cyclic_groups, ac1_groups


def _weight_for_candidate(
    row: tuple[Tensor, Tensor],
    candidate_id: int,
    strategy: ScoringStrategy,
) -> tuple[int, float] | None:
    """Return one candidate's grounding count and evidence weight, if reached.

    Parameters
    ----------
    row : tuple[Tensor, Tensor]
        Candidate indices and grounding counts from a chain grounding.
    candidate_id : int
        The candidate to look for.
    strategy : ScoringStrategy
        The weighting scheme.

    Returns
    -------
    tuple[int, float] | None
        ``(groundings, weight)``, or ``None`` when the chain does not
        reach this candidate.
    """
    indices, groundings = row
    if indices.numel() == 0:
        return None
    position = torch.nonzero(indices == candidate_id, as_tuple=True)[0]
    if position.numel() == 0:
        return None
    count = groundings[position[0]]
    weight = _evidence_weights(count.reshape(1), strategy)[0]
    return int(count.item()), float(weight.item())


def _firing(
    chain: BodyChainKey,
    rule_type: RuleType,
    confidence: float,
    groundings: int,
    weight: float,
) -> RuleFiring:
    """Build a :class:`RuleFiring` with its standalone contribution."""
    return RuleFiring(
        chain=chain,
        rule_type=rule_type,
        confidence=confidence,
        groundings=groundings,
        contribution=1.0 - (1.0 - confidence) ** weight,
    )


def _unique_body_chains(
    graph: HeteroGraph,
    results: list[tuple[Rule, RuleMetrics]],
) -> dict[BodyChainKey, tuple[Tensor, ...]]:
    """Collect the body matrices of each distinct chain, skipping AC2 rules.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph.
    results : list[tuple[Rule, RuleMetrics]]
        All rules to process.

    Returns
    -------
    dict[BodyChainKey, tuple[Tensor, ...]]
        Body matrices per chain, in chain order.
    """
    ac2_count = sum(1 for rule, _ in results if rule.rule_type is RuleType.AC2)
    if ac2_count:
        logger.debug(
            "Skipping %d AC2 rule(s): AC2 rules are evaluated for quality "
            "metrics but do not contribute to predictions (their predictions "
            "are Cartesian products, making individual rule firing untractable).",
            ac2_count,
        )

    chains: dict[BodyChainKey, tuple[Tensor, ...]] = {}
    for rule, _ in results:
        if rule.rule_type is RuleType.AC2:
            continue
        key = _body_chain_key(rule)
        if key not in chains:
            chains[key] = tuple(
                graph.get_csr_matrix(atom.edge_signature) for atom in rule.body
            )
    return chains


def _build_grounding(
    matrices: Sequence[Tensor],
    mode: GroundingMode,
) -> ChainGrounding:
    """Build the grounding for one chain under the chosen mode.

    Parameters
    ----------
    matrices : Sequence[Tensor]
        Body atom adjacency matrices, in chain order.
    mode : GroundingMode
        Whether to materialise the product or ground per query.

    Returns
    -------
    ChainGrounding
        The grounding for this chain.
    """
    match mode:
        case GroundingMode.ON_DEMAND:
            return _OnDemandChain(matrices=tuple(matrices))
        case GroundingMode.MATERIALISED:
            product = matrices[0]
            for matrix in matrices[1:]:
                product = product @ matrix
            return _MaterialisedChain(
                product=product,
                product_transposed=product.t().to_sparse_csr(),
            )
        case _ as unreachable:
            assert_never(unreachable)


def _body_chain_key(rule: Rule) -> BodyChainKey:
    """Extract the body chain key from a rule.

    Parameters
    ----------
    rule : Rule
        The rule to extract the key from.

    Returns
    -------
    BodyChainKey
        Tuple of edge-type triples for each body atom.
    """
    return tuple(atom.edge_signature for atom in rule.body)
