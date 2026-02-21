"""Rule prediction: ground rules against the graph and aggregate scores.

Groups rules by body chain to avoid redundant sparse matmul computation.
On DBLP there are only ~36 unique chains across thousands of rules, so
this yields ~100x fewer matmul calls compared to per-rule grounding.

Prediction iterates per head entity, calling ``score_tails()`` which
uses vectorized ``torch.isin`` operations on pre-computed tensors.
"""

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import assert_never

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


@dataclass(frozen=True, slots=True)
class _ChainProduct:
    """Cached product matrix for a unique body chain.

    Parameters
    ----------
    chain_key : BodyChainKey
        The edge-type sequence that produced this product.
    product : Tensor
        Sparse CSR float tensor, computed once via chain matmul.
    product_transposed : Tensor
        Transposed product for reverse-direction queries.
    """

    chain_key: BodyChainKey
    product: Tensor
    product_transposed: Tensor


@dataclass(frozen=True, slots=True)
class _CyclicChainGroup:
    """All cyclic rules sharing a body chain, pre-aggregated.

    Parameters
    ----------
    chain_product : _ChainProduct
        The shared product matrix.
    aggregated_confidence : float
        Noisy-or of all rule confidences in this group.
    """

    chain_product: _ChainProduct
    aggregated_confidence: float


@dataclass(frozen=True, slots=True)
class _AC1ChainGroup:
    """All AC1 rules sharing a body chain, indexed by grounded entity.

    Parameters
    ----------
    chain_product : _ChainProduct
        The shared product matrix.
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

    chain_product: _ChainProduct
    subject_grounded: dict[int, float]
    object_grounded: dict[int, float]
    subject_entity_ids: Tensor
    subject_confidences: Tensor
    object_entity_ids: Tensor
    object_confidences: Tensor


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
        Noisy-or aggregated confidence across all firing rules.
    """

    head_id: int
    tail_id: int
    head_type: str
    tail_type: str
    relation: str
    score: float


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

    Raises
    ------
    ValueError
        If rules predict different head relations or results is empty.
    """

    def __init__(
        self,
        graph: HeteroGraph,
        results: list[tuple[Rule, RuleMetrics]],
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

        first_rule = results[0][0]
        self._head_type = first_rule.head.subject.node_type
        self._tail_type = first_rule.head.object_.node_type
        self._relation = first_rule.head.relation
        self._head_edge_type: EdgeTypeTuple = first_rule.head.edge_signature

        self._num_heads = graph.node_count(self._head_type)
        self._num_tails = graph.node_count(self._tail_type)

        t_chain = time.perf_counter()
        chain_products = self._build_chain_products(graph, results)
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
            cols = self._extract_csr_row(cyc_group.chain_product.product, head_id)
            if cols.numel() > 0:
                complement[cols] *= 1.0 - cyc_group.aggregated_confidence

        for ac1_group in self._ac1_groups:
            cols = self._extract_csr_row(ac1_group.chain_product.product, head_id)
            if cols.numel() == 0:
                continue

            subj_conf = ac1_group.subject_grounded.get(head_id)
            if subj_conf is not None:
                complement[cols] *= 1.0 - subj_conf

            if ac1_group.object_entity_ids.numel() > 0:
                in_reachable = torch.isin(ac1_group.object_entity_ids, cols)
                if in_reachable.any():
                    matching_eids = ac1_group.object_entity_ids[in_reachable]
                    matching_confs = ac1_group.object_confidences[in_reachable]
                    complement[matching_eids] *= 1.0 - matching_confs

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
            rows = self._extract_csr_row(
                cyc_group.chain_product.product_transposed, tail_id
            )
            if rows.numel() > 0:
                complement[rows] *= 1.0 - cyc_group.aggregated_confidence

        for ac1_group in self._ac1_groups:
            rows = self._extract_csr_row(
                ac1_group.chain_product.product_transposed, tail_id
            )
            if rows.numel() == 0:
                continue

            obj_conf = ac1_group.object_grounded.get(tail_id)
            if obj_conf is not None:
                complement[rows] *= 1.0 - obj_conf

            if ac1_group.subject_entity_ids.numel() > 0:
                in_reachable = torch.isin(ac1_group.subject_entity_ids, rows)
                if in_reachable.any():
                    matching_eids = ac1_group.subject_entity_ids[in_reachable]
                    matching_confs = ac1_group.subject_confidences[in_reachable]
                    complement[matching_eids] *= 1.0 - matching_confs

        logger.debug(
            "score_heads(tail=%d): %.3fs", tail_id, time.perf_counter() - t_start
        )
        return 1.0 - complement

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

    @staticmethod
    def _extract_csr_row(csr_matrix: Tensor, row_id: int) -> Tensor:
        """Extract column indices for a single row from a CSR matrix.

        Parameters
        ----------
        csr_matrix : Tensor
            A sparse CSR tensor.
        row_id : int
            The row to extract.

        Returns
        -------
        Tensor
            1-D tensor of column indices (may be empty).
        """
        crow = csr_matrix.crow_indices()
        col = csr_matrix.col_indices()
        start = int(crow[row_id].item())
        end = int(crow[row_id + 1].item())
        return col[start:end]

    @staticmethod
    def _build_chain_products(
        graph: HeteroGraph,
        results: list[tuple[Rule, RuleMetrics]],
    ) -> dict[BodyChainKey, _ChainProduct]:
        """Compute product matrices for each unique body chain.

        Parameters
        ----------
        graph : HeteroGraph
            The knowledge graph.
        results : list[tuple[Rule, RuleMetrics]]
            All rules to process.

        Returns
        -------
        dict[BodyChainKey, _ChainProduct]
            Mapping from chain key to cached product.
        """
        unique_chains: dict[BodyChainKey, list[Tensor]] = {}

        for rule, _ in results:
            if rule.rule_type is RuleType.AC2:
                continue
            key = _body_chain_key(rule)
            if key not in unique_chains:
                matrices = [
                    graph.get_csr_matrix(atom.edge_signature) for atom in rule.body
                ]
                unique_chains[key] = matrices

        products: dict[BodyChainKey, _ChainProduct] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")
            for chain_key, matrices in unique_chains.items():
                product = matrices[0]
                for matrix in matrices[1:]:
                    product = product @ matrix

                product_t = product.t().to_sparse_csr()

                products[chain_key] = _ChainProduct(
                    chain_key=chain_key,
                    product=product,
                    product_transposed=product_t,
                )

        logger.debug(
            "Built %d unique chain products from %d rules",
            len(products),
            len(results),
        )
        return products

    @staticmethod
    def _build_groups(
        results: list[tuple[Rule, RuleMetrics]],
        chain_products: dict[BodyChainKey, _ChainProduct],
    ) -> tuple[list[_CyclicChainGroup], list[_AC1ChainGroup]]:
        """Group rules by type and body chain, pre-aggregating confidences.

        Parameters
        ----------
        results : list[tuple[Rule, RuleMetrics]]
            All rules with metrics.
        chain_products : dict[BodyChainKey, _ChainProduct]
            Pre-computed product matrices.

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
                    chain_product=chain_products[key],
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
                    chain_product=chain_products[key],
                    subject_grounded=subj_aggregated,
                    object_grounded=obj_aggregated,
                    subject_entity_ids=torch.tensor(subj_ids, dtype=torch.long),
                    subject_confidences=torch.tensor(subj_confs, dtype=torch.float),
                    object_entity_ids=torch.tensor(obj_ids, dtype=torch.long),
                    object_confidences=torch.tensor(obj_confs, dtype=torch.float),
                )
            )

        return cyclic_groups, ac1_groups


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
