"""Candidate features that mean the same in every relation.

Unlike :class:`~anyburl.aggregation.CandidateFeaturizer`, which adds one
column per body chain, these features summarise the rule evidence by kind, so
one model can be fitted across relations with different rules.
"""

from enum import StrEnum
from typing import assert_never

import torch
from torch import Tensor

from ..evaluation import QueryDirection
from ..prediction import AcyclicEvidence, ChainEvidence, RulePredictor
from ..rule import RuleType
from ..stratification import RelationDegrees
from .features import LOG_SCORE_FLOOR

__all__ = ["SharedFeature", "SharedFeaturizer", "build_shared_features"]

SECOND_STRONGEST: int = 2
"""How many of the strongest contributions ``topk`` keeps."""


class SharedFeature(StrEnum):
    """Relation-independent features, in column order.

    Attributes
    ----------
    RULE_SCORE : str
        Noisy-or of every chain contribution and acyclic rule confidence.
    LOG_RULE_SCORE : str
        ``log(RULE_SCORE + LOG_SCORE_FLOOR)``.
    MAX_CONTRIBUTION : str
        The strongest single chain or acyclic rule.
    SECOND_CONTRIBUTION : str
        The second strongest, zero when fewer than two fire.
    MAX_CYCLIC : str
        The strongest cyclic chain.
    MAX_AC1 : str
        The strongest AC1 chain.
    MAX_ACYCLIC : str
        The strongest acyclic rule.
    LOG_FIRING_CHAINS : str
        ``log1p`` of the number of chains reaching the candidate.
    LOG_ACYCLIC_FIRINGS : str
        ``log1p`` of the number of acyclic rules firing for the candidate.
    LOG_GROUNDINGS : str
        ``log1p`` of the body groundings summed over chains.
    LOG_DEGREE : str
        ``log1p`` of the candidate's training degree in the target relation.
    IS_UNSEEN : str
        1.0 when the candidate has no training edge of the target relation.
    """

    RULE_SCORE = "rule_score"
    LOG_RULE_SCORE = "log_rule_score"
    MAX_CONTRIBUTION = "max_contribution"
    SECOND_CONTRIBUTION = "second_contribution"
    MAX_CYCLIC = "max_cyclic"
    MAX_AC1 = "max_ac1"
    MAX_ACYCLIC = "max_acyclic"
    LOG_FIRING_CHAINS = "log_firing_chains"
    LOG_ACYCLIC_FIRINGS = "log_acyclic_firings"
    LOG_GROUNDINGS = "log_groundings"
    LOG_DEGREE = "log_degree"
    IS_UNSEEN = "is_unseen"


def build_shared_features(
    chains: ChainEvidence,
    chain_types: tuple[RuleType, ...],
    acyclic: AcyclicEvidence,
    candidate_degrees: Tensor,
) -> Tensor:
    """Assemble the relation-independent feature matrix for one query.

    Parameters
    ----------
    chains : ChainEvidence
        Per-chain contributions and groundings over every candidate.
    chain_types : tuple[RuleType, ...]
        The rule type of each row of ``chains``.
    acyclic : AcyclicEvidence
        Acyclic rule evidence over every candidate.
    candidate_degrees : Tensor
        ``(num_candidates,)`` training degree of each candidate.

    Returns
    -------
    Tensor
        ``(num_candidates, len(SharedFeature))`` features.
    """
    contributions = chains.contributions
    complement = torch.prod(1.0 - contributions, dim=0) * (1.0 - acyclic.noisy_or)
    rule_score = 1.0 - complement
    strongest = torch.cat([contributions, acyclic.max_confidence.unsqueeze(0)])
    top = torch.topk(strongest, min(SECOND_STRONGEST, strongest.size(0)), dim=0).values
    second = top[1] if top.size(0) == SECOND_STRONGEST else torch.zeros_like(rule_score)
    columns = [
        rule_score,
        torch.log(rule_score + LOG_SCORE_FLOOR),
        top[0],
        second,
        _max_of_type(contributions, chain_types, RuleType.CYCLIC),
        _max_of_type(contributions, chain_types, RuleType.AC1),
        acyclic.max_confidence,
        torch.log1p((contributions > 0).sum(dim=0).to(rule_score.dtype)),
        torch.log1p(acyclic.firings),
        torch.log1p(chains.groundings.sum(dim=0)),
        torch.log1p(candidate_degrees),
        (candidate_degrees == 0).to(rule_score.dtype),
    ]
    return torch.stack(columns, dim=1)


def _max_of_type(
    contributions: Tensor, chain_types: tuple[RuleType, ...], rule_type: RuleType
) -> Tensor:
    """Return the strongest contribution among chains of ``rule_type``."""
    rows = [index for index, kind in enumerate(chain_types) if kind is rule_type]
    if not rows:
        return torch.zeros(contributions.size(1))
    return contributions[rows].max(dim=0).values


class SharedFeaturizer:
    """Builds relation-independent features against a fitted predictor.

    Parameters
    ----------
    predictor : RulePredictor
        Rules grounded on the graph the features describe.
    degrees : RelationDegrees
        Training degrees of the target relation in that same graph.
    """

    def __init__(self, predictor: RulePredictor, degrees: RelationDegrees) -> None:
        self._predictor = predictor
        self._degrees = degrees
        self._chain_types = tuple(rule_type for _, rule_type in predictor.chains)

    @property
    def score_column(self) -> int:
        """Column of the rule score candidates are sampled by."""
        return list(SharedFeature).index(SharedFeature.RULE_SCORE)

    def feature_names(self) -> tuple[str, ...]:
        """Return the column names of :meth:`featurize`."""
        return tuple(feature.value for feature in SharedFeature)

    def featurize(self, direction: QueryDirection, query_id: int) -> Tensor:
        """Return the feature matrix over every candidate of one query.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` ranks tails of head ``query_id``; ``HEAD`` ranks heads
            of tail ``query_id``.
        query_id : int
            The entity being queried.

        Returns
        -------
        Tensor
            ``(num_candidates, len(SharedFeature))`` features.
        """
        return build_shared_features(
            self._predictor.chain_evidence(direction, query_id),
            self._chain_types,
            self._predictor.acyclic_evidence(direction, query_id),
            self._candidate_degrees(direction),
        )

    def _candidate_degrees(self, direction: QueryDirection) -> Tensor:
        """Return the degrees of the entities ``direction`` ranks."""
        match direction:
            case QueryDirection.TAIL:
                return self._degrees.tail
            case QueryDirection.HEAD:
                return self._degrees.head
            case _ as unreachable:
                assert_never(unreachable)
