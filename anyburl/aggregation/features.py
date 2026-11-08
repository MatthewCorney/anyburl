"""Per-candidate features a learned aggregator ranks from.

Each query yields a ``(num_candidates, num_features)`` matrix: summary
features of the rule evidence and the candidate's training degree, then one
contribution column per body chain. A linear model over ``LOG_RULE_SCORE``
and ``LOG_DEGREE`` alone can express the fixed popularity discount
``score / (degree + 1) ** alpha``, so the learned model generalises it.
"""

from enum import StrEnum
from typing import Protocol

import torch
from torch import Tensor

from ..core import QueryDirection
from ..prediction import ChainEvidence, RuleEvidence, format_chain
from ..stratification import RelationDegrees

__all__ = [
    "CandidateFeature",
    "CandidateFeaturizer",
    "QueryFeaturizer",
    "build_features",
    "column_of",
]

LOG_SCORE_FLOOR: float = 1e-6
"""Added to rule scores before the log, so unreached candidates stay finite."""


class CandidateFeature(StrEnum):
    """Summary features, in column order, before the per-chain columns.

    Attributes
    ----------
    RULE_SCORE : str
        The predictor's own score, ``1 - prod(1 - contributions)``.
    LOG_RULE_SCORE : str
        ``log(RULE_SCORE + LOG_SCORE_FLOOR)``.
    MAX_CONTRIBUTION : str
        The strongest single chain's contribution.
    FIRING_CHAINS : str
        Number of chains reaching the candidate.
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
    FIRING_CHAINS = "firing_chains"
    LOG_GROUNDINGS = "log_groundings"
    LOG_DEGREE = "log_degree"
    IS_UNSEEN = "is_unseen"


def column_of(feature: CandidateFeature) -> int:
    """Return the column index of a summary feature in the feature matrix.

    Parameters
    ----------
    feature : CandidateFeature
        The summary feature.

    Returns
    -------
    int
        Its column in :func:`build_features` output.
    """
    return list(CandidateFeature).index(feature)


def build_features(evidence: ChainEvidence, candidate_degrees: Tensor) -> Tensor:
    """Assemble the feature matrix for one query.

    Parameters
    ----------
    evidence : ChainEvidence
        Per-chain contributions and groundings over every candidate.
    candidate_degrees : Tensor
        ``(num_candidates,)`` training degree of each candidate.

    Returns
    -------
    Tensor
        ``(num_candidates, len(CandidateFeature) + num_chains)`` features.
    """
    contributions = evidence.contributions
    rule_score = 1.0 - torch.prod(1.0 - contributions, dim=0)
    max_contribution = (
        contributions.max(dim=0).values
        if contributions.size(0) > 0
        else torch.zeros_like(candidate_degrees)
    )
    summary = torch.stack(
        [
            rule_score,
            torch.log(rule_score + LOG_SCORE_FLOOR),
            max_contribution,
            (contributions > 0).sum(dim=0).to(rule_score.dtype),
            torch.log1p(evidence.groundings.sum(dim=0)),
            torch.log1p(candidate_degrees),
            (candidate_degrees == 0).to(rule_score.dtype),
        ],
        dim=1,
    )
    return torch.cat([summary, contributions.T], dim=1)


class QueryFeaturizer(Protocol):
    """Builds a feature matrix over the candidates of a query."""

    @property
    def score_column(self) -> int:
        """Column of the rule score candidates are sampled by."""
        ...

    def feature_names(self) -> tuple[str, ...]:
        """Return the column names of :meth:`featurize`."""
        ...

    def num_candidates(self, direction: QueryDirection) -> int:
        """Return how many entities a query in ``direction`` ranks."""
        ...

    def reached(self, direction: QueryDirection, query_id: int) -> Tensor:
        """Return the sorted ids of candidates some rule scores positively."""
        ...

    def featurize(
        self,
        direction: QueryDirection,
        query_id: int,
        *,
        candidates: Tensor | None = None,
    ) -> Tensor:
        """Return ``(len(candidates), num_features)`` features of one query."""
        ...


class CandidateFeaturizer:
    """Builds per-candidate features from one relation's rule evidence.

    Parameters
    ----------
    evidence : RuleEvidence
        Rules grounded on the graph the features describe.
    degrees : RelationDegrees
        Training degrees of the target relation in that same graph.
    """

    def __init__(self, evidence: RuleEvidence, degrees: RelationDegrees) -> None:
        self._evidence = evidence
        self._degrees = degrees

    @property
    def score_column(self) -> int:
        """Column of the rule score candidates are sampled by."""
        return column_of(CandidateFeature.RULE_SCORE)

    def feature_names(self) -> tuple[str, ...]:
        """Return the column names of :meth:`featurize`.

        Returns
        -------
        tuple[str, ...]
            Summary feature names followed by one name per body chain.
        """
        chain_names = tuple(
            f"chain[{rule_type.value}]: {format_chain(chain)}"
            for chain, rule_type in self._evidence.chains
        )
        return (*(feature.value for feature in CandidateFeature), *chain_names)

    def num_candidates(self, direction: QueryDirection) -> int:
        """Return how many entities a query in ``direction`` ranks."""
        return self._evidence.num_candidates(direction)

    def reached(self, direction: QueryDirection, query_id: int) -> Tensor:
        """Return the sorted ids of candidates some rule scores positively."""
        return self._evidence.reached_candidates(direction, query_id)

    def featurize(
        self,
        direction: QueryDirection,
        query_id: int,
        *,
        candidates: Tensor | None = None,
    ) -> Tensor:
        """Return the feature matrix over the candidates of one query.

        Parameters
        ----------
        direction : QueryDirection
            ``TAIL`` ranks tails of head ``query_id``; ``HEAD`` ranks heads
            of tail ``query_id``.
        query_id : int
            The entity being queried.
        candidates : Tensor | None
            Unique ids of the candidates to featurize, in row order.
            ``None`` featurizes every candidate.

        Returns
        -------
        Tensor
            ``(num_candidates, num_features)`` features.
        """
        evidence = self._evidence.chain_evidence(
            direction, query_id, candidates=candidates
        )
        return build_features(
            evidence, select_degrees(self._degrees, direction, candidates)
        )


def select_degrees(
    degrees: RelationDegrees, direction: QueryDirection, candidates: Tensor | None
) -> Tensor:
    """Return the training degrees of ``candidates``, or of every candidate.

    Parameters
    ----------
    degrees : RelationDegrees
        Training degrees of the target relation.
    direction : QueryDirection
        The query direction, which picks head or tail degrees.
    candidates : Tensor | None
        Candidate ids; ``None`` selects all.

    Returns
    -------
    Tensor
        One degree per selected candidate.
    """
    all_degrees = degrees.of_candidates(direction)
    return all_degrees if candidates is None else all_degrees[candidates]
