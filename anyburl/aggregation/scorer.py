"""Rank candidates with a fitted aggregator, as an entity scorer."""

from typing import assert_never

import torch
from torch import Tensor

from ..core import QueryDirection
from ..exceptions import ConfigurationError
from .features import QueryFeaturizer
from .model import LinearAggregator
from .training import CandidateScope

__all__ = ["UNREACHED_MARGIN", "AggregatedScorer"]

UNREACHED_MARGIN: float = 1.0
"""How far below the lowest reached score unreached candidates are placed."""


class AggregatedScorer:
    """Scores the candidates of a query with a learned aggregator.

    Satisfies :class:`~anyburl.core.EntityScorer`.

    Parameters
    ----------
    featurizer : QueryFeaturizer
        Featurizes queries; its rules must be the ones the model was fitted
        on, so the per-chain columns line up.
    aggregator : LinearAggregator
        The fitted model.
    scope : CandidateScope
        Which candidates the model ranks. With ``REACHED`` the rest tie
        below every reached candidate.

    Raises
    ------
    ConfigurationError
        If the featurizer's feature count differs from the model's.
    """

    def __init__(
        self,
        featurizer: QueryFeaturizer,
        aggregator: LinearAggregator,
        *,
        scope: CandidateScope = CandidateScope.ALL,
    ) -> None:
        num_features = len(featurizer.feature_names())
        if num_features != aggregator.weights.numel():
            raise ConfigurationError(
                f"featurizer yields {num_features} features but the aggregator "
                f"was fitted on {aggregator.weights.numel()}"
            )
        self._featurizer = featurizer
        self._aggregator = aggregator
        self._scope = scope

    def score_tails(self, head_id: int) -> Tensor:
        """Return learned scores over all candidate tails of ``head_id``."""
        return self._score(QueryDirection.TAIL, head_id)

    def score_heads(self, tail_id: int) -> Tensor:
        """Return learned scores over all candidate heads of ``tail_id``."""
        return self._score(QueryDirection.HEAD, tail_id)

    def _score(self, direction: QueryDirection, query_id: int) -> Tensor:
        """Return one score per candidate of a query, as the scope dictates."""
        match self._scope:
            case CandidateScope.ALL:
                features = self._featurizer.featurize(direction, query_id)
                return self._aggregator.score(features)
            case CandidateScope.REACHED:
                return self._score_reached(direction, query_id)
            case _ as unreachable:
                assert_never(unreachable)

    def _score_reached(self, direction: QueryDirection, query_id: int) -> Tensor:
        """Score reached candidates; tie the rest below the lowest of them."""
        scores = torch.zeros(self._featurizer.num_candidates(direction))
        reached = self._featurizer.reached(direction, query_id)
        if reached.numel() == 0:
            return scores
        learned = self._aggregator.score(
            self._featurizer.featurize(direction, query_id, candidates=reached)
        )
        scores.fill_(float(learned.min().item()) - UNREACHED_MARGIN)
        scores[reached] = learned
        return scores
