"""Rank candidates with a fitted aggregator, as an entity scorer."""

from torch import Tensor

from ..core import QueryDirection
from ..exceptions import ConfigurationError
from .features import QueryFeaturizer
from .model import LinearAggregator

__all__ = ["AggregatedScorer"]


class AggregatedScorer:
    """Scores every candidate of a query with a learned aggregator.

    Satisfies :class:`~anyburl.core.EntityScorer`.

    Parameters
    ----------
    featurizer : QueryFeaturizer
        Featurizes queries; its rules must be the ones the model was fitted
        on, so the per-chain columns line up.
    aggregator : LinearAggregator
        The fitted model.

    Raises
    ------
    ConfigurationError
        If the featurizer's feature count differs from the model's.
    """

    def __init__(
        self, featurizer: QueryFeaturizer, aggregator: LinearAggregator
    ) -> None:
        num_features = len(featurizer.feature_names())
        if num_features != aggregator.weights.numel():
            raise ConfigurationError(
                f"featurizer yields {num_features} features but the aggregator "
                f"was fitted on {aggregator.weights.numel()}"
            )
        self._featurizer = featurizer
        self._aggregator = aggregator

    def score_tails(self, head_id: int) -> Tensor:
        """Return learned scores over all candidate tails of ``head_id``."""
        return self._aggregator.score(
            self._featurizer.featurize(QueryDirection.TAIL, head_id)
        )

    def score_heads(self, tail_id: int) -> Tensor:
        """Return learned scores over all candidate heads of ``tail_id``."""
        return self._aggregator.score(
            self._featurizer.featurize(QueryDirection.HEAD, tail_id)
        )
