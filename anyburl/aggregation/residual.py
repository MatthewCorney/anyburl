"""Per-relation corrections fitted on top of a shared aggregator.

A shared model ranks every relation from features that mean the same
everywhere. A relation with enough validation queries can then fit its own
linear correction, over features such as its per-chain contributions, with
the shared score held fixed as an offset. A strong L2 penalty keeps the
correction near zero where the data are thin.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core import QueryDirection
from .features import QueryFeaturizer
from .model import AggregatorConfig, LinearAggregator
from .training import TrainingSet

__all__ = ["JointFeaturizer", "ResidualAggregator"]


class JointFeaturizer:
    """Places a second featurizer's columns after a base featurizer's.

    Candidates are reached and sampled as the base featurizer decides.

    Parameters
    ----------
    base : QueryFeaturizer
        The features the shared model reads.
    extra : QueryFeaturizer
        The features the correction reads; must rank the same candidates.
    """

    def __init__(self, base: QueryFeaturizer, extra: QueryFeaturizer) -> None:
        self._base = base
        self._extra = extra

    @property
    def score_column(self) -> int:
        """Column of the base featurizer's rule score."""
        return self._base.score_column

    @property
    def num_base_features(self) -> int:
        """Number of leading columns that come from the base featurizer."""
        return len(self._base.feature_names())

    def feature_names(self) -> tuple[str, ...]:
        """Return the base column names, then the extra ones."""
        return (*self._base.feature_names(), *self._extra.feature_names())

    def num_candidates(self, direction: QueryDirection) -> int:
        """Return how many entities a query in ``direction`` ranks."""
        return self._base.num_candidates(direction)

    def reached(self, direction: QueryDirection, query_id: int) -> Tensor:
        """Return the candidates the base featurizer's rules reach."""
        return self._base.reached(direction, query_id)

    def featurize(
        self,
        direction: QueryDirection,
        query_id: int,
        *,
        candidates: Tensor | None = None,
    ) -> Tensor:
        """Return the base features with the extra features appended.

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
            ``(num_candidates, num_base + num_extra)`` features.
        """
        return torch.cat(
            [
                self._base.featurize(direction, query_id, candidates=candidates),
                self._extra.featurize(direction, query_id, candidates=candidates),
            ],
            dim=1,
        )


@dataclass(frozen=True, slots=True)
class ResidualAggregator:
    """A shared model's score plus a fitted correction.

    Reads features laid out as :class:`JointFeaturizer` lays them out.

    Parameters
    ----------
    base : LinearAggregator
        The shared model, over the leading columns.
    residual : LinearAggregator
        The correction, over the remaining columns.
    """

    base: LinearAggregator
    residual: LinearAggregator

    @property
    def num_features(self) -> int:
        """Number of feature columns the model reads."""
        return self.base.num_features + self.residual.num_features

    @classmethod
    def fit(
        cls,
        training_set: TrainingSet,
        base: LinearAggregator,
        config: AggregatorConfig,
    ) -> "ResidualAggregator":
        """Fit a correction to ``base`` on one relation's queries.

        Parameters
        ----------
        training_set : TrainingSet
            Joint features, base columns first, answer first in each query.
        base : LinearAggregator
            The fitted shared model, held fixed.
        config : AggregatorConfig
            Regularisation of the correction; a strong penalty keeps it
            close to the shared model.

        Returns
        -------
        ResidualAggregator
            The shared model with the fitted correction.
        """
        num_base = base.num_features
        offsets = base.score(training_set.features[..., :num_base])
        extra = TrainingSet(training_set.features[..., num_base:], training_set.mask)
        residual = LinearAggregator.fit(extra, config, offsets=offsets)
        return cls(base=base, residual=residual)

    def score(self, features: Tensor) -> Tensor:
        """Return the shared score plus the correction.

        Parameters
        ----------
        features : Tensor
            ``(..., num_features)`` joint features.

        Returns
        -------
        Tensor
            One score per candidate row.
        """
        num_base = self.base.num_features
        return self.base.score(features[..., :num_base]) + self.residual.score(
            features[..., num_base:]
        )
