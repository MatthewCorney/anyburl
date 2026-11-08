"""A linear listwise model that ranks candidates from their features.

Each query's candidates compete in one softmax, and the model maximises the
probability of the correct answer (a conditional logit). Features are
standardised on the training rows, so the L2 penalty treats them alike and
the fitted weights are comparable across features.
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ..exceptions import ConfigurationError
from .training import TARGET_POSITION, TrainingSet

__all__ = ["Aggregator", "AggregatorConfig", "LinearAggregator", "listwise_loss"]

DEFAULT_L2_PENALTY: float = 1e-3
"""Weight on the squared norm of the standardised weights by default."""

DEFAULT_MAX_ITERATIONS: int = 200
"""L-BFGS iteration budget by default."""

MIN_FEATURE_SCALE: float = 1e-8
"""Standard deviations below this are treated as constant features."""

LINE_SEARCH: str = "strong_wolfe"
"""L-BFGS line search; without one the step size can diverge."""


class Aggregator(Protocol):
    """Scores candidates from their feature rows."""

    @property
    def num_features(self) -> int:
        """Number of feature columns the model reads."""
        ...

    def score(self, features: Tensor) -> Tensor:
        """Return one score per candidate row; higher ranks first."""
        ...


@dataclass(frozen=True, slots=True)
class AggregatorConfig:
    """How the linear aggregator is fitted.

    Parameters
    ----------
    l2_penalty : float
        Weight on the squared norm of the weights. Must not be negative.
    max_iterations : int
        L-BFGS iteration budget. Must be positive.

    Raises
    ------
    ConfigurationError
        If either value is out of range.
    """

    l2_penalty: float = DEFAULT_L2_PENALTY
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.l2_penalty < 0.0:
            raise ConfigurationError(
                f"l2_penalty must not be negative, got {self.l2_penalty}"
            )
        if self.max_iterations < 1:
            raise ConfigurationError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )


@dataclass(frozen=True, slots=True)
class LinearAggregator:
    """Scores candidates as a weighted sum of standardised features.

    Parameters
    ----------
    weights : Tensor
        ``(num_features,)`` weights on standardised features.
    mean : Tensor
        ``(num_features,)`` training mean of each feature.
    scale : Tensor
        ``(num_features,)`` training standard deviation, 1.0 where constant.
    """

    weights: Tensor
    mean: Tensor
    scale: Tensor

    @property
    def num_features(self) -> int:
        """Number of feature columns the model reads."""
        return self.weights.numel()

    @classmethod
    def fit(
        cls,
        training_set: TrainingSet,
        config: AggregatorConfig,
        *,
        offsets: Tensor | None = None,
    ) -> "LinearAggregator":
        """Fit weights that rank each query's correct answer first.

        Parameters
        ----------
        training_set : TrainingSet
            Padded candidate features, correct answer first in each query.
        config : AggregatorConfig
            Regularisation and iteration budget.
        offsets : Tensor | None
            ``(num_queries, max_candidates)`` fixed scores the fitted ones
            are added to, such as another model's; ``None`` adds nothing.

        Returns
        -------
        LinearAggregator
            The fitted model.
        """
        rows = training_set.features[training_set.mask]
        mean = rows.mean(dim=0)
        std = rows.std(dim=0) if rows.size(0) > 1 else torch.zeros_like(mean)
        scale = torch.where(std > MIN_FEATURE_SCALE, std, torch.ones_like(std))
        standardised = (training_set.features - mean) / scale
        base = offsets if offsets is not None else torch.zeros(training_set.mask.shape)

        weights = torch.zeros(rows.size(1), requires_grad=True)
        optimiser = torch.optim.LBFGS(
            [weights], max_iter=config.max_iterations, line_search_fn=LINE_SEARCH
        )

        def closure() -> Tensor:
            optimiser.zero_grad()
            loss = listwise_loss(standardised, training_set.mask, weights, offsets=base)
            loss = loss + config.l2_penalty * weights.square().sum()
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimiser.step(closure)  # type: ignore[no-untyped-call]
        return cls(weights=weights.detach(), mean=mean, scale=scale)

    def score(self, features: Tensor) -> Tensor:
        """Score candidates; higher ranks first.

        Parameters
        ----------
        features : Tensor
            ``(num_candidates, num_features)`` raw features.

        Returns
        -------
        Tensor
            ``(num_candidates,)`` scores.

        Raises
        ------
        ConfigurationError
            If the feature count differs from the one fitted on.
        """
        if features.size(-1) != self.weights.numel():
            raise ConfigurationError(
                f"expected {self.weights.numel()} features, got {features.size(-1)}"
            )
        return ((features - self.mean) / self.scale) @ self.weights


def listwise_loss(
    features: Tensor,
    mask: Tensor,
    weights: Tensor,
    *,
    offsets: Tensor | None = None,
) -> Tensor:
    """Return the mean negative log-probability of each query's answer.

    Parameters
    ----------
    features : Tensor
        ``(num_queries, max_candidates, num_features)``, answer first.
    mask : Tensor
        ``(num_queries, max_candidates)`` bool, ``False`` on padding.
    weights : Tensor
        ``(num_features,)`` model weights.
    offsets : Tensor | None
        ``(num_queries, max_candidates)`` scores added to the logits.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    logits = features @ weights
    if offsets is not None:
        logits = logits + offsets
    logits = logits.masked_fill(~mask, float("-inf"))
    log_probabilities = logits[:, TARGET_POSITION] - torch.logsumexp(logits, dim=1)
    return -log_probabilities.mean()
