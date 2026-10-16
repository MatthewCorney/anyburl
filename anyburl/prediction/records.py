"""Prediction settings and the records a predictor returns."""

from dataclasses import dataclass

from torch import Tensor

from ..exceptions import ConfigurationError
from ..rule import RuleType
from .grounding import BodyChainKey, GroundingMode
from .scoring import NO_UNSEEN_NEGATIVES, ConfidenceAggregation, ScoringStrategy

__all__ = [
    "ChainEvidence",
    "Prediction",
    "PredictionConfig",
    "RuleFiring",
    "format_chain",
]


def format_chain(chain: BodyChainKey) -> str:
    """Spell a body chain with both endpoints of every edge type.

    Parameters
    ----------
    chain : BodyChainKey
        The edge types of a rule body, in order.

    Returns
    -------
    str
        For example ``"person_born_in_city -> city_near_city"``.
    """
    return " -> ".join(f"{src}_{relation}_{dst}" for src, relation, dst in chain)


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    """How a predictor scores candidates and grounds its chains.

    Parameters
    ----------
    scoring_strategy : ScoringStrategy
        How rule confidences are weighted against a candidate.
    grounding_mode : GroundingMode
        Whether body chains are materialised, grounded per query, or chosen
        between by size.
    aggregation : ConfidenceAggregation
        How the contributions of different body chains combine.
    unseen_negatives : int
        Assumed incorrect predictions added to each rule's denominator
        before scoring. Zero scores rules by their raw confidence.

    Raises
    ------
    ConfigurationError
        If ``unseen_negatives`` is negative.
    """

    scoring_strategy: ScoringStrategy = ScoringStrategy.PATH_WEIGHTED
    grounding_mode: GroundingMode = GroundingMode.AUTO
    aggregation: ConfidenceAggregation = ConfidenceAggregation.NOISY_OR
    unseen_negatives: int = NO_UNSEEN_NEGATIVES

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.unseen_negatives < 0:
            raise ConfigurationError(
                f"unseen_negatives must be non-negative, got {self.unseen_negatives}"
            )


@dataclass(frozen=True, slots=True)
class RuleFiring:
    """One body chain's contribution to a single predicted pair.

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

        Edge types are spelled with both endpoints, because one relation
        name may be shared by several edge types.
        """
        return (
            f"{format_chain(self.chain)} [{self.rule_type.value}] "
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
        Noisy-or aggregated confidence across all firing rules.
    explanations : tuple[RuleFiring, ...]
        The rule firings behind the score, strongest first. Only
        :meth:`~anyburl.prediction.RulePredictor.top_tails` fills this in.
    """

    head_id: int
    tail_id: int
    head_type: str
    tail_type: str
    relation: str
    score: float
    explanations: tuple[RuleFiring, ...] = ()


@dataclass(frozen=True, slots=True)
class AcyclicEvidence:
    """The acyclic rules firing for every candidate of a query.

    Parameters
    ----------
    max_confidence : Tensor
        ``(num_candidates,)`` confidence of the strongest firing rule.
    noisy_or : Tensor
        ``(num_candidates,)`` ``1 - prod(1 - confidence)`` over firing rules.
    firings : Tensor
        ``(num_candidates,)`` number of rules firing.
    """

    max_confidence: Tensor
    noisy_or: Tensor
    firings: Tensor


@dataclass(frozen=True, slots=True)
class ChainEvidence:
    """Each body chain's separate contribution to every candidate of a query.

    Rows follow :attr:`~anyburl.prediction.RulePredictor.chains`.

    Parameters
    ----------
    contributions : Tensor
        ``(num_chains, num_candidates)`` probability mass each chain alone
        gives each candidate. The predictor's score is
        ``1 - prod(1 - contributions)`` over chains.
    groundings : Tensor
        ``(num_chains, num_candidates)`` body groundings reaching each
        candidate, zero where the chain does not reach it.
    """

    contributions: Tensor
    groundings: Tensor
