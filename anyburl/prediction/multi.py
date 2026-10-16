"""Apply a rule set that spans several relations."""

from ..baselines import ZeroScorer
from ..core import EdgeTypeTuple, EntityScorer
from ..exceptions import ConfigurationError
from ..graph import HeteroGraph
from ..metrics import RuleSet
from .predictor import RulePredictor
from .records import Prediction, PredictionConfig

__all__ = ["MultiRelationPredictor"]


class MultiRelationPredictor:
    """Scores queries of any relation with the rules learned for it.

    One :class:`RulePredictor` is built per relation, on first use. A
    relation with no rules scores every candidate zero.

    Parameters
    ----------
    graph : HeteroGraph
        The knowledge graph to ground rules against.
    rule_set : RuleSet
        Rules for one or more relations.
    config : PredictionConfig | None
        Scoring, aggregation and grounding settings shared by every
        relation. ``None`` uses the defaults.
    """

    def __init__(
        self,
        graph: HeteroGraph,
        rule_set: RuleSet,
        *,
        config: PredictionConfig | None = None,
    ) -> None:
        self._graph = graph
        self._rule_set = rule_set
        self._config = config
        self._predictors: dict[EdgeTypeTuple, RulePredictor] = {}

    @property
    def edge_types(self) -> tuple[EdgeTypeTuple, ...]:
        """Return the relations that have rules, in learned order."""
        return self._rule_set.head_edge_types()

    def predictor_for(self, edge_type: EdgeTypeTuple) -> RulePredictor:
        """Return the predictor for one relation.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The relation to predict.

        Returns
        -------
        RulePredictor
            A predictor over that relation's rules.

        Raises
        ------
        ConfigurationError
            If no rules predict ``edge_type``.
        """
        predictor = self._predictors.get(edge_type)
        if predictor is None:
            results = self._rule_set.for_edge_type(edge_type)
            if not results:
                raise ConfigurationError(f"no rules predict {edge_type!r}")
            predictor = RulePredictor(self._graph, results, config=self._config)
            self._predictors[edge_type] = predictor
        return predictor

    def scorer_for(self, edge_type: EdgeTypeTuple) -> EntityScorer:
        """Return a scorer for one relation, zero everywhere if it has no rules.

        Parameters
        ----------
        edge_type : EdgeTypeTuple
            The relation to score.

        Returns
        -------
        EntityScorer
            The relation's predictor, or a :class:`ZeroScorer`.
        """
        if edge_type in self.edge_types:
            return self.predictor_for(edge_type)
        return ZeroScorer(self._graph, edge_type)

    def predict(
        self, *, filter_known: bool = False, min_score: float = 0.0
    ) -> list[Prediction]:
        """Score every candidate pair of every relation that has rules.

        Parameters
        ----------
        filter_known : bool
            If ``True``, exclude pairs already present in the graph.
        min_score : float
            Pairs scoring at or below this value are excluded.

        Returns
        -------
        list[Prediction]
            Predictions of all relations, sorted by score descending.
        """
        predictions = [
            prediction
            for edge_type in self.edge_types
            for prediction in self.predictor_for(edge_type).predict(
                filter_known=filter_known, min_score=min_score
            )
        ]
        predictions.sort(key=lambda p: p.score, reverse=True)
        return predictions
