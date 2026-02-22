"""Benchmarks for RulePredictor init, predict, and per-entity score queries.

``test_predict`` and ``test_predict_filter_known`` use the session-scoped
``small_graph`` fixture (500 nodes/type) because ``predict()`` iterates
over every head entity.  For a 50 000-node graph that would be
50 000 ``score_tails`` calls per benchmark round, making the suite
impractically slow.

``test_predictor_init``, ``test_score_tails``, and ``test_score_heads``
use the parametrised ``graph`` fixture (small/medium/large) because those
operations are single-query and scale well.
"""

from anyburl.prediction import RulePredictor

from ._helpers import make_ac1_rule, make_cyclic_rule, make_metrics


def test_predictor_init(benchmark, graph):
    """Benchmark RulePredictor.__init__ (chain-product matrix computation)."""
    results = [
        (make_cyclic_rule(), make_metrics()),
        (make_ac1_rule(), make_metrics()),
    ]
    benchmark(RulePredictor, graph, results)


def test_predict(benchmark, small_graph):
    """Benchmark full predict() on a small graph (500 nodes/type)."""
    predictor = RulePredictor(small_graph, [(make_cyclic_rule(), make_metrics())])
    benchmark(predictor.predict)


def test_predict_filter_known(benchmark, small_graph):
    """Benchmark predict(filter_known=True) on a small graph."""
    predictor = RulePredictor(small_graph, [(make_cyclic_rule(), make_metrics())])
    benchmark(predictor.predict, filter_known=True)


def test_score_tails(benchmark, graph):
    """Benchmark score_tails() for a single head entity."""
    predictor = RulePredictor(graph, [(make_cyclic_rule(), make_metrics())])
    benchmark(predictor.score_tails, 0)


def test_score_heads(benchmark, graph):
    """Benchmark score_heads() for a single tail entity."""
    predictor = RulePredictor(graph, [(make_cyclic_rule(), make_metrics())])
    benchmark(predictor.score_heads, 0)
