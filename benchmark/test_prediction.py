"""Benchmarks for RulePredictor init, predict, and per-entity score queries.

``predict()`` scores every head entity, so its benchmarks use the small graph
only; single-query operations run on every graph size.
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
