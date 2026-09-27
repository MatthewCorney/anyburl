"""Benchmark RulePredictor init / predict / score_tails / score_heads.

Loads the DBLP dataset, holds out a fraction of the target edges, runs the
AnyBURL pipeline on the remainder, then times each prediction operation and
prints a summary table.

Rules are learned on the training split so the reported link prediction
quality is honest. Fitting on the full graph and then ranking its own edges
measures memorisation instead, and scores far higher.
"""

import random
import time
import warnings

from torch_geometric.datasets import DBLP

from anyburl import (
    AnyBURL,
    AnyBURLConfig,
    RulePredictor,
    SamplingStrategy,
    ScoringStrategy,
    SplitConfig,
    TieHandling,
    split_target_edges,
)
from anyburl.graph import HeteroGraph

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")

TARGET_EDGE_TYPE = ("author", "to", "paper")
DATA_ROOT = "./data/DBLP"

SAMPLE_SIZE = 4000
MAX_WALK_LENGTH = 4
MIN_WALK_LENGTH = 2
MAX_WALK_ATTEMPTS = 1000
MIN_SUPPORT = 3
MIN_CONFIDENCE = 0.001
MIN_HEAD_COVERAGE = 0.005
SEED = 42

NUM_SCORE_QUERIES = 100
TEST_FRACTION = 0.1
NUM_TEST_TRIPLES = 300
K_VALUES = (1, 3, 10)


def main() -> None:
    print("=" * 70)
    print("Prediction Benchmark - DBLP Dataset")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Load dataset and run pipeline
    # ------------------------------------------------------------------
    dataset = DBLP(root=DATA_ROOT)
    data = dataset[0]
    print(f"Loaded DBLP: {data}")

    split = split_target_edges(
        data,
        SplitConfig(target_edge_type=TARGET_EDGE_TYPE, test_fraction=TEST_FRACTION),
    )
    train_data = split.train_data
    test_triples = split.test_triples[:NUM_TEST_TRIPLES]
    print(
        f"Split: {train_data[TARGET_EDGE_TYPE].edge_index.size(1)} train edges, "
        f"{len(split.test_triples)} held out ({len(test_triples)} evaluated)"
    )
    print()

    config = AnyBURLConfig(
        sample_size=SAMPLE_SIZE,
        sampling_strategy=SamplingStrategy.UNIFORM,
        target_edge_type=TARGET_EDGE_TYPE,
        max_walk_length=MAX_WALK_LENGTH,
        min_walk_length=MIN_WALK_LENGTH,
        max_walk_attempts=MAX_WALK_ATTEMPTS,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        min_head_coverage=MIN_HEAD_COVERAGE,
        seed=SEED,
    )

    t0 = time.perf_counter()
    pipeline = AnyBURL(config).fit(train_data)
    fit_elapsed = time.perf_counter() - t0
    print(f"Pipeline fit: {fit_elapsed:.2f}s")
    print(f"  Rules passing filter: {len(pipeline.results)}")
    print()

    if not pipeline.results:
        print("No rules found, cannot benchmark prediction.")
        return

    graph = HeteroGraph(train_data)

    # ------------------------------------------------------------------
    # 2. Time RulePredictor.__init__
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    predictor = RulePredictor(graph, pipeline.results)
    init_elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 3. Time predict()
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    predictions = predictor.predict()
    predict_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    predictions_filtered = predictor.predict(filter_known=True)
    predict_filtered_elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 4. Time score_tails for random heads
    # ------------------------------------------------------------------
    num_heads = graph.node_count("author")
    head_ids = random.sample(range(num_heads), min(NUM_SCORE_QUERIES, num_heads))

    t0 = time.perf_counter()
    for head_id in head_ids:
        predictor.score_tails(head_id)
    score_tails_elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 5. Time score_heads for random tails
    # ------------------------------------------------------------------
    num_tails = graph.node_count("paper")
    tail_ids = random.sample(range(num_tails), min(NUM_SCORE_QUERIES, num_tails))

    t0 = time.perf_counter()
    for tail_id in tail_ids:
        predictor.score_heads(tail_id)
    score_heads_elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()
    print(f"  {'Operation':<35} {'Time':>10} {'Details':>20}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 20}")
    print(f"  {'RulePredictor.__init__':<35} {init_elapsed:>9.3f}s")
    print(
        f"  {'predict()':<35} {predict_elapsed:>9.3f}s"
        f" {len(predictions):>10} predictions"
    )
    print(
        f"  {'predict(filter_known=True)':<35} {predict_filtered_elapsed:>9.3f}s"
        f" {len(predictions_filtered):>10} predictions"
    )
    n_tails = len(head_ids)
    print(
        f"  {'score_tails() x ' + str(n_tails):<35} {score_tails_elapsed:>9.3f}s"
        f" {score_tails_elapsed / n_tails * 1000:>13.1f} ms/call"
    )
    n_heads = len(tail_ids)
    print(
        f"  {'score_heads() x ' + str(n_heads):<35} {score_heads_elapsed:>9.3f}s"
        f" {score_heads_elapsed / n_heads * 1000:>13.1f} ms/call"
    )
    print()

    # ------------------------------------------------------------------
    # 7. Link prediction quality on the held-out split
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Link prediction quality (held-out split)")
    print("=" * 70)
    print()
    print(
        f"  {'scoring':<16} {'ties':<12} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}"
    )
    print(f"  {'-' * 16} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for scoring in ScoringStrategy:
        for ties in (TieHandling.OPTIMISTIC, TieHandling.AVERAGE):
            metrics = pipeline.evaluate_predictions(
                test_triples,
                k_values=K_VALUES,
                tie_handling=ties,
                scoring_strategy=scoring,
            )
            hits = metrics.hits_at_k
            print(
                f"  {scoring.value:<16} {ties.value:<12} {metrics.mrr:>8.4f} "
                f"{hits[1]:>8.4f} {hits[3]:>8.4f} {hits[10]:>8.4f}"
            )
    print()

    # Chain grounding stats
    print("Chain grounding stats:")
    print(f"  Cyclic groups: {len(predictor._cyclic_groups)}")
    print(f"  AC1 groups:    {len(predictor._ac1_groups)}")
    for i, g in enumerate(predictor._cyclic_groups):
        nnz = g.grounding.product.col_indices().numel()
        shape = tuple(g.grounding.product.shape)
        print(
            f"    cyclic[{i}]: shape={shape} nnz={nnz} "
            f"conf={g.aggregated_confidence:.4f}"
        )
    for i, g in enumerate(predictor._ac1_groups):
        nnz = g.grounding.product.col_indices().numel()
        shape = tuple(g.grounding.product.shape)
        print(
            f"    ac1[{i}]: shape={shape} nnz={nnz} "
            f"subj_grounded={len(g.subject_grounded)} "
            f"obj_grounded={len(g.object_grounded)}"
        )
    print()


if __name__ == "__main__":
    main()
