"""Benchmark RulePredictor init / predict / score_tails / score_heads.

Loads the DBLP dataset, runs the AnyBURL pipeline to learn rules,
then times each prediction operation and prints a summary table.
"""

import random
import time
import warnings

from torch_geometric.datasets import DBLP

from anyburl import AnyBURL, AnyBURLConfig, RulePredictor, SamplingStrategy
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
    pipeline = AnyBURL(config).fit(data)
    fit_elapsed = time.perf_counter() - t0
    print(f"Pipeline fit: {fit_elapsed:.2f}s")
    print(f"  Rules passing filter: {len(pipeline.results)}")
    print()

    if not pipeline.results:
        print("No rules found, cannot benchmark prediction.")
        return

    graph = HeteroGraph(data)

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

    # Chain product stats
    print("Chain product stats:")
    cyclic_groups = predictor._cyclic_groups
    ac1_groups = predictor._ac1_groups
    print(f"  Cyclic groups: {len(cyclic_groups)}")
    print(f"  AC1 groups:    {len(ac1_groups)}")
    for i, g in enumerate(cyclic_groups):
        nnz = g.chain_product.product.col_indices().numel()
        shape = tuple(g.chain_product.product.shape)
        print(
            f"    cyclic[{i}]: shape={shape} nnz={nnz} conf={g.aggregated_confidence:.4f}"
        )
    for i, g in enumerate(ac1_groups):
        nnz = g.chain_product.product.col_indices().numel()
        shape = tuple(g.chain_product.product.shape)
        n_subj = len(g.subject_grounded)
        n_obj = len(g.object_grounded)
        print(
            f"    ac1[{i}]: shape={shape} nnz={nnz} subj_grounded={n_subj} obj_grounded={n_obj}"
        )
    print()


if __name__ == "__main__":
    main()
