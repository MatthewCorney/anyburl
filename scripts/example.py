"""End-to-end AnyBURL pipeline on the BIOKG dataset.

Uses `AnyBURLPipeline` to learn rules from a real heterogeneous knowledge
graph and print the best results.

BIOKG has 4 node types (drug, function, phenotype, protein) and 83 edge
types spanning drug-protein interactions, GO ontology relations, phenotype
associations, and protein-protein interactions -- ideal for testing rule
learning over diverse relation types.

This is a standalone demo script -- not production code. It intentionally
skips strict typing and enum usage for brevity.
"""
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from anyburl import AnyBURLConfig, AnyBURL, SamplingStrategy

# Suppress beta warnings from sparse CSR tensors
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")

PICKLE_PATH = Path(__file__).resolve().parent / "data" / "BIOKG" / "KG_100.pickle"

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
TOP_K_RULES = 20
TARGET_EDGE_TYPE = ("protein", "is_annotated_to", "phenotype")


def _load_biokg_pickle() -> HeteroData:
    """Load BIOKG pickle and convert to PyG HeteroData."""
    with PICKLE_PATH.open("rb") as f:
        nx_graph = pickle.load(f)  # noqa: S301

    # Map each node to a contiguous integer within its type
    type_to_ids: dict[str, dict[str, int]] = defaultdict(dict)
    for node, attrs in nx_graph.nodes(data=True):
        node_type = attrs["tipo"]
        mapping = type_to_ids[node_type]
        if node not in mapping:
            mapping[node] = len(mapping)

    # Group edges by (src_type, rel, dst_type), deduplicating
    edge_sets: dict[tuple[str, str, str], set[tuple[int, int]]] = defaultdict(set)
    for u, v, attrs in nx_graph.edges(data=True):
        src_type = nx_graph.nodes[u]["tipo"]
        dst_type = nx_graph.nodes[v]["tipo"]
        rel = attrs["rel_type"]
        src_id = type_to_ids[src_type][u]
        dst_id = type_to_ids[dst_type][v]
        edge_sets[(src_type, rel, dst_type)].add((src_id, dst_id))

    data = HeteroData()
    for node_type, mapping in type_to_ids.items():
        data[node_type].num_nodes = len(mapping)
    for edge_type, pairs in edge_sets.items():
        src_ids, dst_ids = zip(*pairs)
        data[edge_type].edge_index = torch.tensor(
            [src_ids, dst_ids], dtype=torch.long
        )

    return data


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load BIOKG dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("AnyBURL End-to-End Pipeline - BIOKG Dataset")
    print("=" * 60)
    print()

    t0 = time.perf_counter()
    data = _load_biokg_pickle()
    elapsed = time.perf_counter() - t0
    print(f"[1/2] Loaded BIOKG pickle ({elapsed:.2f}s)")

    # ------------------------------------------------------------------
    # 2. Run pipeline
    # ------------------------------------------------------------------
    config = AnyBURLConfig(
        sample_size=4000,
        sampling_strategy=SamplingStrategy.UNIFORM,
        target_edge_type=TARGET_EDGE_TYPE,
        max_walk_length=4,
        min_walk_length=2,
        max_walk_attempts=1000,
        min_support=3,
        min_confidence=0.001,
        min_head_coverage=0.005,
        seed=42,
    )

    t0 = time.perf_counter()
    pipeline = AnyBURL(config).fit(data)
    elapsed = time.perf_counter() - t0
    print(f"[2/2] Pipeline completed ({elapsed:.2f}s)")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    results = pipeline.results
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Triples sampled:     {len(pipeline.triples)}")
    print(f"  Paths found:         {len(pipeline.paths)}")
    print(f"  Unique rules:        {len(pipeline.rules)}")
    print(f"  Rules passing filter:{len(results)}")
    print()

    # Sort by confidence descending, then support descending
    results.sort(key=lambda x: (x[1].confidence, x[1].support), reverse=True)

    print(f"Top {min(TOP_K_RULES, len(results))} Rules (by confidence):")
    print("-" * 100)
    print(
        f"{'#':<4} {'Type':<8} {'Support':<10} {'Conf':<10} "
        f"{'HeadCov':<10} {'Preds':<10} {'Rule'}"
    )
    print("-" * 100)

    for i, (rule, metrics) in enumerate(results[:TOP_K_RULES]):
        print(
            f"{i + 1:<4} {rule.rule_type.value:<8} {metrics.support:<10} "
            f"{metrics.confidence:<10.4f} {metrics.head_coverage:<10.4f} "
            f"{metrics.num_predictions:<10} {rule}"
        )

    print("-" * 100)

    # ------------------------------------------------------------------
    # 3. Predictions
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Predictions (filter_known=True)")
    print("=" * 60)

    t0 = time.perf_counter()
    predictions = pipeline.predict(filter_known=True)
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(predictions)} predictions ({elapsed:.2f}s)")
    print()

    TOP_K_PREDICTIONS = 20
    n_show = min(TOP_K_PREDICTIONS, len(predictions))
    print(f"Top {n_show} Predictions (by aggregated confidence):")
    print("-" * 100)
    print(
        f"{'#':<4} {'Head':<20} {'Tail':<20} "
        f"{'AggConf':<10} {'MaxConf':<10} {'MeanConf':<10} {'#Rules':<8}"
    )
    print("-" * 100)

    for i, pred in enumerate(predictions[:TOP_K_PREDICTIONS]):
        head_label = f"{pred.head_type}:{pred.head_id}"
        tail_label = f"{pred.tail_type}:{pred.tail_id}"
        print(
            f"{i + 1:<4} {head_label:<20} {tail_label:<20} "
            f"{pred.aggregated_confidence:<10.4f} {pred.max_confidence:<10.4f} "
            f"{pred.mean_confidence:<10.4f} {pred.num_rules:<8}"
        )

    print("-" * 100)


if __name__ == "__main__":
    main()
