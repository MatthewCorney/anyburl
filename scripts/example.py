"""End-to-end AnyBURL pipeline on the DBLP dataset.

Wires together all four engine components -- HeteroGraph, TripleSampler,
WalkEngine, RuleGeneralizer/RuleEvaluator -- on a real PyG dataset to verify
the pipeline works: sample triples, walk, generalize into rules, evaluate,
and print the best rules.

DBLP has 4 node types (Author, Paper, Term, Conference) and 3 distinct edge
semantics (author-paper, paper-term, conference-paper), making it ideal for
demonstrating heterogeneous rule learning.

This is a standalone demo script -- not production code. It intentionally
skips strict typing and enum usage for brevity.
"""
import time
import warnings

from torch_geometric.datasets import DBLP

from anyburl.graph import HeteroGraph
from anyburl.metrics import RuleEvaluator
from anyburl.rule import Rule, RuleConfig, RuleGeneralizer
from anyburl.sampler import SamplerConfig, SamplingStrategy, TripleSampler
from anyburl.walk import WalkConfig, WalkEngine, WalkStrategy

# Suppress beta warnings from sparse CSR tensors
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 500
MAX_WALK_LENGTH = 3
MIN_WALK_LENGTH = 2
MAX_WALK_ATTEMPTS = 50
TOP_K_RULES = 20
SEED = 42

# Filter out low-quality rules
MIN_SUPPORT = 2
MIN_CONFIDENCE = 0.01
MIN_HEAD_COVERAGE = 0.01


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load DBLP dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("AnyBURL End-to-End Pipeline - DBLP Dataset")
    print("=" * 60)
    print()

    t0 = time.perf_counter()
    dataset = DBLP(root="./data/DBLP")
    data = dataset[0]
    elapsed = time.perf_counter() - t0
    print(f"[1/6] Loaded DBLP dataset ({elapsed:.2f}s)")

    # ------------------------------------------------------------------
    # 2. Build HeteroGraph
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    graph = HeteroGraph(data)
    elapsed = time.perf_counter() - t0
    print(f"[2/6] Built HeteroGraph ({elapsed:.2f}s)")

    print()
    print("  Node types:")
    for nt in graph.node_types:
        print(f"    {nt}: {graph.node_count(nt)} nodes")
    print("  Edge types:")
    for et in graph.edge_types:
        print(f"    {et}: {graph.edge_count(et)} edges")
    print(f"  Total edges: {graph.total_edge_count()}")
    print()

    # ------------------------------------------------------------------
    # 3. Sample triples
    # ------------------------------------------------------------------
    sampler_config = SamplerConfig(
        sample_size=SAMPLE_SIZE,
        strategy=SamplingStrategy.UNIFORM,
        seed=SEED,
    )
    sampler = TripleSampler(graph, sampler_config)

    t0 = time.perf_counter()
    triples = sampler.sample()
    elapsed = time.perf_counter() - t0
    print(f"[3/6] Sampled {len(triples)} triples ({elapsed:.2f}s)")

    # ------------------------------------------------------------------
    # 4. Walk from each triple
    # ------------------------------------------------------------------
    walk_config = WalkConfig(
        max_length=MAX_WALK_LENGTH,
        min_length=MIN_WALK_LENGTH,
        max_attempts=MAX_WALK_ATTEMPTS,
        strategy=WalkStrategy.UNIFORM,
        seed=SEED,
    )
    walker = WalkEngine(graph, walk_config)

    t0 = time.perf_counter()
    all_paths = []
    triples_with_paths = 0
    for triple in triples:
        paths = walker.walk_from_triple(triple)
        if paths:
            triples_with_paths += 1
        for path in paths:
            all_paths.append((path, triple))
    elapsed = time.perf_counter() - t0
    print(
        f"[4/6] Walked {len(all_paths)} paths from "
        f"{triples_with_paths}/{len(triples)} triples ({elapsed:.2f}s)"
    )

    # ------------------------------------------------------------------
    # 5. Generalize paths into rules
    # ------------------------------------------------------------------
    rule_config = RuleConfig(
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        min_head_coverage=MIN_HEAD_COVERAGE,
    )
    generalizer = RuleGeneralizer(rule_config)

    t0 = time.perf_counter()
    seen_rules: set[str] = set()
    unique_rules: list[Rule] = []
    generalization_errors = 0

    for path, triple in all_paths:
        try:
            rules = generalizer.generalize(
                path,
                target_relation=triple.relation,
                head_type=triple.head_type,
                tail_type=triple.tail_type,
            )
        except ValueError:
            generalization_errors += 1
            continue

        for rule in rules:
            rule_str = str(rule)
            if rule_str not in seen_rules:
                seen_rules.add(rule_str)
                unique_rules.append(rule)

    elapsed = time.perf_counter() - t0
    print(
        f"[5/6] Generalized into {len(unique_rules)} unique rules "
        f"({generalization_errors} errors) ({elapsed:.2f}s)"
    )

    # ------------------------------------------------------------------
    # 6. Evaluate rules
    # ------------------------------------------------------------------
    metrics_config = RuleConfig(
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        min_head_coverage=MIN_HEAD_COVERAGE,
    )
    evaluator = RuleEvaluator(graph, metrics_config)

    t0 = time.perf_counter()
    results = evaluator.evaluate_batch(unique_rules)
    elapsed = time.perf_counter() - t0
    print(f"[6/6] Evaluated rules: {len(results)} pass thresholds ({elapsed:.2f}s)")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Triples sampled:     {len(triples)}")
    print(f"  Paths found:         {len(all_paths)}")
    print(f"  Unique rules:        {len(unique_rules)}")
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


if __name__ == "__main__":
    main()
