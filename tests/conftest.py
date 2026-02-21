"""Shared fixtures for AnyBURL tests."""

import pytest
import torch
from torch_geometric.data import HeteroData

from anyburl.graph import HeteroGraph
from anyburl.rule import Atom, Rule, RuleType, Term


@pytest.fixture
def simple_hetero_data() -> HeteroData:
    """Create a small synthetic heterogeneous graph.

    Graph structure (5 persons, 3 cities):

    Person -[lives_in]-> City:
        0 -> 0, 0 -> 1, 1 -> 1, 2 -> 2, 3 -> 0, 4 -> 2

    Person -[knows]-> Person:
        0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 0

    City -[near]-> City:
        0 -> 1, 1 -> 2, 2 -> 0
    """
    data = HeteroData()
    data["person"].num_nodes = 5
    data["city"].num_nodes = 3

    data["person", "lives_in", "city"].edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4], [0, 1, 1, 2, 0, 2]], dtype=torch.long
    )
    data["person", "knows", "person"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
    )
    data["city", "near", "city"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]], dtype=torch.long
    )
    return data


@pytest.fixture
def simple_graph(simple_hetero_data: HeteroData) -> HeteroGraph:
    """Create a HeteroGraph from the simple synthetic data."""
    return HeteroGraph(simple_hetero_data)


@pytest.fixture
def evaluator_graph() -> HeteroGraph:
    """Create a graph for evaluator and prediction tests.

    Graph structure (4 persons, 3 cities):

    Person -[lives_in]-> City:
        0 -> 0, 1 -> 1, 2 -> 2, 3 -> 0

    Person -[born_in]-> City:
        0 -> 0, 1 -> 1, 2 -> 0, 3 -> 2

    City -[near]-> City:
        0 -> 1, 1 -> 2, 2 -> 0
    """
    data = HeteroData()
    data["person"].num_nodes = 4
    data["city"].num_nodes = 3

    data["person", "lives_in", "city"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]], dtype=torch.long
    )
    data["person", "born_in", "city"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 0, 2]], dtype=torch.long
    )
    data["city", "near", "city"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]], dtype=torch.long
    )
    return HeteroGraph(data)


@pytest.fixture
def perfect_eval_graph() -> HeteroGraph:
    """Create a graph where born_in @ near perfectly predicts lives_in.

    Graph structure (3 persons, 3 cities):

    Person -[lives_in]-> City:
        0 -> 1, 1 -> 2, 2 -> 0

    Person -[born_in]-> City:
        0 -> 0, 1 -> 1, 2 -> 2

    City -[near]-> City:
        0 -> 1, 1 -> 2, 2 -> 0

    born_in @ near exactly reproduces lives_in:
        person 0: born_in(0,0), near(0,1) -> lives_in(0,1)
        person 1: born_in(1,1), near(1,2) -> lives_in(1,2)
        person 2: born_in(2,2), near(2,0) -> lives_in(2,0)
    """
    data = HeteroData()
    data["person"].num_nodes = 3
    data["city"].num_nodes = 3

    data["person", "lives_in", "city"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]], dtype=torch.long
    )
    data["person", "born_in", "city"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data["city", "near", "city"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]], dtype=torch.long
    )
    return HeteroGraph(data)


@pytest.fixture
def cyclic_rule() -> Rule:
    """Create: lives_in(X, Y) :- born_in(X, Z0), near(Z0, Y)."""
    head = Atom(
        relation="lives_in",
        subject=Term.variable("X", node_type="person"),
        object_=Term.variable("Y", node_type="city"),
    )
    body = (
        Atom(
            relation="born_in",
            subject=Term.variable("X", node_type="person"),
            object_=Term.variable("Z0", node_type="city"),
        ),
        Atom(
            relation="near",
            subject=Term.variable("Z0", node_type="city"),
            object_=Term.variable("Y", node_type="city"),
        ),
    )
    return Rule(head=head, body=body, rule_type=RuleType.CYCLIC)
