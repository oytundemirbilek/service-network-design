"""Test graph dataset classes."""

from __future__ import annotations

import os

import numpy as np

from modelname.dataset import ServiceNetworkDataset
from modelname.model import HubLocationModel, ServiceNetworkModel

FILE_PATH = os.path.dirname(__file__)
MAPDATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "neighborhoods-in-new-york")
DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "nyc-taxi-trip-duration-extended")
GOLD_STANDARD_PATH = os.path.join(FILE_PATH, "expected")
TEST_INPUT_PATH = os.path.join(FILE_PATH, "inputs")
MODELS_PATH = os.path.join(FILE_PATH, "..", "models")


def test_hub_location_model() -> None:
    """Test if the hub location model could be built and solved."""
    np.random.seed(0)
    n_nodes = 10
    distances = np.random.randn(n_nodes, n_nodes)
    demands = np.random.randn(n_nodes, n_nodes)
    model = HubLocationModel(n_nodes, 5)
    model.solve(distances, demands)
    solution_hubs, solution_arcs = model.get_solution()
    assert solution_hubs is not None
    assert solution_hubs.shape == (n_nodes,), solution_hubs.shape
    assert solution_arcs is not None
    assert solution_arcs.shape == (n_nodes, n_nodes), solution_arcs.shape


def test_service_network_model() -> None:
    """Test if the service network model could be built and solved."""
    np.random.seed(0)
    n_nodes = 10
    distances = np.random.randn(n_nodes, n_nodes)
    demands = np.random.randn(n_nodes, n_nodes)
    model = ServiceNetworkModel(n_nodes, 2.0, 1.5, 0.5)
    hubs = np.concatenate([np.ones(n_nodes // 2), np.zeros(n_nodes // 2)])
    hub_zones = np.ones_like(distances)
    fixed_costs = np.ones(n_nodes) * 10.0
    capacities = np.ones(n_nodes) * 5
    model.solve(distances, demands, hubs, hub_zones, fixed_costs, capacities)
    solution_nflights, solution_trips, solution_u = model.get_solution()
    assert solution_nflights is not None
    assert solution_trips is not None
    assert solution_u is not None


def test_service_network_data() -> None:
    """Test if the data operations can be done correctly."""
    n_nodes = 59
    n_edges = 254
    dataset = ServiceNetworkDataset("mock_data.csv", TEST_INPUT_PATH)
    assert len(dataset.nyc_neighborhoods) == n_nodes, len(dataset.nyc_neighborhoods)

    demands = dataset.get_demands()
    distances = dataset.get_distances()

    assert demands.shape == (n_nodes, n_nodes), demands.shape
    assert distances.shape == (n_nodes, n_nodes), distances.shape

    nodes = dataset.get_nodes()
    assert len(nodes) == n_nodes, len(nodes)

    edges = dataset.get_edges(demands)
    assert len(edges) == n_edges, len(edges)

    dataset.visualize_hubs(["Dyker Heights", "Kensington", "Harlem"], False)
