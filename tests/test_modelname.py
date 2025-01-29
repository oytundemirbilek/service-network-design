"""Test graph dataset classes."""

from __future__ import annotations

import os
import numpy as np

from modelname.dataset import ServiceNetworkDataset
from modelname.model import HubLocationModel

FILE_PATH = os.path.dirname(__file__)
MAPDATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "neighborhoods-in-new-york")
DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "nyc-taxi-trip-duration-extended")
GOLD_STANDARD_PATH = os.path.join(FILE_PATH, "expected")
TEST_INPUT_PATH = os.path.join(FILE_PATH, "inputs")
MODELS_PATH = os.path.join(FILE_PATH, "..", "models")


def test_hub_location_model() -> None:
    """Test if the model can be iterated - cpu based."""
    np.random.seed(0)
    distances = np.random.randn(100, 100)
    demands = np.random.randn(100, 100)
    model = HubLocationModel(100, 5)
    model.solve(distances, demands)
    solution_hubs, solution_arcs = model.get_solution()
    assert solution_hubs is not None
    assert solution_arcs is not None


def test_service_network_data() -> None:
    """"""
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

    dataset.visualize(["Dyker Heights", "Kensington", "Harlem"], False)
