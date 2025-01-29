"""Module to wrap experiments, automatically handle cross validation and collect results."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import numpy as np

from modelname.dataset import ServiceNetworkDataset
from modelname.model import HubLocationModel

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs."""

    def run(self) -> None:
        data = ServiceNetworkDataset()

        nodes = data.nyc_neighborhoods
        N = np.arange(len(nodes))
        distances = data.get_distances()  # from node i to node j
        demands = data.get_demands()  # between node i and node j

        model = HubLocationModel(len(nodes), 5)
        model.solve(distances, demands)
        solution_hubs, solution_arcs = model.get_solution()

        hub_names = list(data.nyc_neighborhoods[solution_hubs.astype(np.bool)])
        print(hub_names)
        print(np.all(solution_arcs == 0))

        data.visualize(hub_names)
        data.visualize_solution(solution_arcs)


if __name__ == "__main__":
    exp = Experiment()
    exp.run()
