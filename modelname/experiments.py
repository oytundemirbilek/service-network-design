"""Module to wrap experiments, automatically handle cross validation and collect results."""

from __future__ import annotations

import os
from typing import Any

from modelname.dataset import ServiceNetworkDataset
from modelname.model import HubLocationModel

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs."""

    @staticmethod
    def run() -> None:
        """Run an experiment to select the hub locations and service frequencies."""
        data = ServiceNetworkDataset()

        distances = data.get_distances()
        demands = data.get_demands()

        model = HubLocationModel(len(data.nyc_neighborhoods), 5)
        model.solve(distances, demands)
        solution_hubs, solution_arcs = model.get_solution()

        if solution_arcs is None or solution_hubs is None:
            return

        hub_names = data.nyc_neighborhoods[solution_hubs.astype(bool)].tolist()

        print(hub_names)
        data.visualize_hubs(hub_names)
        data.visualize_solution(solution_arcs)


if __name__ == "__main__":
    exp = Experiment()
    exp.run()
