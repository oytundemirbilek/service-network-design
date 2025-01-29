"""Module to wrap experiments, automatically handle cross validation and collect results."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from modelname.dataset import ServiceNetworkDataset
from modelname.model import HubLocationModel, ServiceNetworkModel

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs."""
        self.n_hubs = 5

    def run(self) -> None:
        """Run an experiment to select the hub locations and service frequencies."""
        data = ServiceNetworkDataset()

        distances = data.get_distances()
        demands = data.get_demands()

        n_nodes = len(data.nyc_neighborhoods)

        model = HubLocationModel(n_nodes, self.n_hubs)
        model.solve(distances, demands)
        solution_hubs, solution_arcs = model.get_solution()

        if solution_arcs is None or solution_hubs is None:
            return

        hub_names = data.nyc_neighborhoods[solution_hubs.astype(bool)].tolist()

        print(hub_names)
        data.visualize_hubs(hub_names)
        data.visualize_solution(solution_arcs)

        fixed_costs = np.ones(n_nodes) * 10.0
        capacities = np.ones(n_nodes) * 10
        service_model = ServiceNetworkModel(n_nodes, 3, 1.5, 0.5, n_hubs=self.n_hubs)
        service_model.solve(
            distances, demands, solution_hubs, solution_arcs, fixed_costs, capacities
        )
        solution_flights, solution_vertiports, solution_u = service_model.get_solution()

        if (
            solution_flights is None
            or solution_vertiports is None
            or solution_u is None
        ):
            return

        removed_port_names = data.nyc_neighborhoods[
            ~solution_vertiports.astype(bool)
        ].tolist()
        print(removed_port_names)
        data.visualize_hubs(removed_port_names)
        data.visualize_solution(solution_flights)


if __name__ == "__main__":
    exp = Experiment()
    exp.run()
