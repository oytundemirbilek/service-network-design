"""Module to wrap experiments, automatically handle cross validation and collect results."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from modelname.dataset import ServiceNetworkDataset
from modelname.model import HubLocationModel, ServiceNetworkModel
from modelname.parameters import ModelParameters

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs."""
        self.n_hubs = 5

    @staticmethod
    def run_mock_sample() -> None:
        """Run an experiment to select the service frequencies on a mock example."""
        data = ServiceNetworkDataset()
        data.nyc_neighborhoods = np.array(
            [
                "Belle Harbor",
                "Chinatown",
                "Clifton",
                "Clinton Hill",
                "East Village",
                "Gravesend",
                "Shore Acres",
                "Westerleigh",
                "Woodrow",
            ]
        )
        n_nodes = len(data.nyc_neighborhoods)
        distances = data.calculate_distances()
        demands = data.calculate_demand()

        params = ModelParameters()

        # hub_ind = solution_hubs.astype(bool)
        # vertiport_ind = np.invert(hub_ind)

        fixed_costs = np.ones(n_nodes) * 6 * 30 * params.fixed_cost_hub
        # fixed_costs[hub_ind] *= params.fixed_cost_hub
        # fixed_costs[vertiport_ind] *= params.fixed_cost_vertiport

        capacities = np.ones(n_nodes) * 6 * 30 * params.cap_hub
        # capacities[hub_ind] *= params.cap_hub
        # capacities[vertiport_ind] *= params.cap_vertiport

        service_model = ServiceNetworkModel(
            n_nodes,
            params.base_price,
            params.price_per_km,
            params.variable_cost_per_km,
        )
        service_model.solve(distances, demands, fixed_costs, capacities)
        solution_flights, _solution_vertiports = service_model.get_solution()
        if solution_flights is None:
            return
        # data.visualize_solution(demands)
        #data.visualize_solution(solution_flights)
        # data.visualize_solution(solution_u)

    def run_hub_location(self) -> None:
        """Run an experiment to select the hub locations."""
        data = ServiceNetworkDataset()

        distances = data.get_distances()
        demands = data.get_demands() / 180  # avg per day

        n_nodes = len(data.nyc_neighborhoods)

        model = HubLocationModel(n_nodes, self.n_hubs)
        model.solve(distances, demands)
        solution_hubs, solution_arcs = model.get_solution()

        if solution_arcs is None or solution_hubs is None:
            return

        hub_ind = solution_hubs.astype(bool)
        # vertiport_ind = np.invert(hub_ind)

        hub_names = data.nyc_neighborhoods[hub_ind].tolist()

        #print(hub_names)
        # data.visualize_hubs(hub_names)
        #data.visualize_solution(solution_arcs)

    @staticmethod
    def run_wo_hubs() -> None:
        """Run an experiment to select the service frequencies."""
        data = ServiceNetworkDataset()

        distances = data.get_distances()
        demands = data.get_demands() // 180

        n_nodes = len(data.nyc_neighborhoods)

        params = ModelParameters()
        fixed_costs = np.ones(n_nodes) * params.fixed_cost_vertiport
        capacities = np.ones(n_nodes) * params.cap_vertiport

        service_model = ServiceNetworkModel(
            n_nodes,
            params.base_price,
            params.price_per_km,
            params.variable_cost_per_km,
        )
        service_model.solve(distances, demands, fixed_costs, capacities)
        solution_flights, solution_vertiports = service_model.get_solution()


        if solution_flights is None or solution_vertiports is None:
            return

        removed_port_names = data.nyc_neighborhoods[
            ~solution_vertiports.astype(bool)
        ].tolist()
        print(removed_port_names)
        print(solution_flights)
        print(solution_vertiports)
        # data.visualize_hubs(removed_port_names)

        flights_per_vertiport = solution_flights.sum(axis=1)
        top_5_indices = np.argsort(flights_per_vertiport)[-5:]  # get highest 5

        print("Top 5 vertiport indices:", top_5_indices)
        print("Flights count for those top 5:", flights_per_vertiport[top_5_indices])

        # (E) Create a filtered flights matrix with only the top 5 => zero out all others
        filtered_flights = np.zeros_like(solution_flights)
        for i in top_5_indices:
            for j in top_5_indices:
                filtered_flights[i, j] = solution_flights[i, j]

        service_level = solution_flights * 4 / demands
        service_level[np.where((solution_flights == 0) | (demands == 0))] = 0
        data.visualize_solution(filtered_flights, xlim=(-74.05, -73.9), ylim=(40.68, 40.83))
        data.plot_service_level_hist(service_level)

    # def run_with_hubs(self) -> None:
    #     """Run an experiment to select the hub locations and service frequencies."""
    #     data = ServiceNetworkDataset()

    #     distances = data.get_distances()
    #     demands = data.get_demands() / 180  # avg per day

    #     n_nodes = len(data.nyc_neighborhoods)

    #     model = HubLocationModel(n_nodes, self.n_hubs)
    #     model.solve(distances, demands)
    #     solution_hubs, solution_arcs = model.get_solution()

    #     if solution_arcs is None or solution_hubs is None:
    #         return

    #     hub_ind = solution_hubs.astype(bool)
    #     vertiport_ind = np.invert(hub_ind)

    #     hub_names = data.nyc_neighborhoods[hub_ind].tolist()

    #     print(hub_names)
    #     # data.visualize_hubs(hub_names)
    #     data.visualize_solution(solution_arcs)

    #     params = ModelParameters()

    #     fixed_costs = np.ones(n_nodes)
    #     fixed_costs[hub_ind] *= params.fixed_cost_hub
    #     fixed_costs[vertiport_ind] *= params.fixed_cost_vertiport

    #     capacities = np.ones(n_nodes)
    #     capacities[hub_ind] *= params.cap_hub
    #     capacities[vertiport_ind] *= params.cap_vertiport

    #     # redirected_demands = ServiceNetworkModelWithHubs.redirect_flights(
    #     #     demands, solution_hubs, solution_arcs
    #     # )
    #     # data.visualize_solution(redirected_demands)

    #     service_model = ServiceNetworkModelWithHubs(
    #         n_nodes,
    #         params.base_price,
    #         params.price_per_km,
    #         params.variable_cost_per_km,
    #         n_hubs=self.n_hubs,
    #     )
    #     service_model.solve(
    #         distances, demands, solution_hubs, solution_arcs, fixed_costs, capacities
    #     )
    #     solution_flights, solution_vertiports, solution_u = service_model.get_solution()

    #     if (
    #         solution_flights is None
    #         or solution_vertiports is None
    #         or solution_u is None
    #     ):
    #         return

    #     removed_port_names = data.nyc_neighborhoods[
    #         ~solution_vertiports.astype(bool)
    #     ].tolist()
    #     print(removed_port_names)
    #     # data.visualize_hubs(removed_port_names)
    #     data.visualize_solution(solution_flights)


if __name__ == "__main__":
    exp = Experiment()
    exp.run_wo_hubs()
