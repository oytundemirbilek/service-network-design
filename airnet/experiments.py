"""Module to wrap experiments, automatically handle cross validation and collect results."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from airnet.dataset import ServiceNetworkDataset
from airnet.model import HubLocationModel, ServiceNetworkModel
from airnet.parameters import ModelParameters

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs."""
        self.n_hubs = 5
        # Solution variables:
        self.service_levels: np.ndarray | None = None
        self.flights: np.ndarray | None = None
        self.vertiports: np.ndarray | None = None
        self.service_model: ServiceNetworkModel | None = None

        self.data = ServiceNetworkDataset(download=kwargs.get("download_data", True))

    def run_hub_location(self) -> None:
        """Run an experiment to select the hub locations."""
        distances = self.data.get_distances()
        demands = self.data.get_demands() / 180  # avg per day

        n_nodes = len(self.data.nyc_neighborhoods)

        model = HubLocationModel(n_nodes, self.n_hubs)
        model.solve(distances, demands)
        solution_hubs, solution_arcs = model.get_solution()

        if solution_arcs is None or solution_hubs is None:
            return

        hub_ind = solution_hubs.astype(bool)
        # vertiport_ind = np.invert(hub_ind)

        hub_names = self.data.nyc_neighborhoods[hub_ind].tolist()

        print(hub_names)
        self.data.visualize_solution(solution_arcs)

    def run_wo_hubs(
        self, parameters: ModelParameters | None = None, plot_solution: bool = True
    ) -> None:
        """Run an experiment to select the service frequencies."""
        self.distances = self.data.get_distances()
        self.demands = self.data.get_demands() // 180

        n_nodes = len(self.data.nyc_neighborhoods)

        params = parameters
        if params is None:
            params = ModelParameters()
        fixed_costs = np.ones(n_nodes) * params.fixed_cost_vertiport
        capacities = np.ones(n_nodes) * params.cap_vertiport

        self.service_model = ServiceNetworkModel(
            n_nodes,
            params.base_price,
            params.price_per_km,
            params.variable_cost_per_km,
        )
        self.service_model.solve(self.distances, self.demands, fixed_costs, capacities)
        solution_flights, solution_vertiports = self.service_model.get_solution()

        if solution_flights is None or solution_vertiports is None:
            return

        removed_port_names = self.data.nyc_neighborhoods[
            ~solution_vertiports.astype(bool)
        ].tolist()
        print(removed_port_names)
        # data.visualize_hubs(removed_port_names)
        self.service_levels = solution_flights * 4 / self.demands
        self.service_levels[np.where((solution_flights == 0) | (self.demands == 0))] = 0
        self.vertiports = solution_vertiports
        self.flights = solution_flights
        if plot_solution:
            self.data.visualize_solution(self.service_levels, show_edges=True)

    def get_solution_variables(self) -> tuple[np.ndarray | None, ...]:
        """Return optimal variables after the model run."""
        return self.flights, self.vertiports, self.service_levels


class SensitivityAnalysis(Experiment):
    """Class to conduct experiments to inspect the effect of different parameters."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.costs = np.arange(1.0, 2.0, 0.1)  # price should be 3.0
        self.prices = np.arange(1.0, 3.0, 0.2)  # cost should be 0.95
        self.capacities = np.arange(100, 350, 25)

    def run_sensitivity_experiment(self, params: ModelParameters) -> tuple[float, ...]:
        """Run a sensitivity experiment with given parameters and return service level and profit."""
        if self.service_model is not None:
            del self.service_model
            self.service_model = None
        self.run_wo_hubs(params, plot_solution=False)
        self.flights, self.vertiports, self.service_levels = (
            self.get_solution_variables()
        )
        service_level = 0.0
        profit = 0.0

        if (
            self.flights is not None
            and self.vertiports is not None
            # and self.service_levels is not None
        ):
            service_level = self.flights.sum() * 4.0 / self.demands.sum()

        if (
            self.service_model is not None
            and self.service_model.optimal_profit is not None
        ):
            profit = self.service_model.optimal_profit
        return (service_level, profit)

    def run_cost_sensitivity(self) -> tuple[np.ndarray, ...]:
        """Run the model several times for different per-km-costs."""
        service_levels = []
        profits = []
        for cost in self.costs:
            params = ModelParameters(variable_cost_per_km=cost)
            service_level, profit = self.run_sensitivity_experiment(params)
            service_levels.append(service_level)
            profits.append(profit)

        return np.array(service_levels), np.array(profits)

    def run_price_sensitivity(self) -> tuple[np.ndarray, ...]:
        """Run the model several times for different per-km-prices."""
        service_levels = []
        profits = []
        for price in self.prices:
            params = ModelParameters(price_per_km=price)
            service_level, profit = self.run_sensitivity_experiment(params)
            service_levels.append(service_level)
            profits.append(profit)

        return np.array(service_levels), np.array(profits)

    def run_capacity_sensitivity(self) -> tuple[np.ndarray, ...]:
        """Run the model several times for different vertiport capacities."""
        service_levels = []
        profits = []
        for cap in self.capacities:
            params = ModelParameters(cap_vertiport=cap)
            service_level, profit = self.run_sensitivity_experiment(params)
            service_levels.append(service_level)
            profits.append(profit)

        return np.array(service_levels), np.array(profits)

    def run_and_save(self) -> None:
        """Run all analysis and save results in a dataframe."""
        price_vs_service, price_vs_profits = self.run_price_sensitivity()
        cost_vs_service, cost_vs_profits = self.run_cost_sensitivity()
        cap_vs_service, cap_vs_profits = self.run_capacity_sensitivity()

        results_df = pd.DataFrame(
            {
                "Prices": self.prices,
                "Price-profit sensitivity": price_vs_profits,
                "Price-service sensitivity": price_vs_service,
                "Costs": self.costs,
                "Cost-profit sensitivity": cost_vs_profits,
                "Cost-service sensitivity": cost_vs_service,
                "Capacities": self.capacities,
                "Capacity-profit sensitivity": cap_vs_profits,
                "Capacity-service sensitivity": cap_vs_service,
            }
        )
        results_df.to_csv("sensitivity.csv", index=False)


if __name__ == "__main__":
    exp = SensitivityAnalysis(download_data=True)
    exp.run_and_save()
