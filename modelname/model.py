"""Module to define neural network, which is our solution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gurobipy as gp
import numpy as np
from typing_extensions import TypeAlias  # noqa: UP035

if TYPE_CHECKING:
    gpvar: TypeAlias = gp.tupledict[Any, gp.Var]


class OptimizationModel:
    """Base class for common functions for all optimization models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def build_solution_1d(x: gpvar, length: int) -> np.ndarray:
        """Convert and return a 1-D gurobi solution variable into a numpy array."""
        solution = []
        for i in range(length):
            solution.append(x[i].X)
        return np.array(solution)

    @staticmethod
    def build_solution_2d(x: gpvar, length: int) -> np.ndarray:
        """Convert and return a 2-D gurobi solution variable into a numpy array."""
        solution = []
        for i in range(length):
            solution_1d = []
            for j in range(length):
                solution_1d.append(x[i, j].X)
            solution.append(solution_1d)
        return np.array(solution)


class HubLocationModel(OptimizationModel):
    """Model that solves a service coverage problem by finding the most optimal hubs."""

    def __init__(self, n_nodes: int, max_hubs: int = 10) -> None:
        super().__init__()
        self.model = gp.Model("Hub selection")
        self.n_nodes = n_nodes
        self.max_hubs = max_hubs
        self.optimal_hubs: np.ndarray | None = None
        self.optimal_arcs: np.ndarray | None = None

    def solve(
        self, distances: np.ndarray, demands: np.ndarray
    ) -> tuple[np.ndarray | None, ...]:
        """
        Build and solve the mathematical optimization of hub locations.

        Parameters
        ----------
        distances:
            an adjacency matrix of pairwise distances between vertiports,
            shape: (n_nodes, n_nodes).
        demands:
            an adjacency matrix of demands from each source and to each destination node,
            shape: (n_nodes, n_nodes).

        Returns
        -------
        hubs: np.ndarray
            shape: (n_nodes)
        arcs: np.ndarray
            shape: (n_nodes, n_nodes)
        """
        ns = np.arange(self.n_nodes)

        x: gpvar = self.model.addVars(*ns.shape, vtype=gp.GRB.BINARY, name="x")
        y: gpvar = self.model.addVars(*distances.shape, vtype=gp.GRB.BINARY, name="y")

        # Objective: Minimize total distance
        self.model.setObjective(
            gp.quicksum(
                y[i, j] / distances[i, j] for i in ns for j in ns if demands[i, j] > 0
            ),
            gp.GRB.MAXIMIZE,
        )
        # + gp.quicksum(
        #     x[i] * demands[i, j] for i in N for j in N
        #     ),

        # Constraint 1: Select maximum n_hubs
        self.model.addConstr(x.sum() == self.max_hubs, "Select_hubs")

        # Constraint 2: Each location j must be assigned to one or no hub
        self.model.addConstrs((y.sum("*", j) <= 1 for j in ns), "Assign_each_location")

        # Constraint 3: A location j can only be assigned to a hub i if i is selected as a hub
        self.model.addConstrs(
            (y[i, j] <= x[i] for i in ns for j in ns), "Assign_to_hub"
        )

        self.model.optimize()

        self.optimal_hubs = self.build_solution_1d(x, self.n_nodes)
        self.optimal_arcs = self.build_solution_2d(y, self.n_nodes)
        return self.optimal_hubs, self.optimal_arcs

    def get_solution(self) -> tuple[np.ndarray | None, ...]:
        """Return lastly solved model's solution."""
        return self.optimal_hubs, self.optimal_arcs


class ServiceNetworkModel(OptimizationModel):
    """Model that solves a service coverage problem by finding the most optimal hubs."""

    def __init__(
        self,
        n_nodes: int,
        base_price: float,
        price_per_km: float,
        cost_per_km: float,
        n_hubs: int = 5,
        max_seats: int = 4,
        budget: float | None = None,
    ) -> None:
        super().__init__()
        self.model = gp.Model("Service Network")
        self.n_nodes = n_nodes
        self.n_hubs = n_hubs
        self.price_per_km = price_per_km
        self.base_price = base_price
        self.cost_per_km = cost_per_km
        self.max_seats = max_seats
        self.budget = budget
        self.optimal_n_flights: np.ndarray | None = None  # f
        self.optimal_vertiports: np.ndarray | None = None  # y
        self.optimal_unsatisfied_demand: np.ndarray | None = None  # u

    @staticmethod
    def redirect_flights(
        demands: np.ndarray, hub_indices: np.ndarray, hub_zones: np.ndarray
    ) -> np.ndarray:
        """Transform the demand matrix in a way to redirect some flights to hubs."""
        # NOTE: maybe dataset class is a better place for this.
        return demands

    def solve(
        self,
        distances: np.ndarray,
        demands: np.ndarray,
        # NOTE: solution could take effect outside this function
        # e.g., demands and distances could be assumed as already adjusted.
        hub_indices: np.ndarray,  # solution of hub selection
        hub_zones: np.ndarray,  # solution of hub selection
        fixed_costs: np.ndarray,
        capacities: np.ndarray,
    ) -> tuple[np.ndarray | None, ...]:
        """
        Build and solve the mathematical optimization of service frequencies.

        Parameters
        ----------
        distances:
            an adjacency matrix of pairwise distances between vertiports,
            shape: (n_nodes, n_nodes).
        demands:
            an adjacency matrix of demands from each source and to each destination node,
            shape: (n_nodes, n_nodes).
        hub_indices:
            a binary array indicating which nodes are hubs or not (1 or 0),
            selected hubs from hub selection problem. shape: (n_nodes)
        hub_zones:
            an adjacency matrix of arcs, indicates which nodes are connected,
            selected arcs from hub selection problem. shape: (n_nodes, n_nodes).
        fixed_costs:
            a float array listing fixed cost for each vertiport. shape: (n_nodes)
        capacities:
            a float array listing capacity for each vertiport. shape: (n_nodes)
        """
        # zero-out all arcs that are not supposed to be connected.
        distances *= hub_zones
        # Create extra flights that need to be transferred from a hub.
        demands = self.redirect_flights(demands, hub_indices, hub_zones)

        # Indices
        ns = np.arange(self.n_nodes)

        # Parameters
        p = self.price_per_km
        b = self.base_price
        c = self.cost_per_km
        s = self.max_seats

        # Decision variables
        f: gpvar = self.model.addVars(*distances.shape, vtype=gp.GRB.INTEGER, name="f")
        y: gpvar = self.model.addVars(*ns.shape, vtype=gp.GRB.BINARY, name="y")
        u: gpvar = self.model.addVars(*distances.shape, vtype=gp.GRB.INTEGER, name="u")

        # Objective: maximize profit
        profits = gp.quicksum(
            f[i, j] * (b + (p - c) * distances[i, j]) for i in ns for j in ns
        )
        total_fixed_costs = gp.quicksum(fixed_costs[i] * y[i] for i in ns)
        self.model.setObjective(profits - total_fixed_costs, gp.GRB.MAXIMIZE)

        # Hubs should be open NOTE: maybe same solution when we remove this?
        self.model.addConstrs(y[h] == 1 for h in hub_indices)

        # Demand satisfaction
        self.model.addConstrs(
            f[i, j] * s >= demands[i, j] - u[i, j] for i in ns for j in ns
        )

        # Flow conversation
        self.model.addConstrs(f[i, j] == f[j, i] for i in ns for j in ns)

        # Capacity
        self.model.addConstrs(
            gp.quicksum(f[i, j] for j in ns) <= capacities[i] * y[i] for i in ns
        )

        # Budget
        if self.budget is not None:
            self.model.addConstr(total_fixed_costs <= self.budget)

        self.model.optimize()

        self.optimal_n_flights = self.build_solution_2d(f, self.n_nodes)
        self.optimal_vertiports = self.build_solution_1d(y, self.n_nodes)
        self.optimal_unsatisfied_demand = self.build_solution_2d(u, self.n_nodes)
        return (
            self.optimal_n_flights,
            self.optimal_vertiports,
            self.optimal_unsatisfied_demand,
        )

    def get_solution(
        self,
    ) -> tuple[np.ndarray | None, ...]:
        """Return lastly solved model's solution."""
        return (
            self.optimal_n_flights,
            self.optimal_vertiports,
            self.optimal_unsatisfied_demand,
        )
