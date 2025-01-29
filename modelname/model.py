"""Module to define neural network, which is our solution."""

from __future__ import annotations

from typing import Any

import gurobipy as gp
import numpy as np

gpvar = gp.tupledict[Any, gp.Var]


class HubLocationModel(gp.Model):
    """Model that solves a service coverage problem by finding the most optimal hubs."""

    def __init__(self, n_nodes: int, max_hubs: int = 10) -> None:
        """"""
        super().__init__("Hub network")
        self.n_nodes = n_nodes
        self.max_hubs = max_hubs
        self.optimal_hubs = None
        self.optimal_arcs = None

    def solve(self, distances: np.ndarray, demands: np.ndarray) -> None:
        """Build and solve the mathematical optimization of hub locations."""
        N = np.arange(self.n_nodes)

        x: gpvar = self.addVars(*N.shape, vtype=gp.GRB.BINARY, name="x")
        y: gpvar = self.addVars(*distances.shape, vtype=gp.GRB.BINARY, name="y")

        # Objective: Minimize total distance
        self.setObjective(
            gp.quicksum(
                y[i, j] * demands[i, j] / distances[i, j]
                for i in N
                for j in N
                if demands[i, j] > 0
            ),
            gp.GRB.MAXIMIZE,
        )

        # Constraint 1: Select maximum n_hubs
        self.addConstr(x.sum() <= self.max_hubs, "Select_hubs")

        # Constraint 2: Each location j must be assigned to one or no hub
        self.addConstrs((y.sum("*", j) <= 1 for j in N), "Assign_each_location")

        # Constraint 3: A location j can only be assigned to a hub i if i is selected as a hub
        self.addConstrs((y[i, j] <= x[i] for i in N for j in N), "Assign_to_hub")

        self.optimize()

        self.build_solution(x, y)

    def build_solution(self, x: gpvar, y: gpvar) -> tuple[np.ndarray, np.ndarray]:
        """Convert and return gurobi solution variables into numpy arrays."""
        solution_hubs = []
        for i in range(self.n_nodes):
            solution_hubs.append(x[i].X)
        solution_hubs = np.array(solution_hubs)

        solution_arcs = []
        for i in range(self.n_nodes):
            solution_arc = []
            for j in range(self.n_nodes):
                solution_arc.append(y[i, j].X)
            solution_arcs.append(solution_arc)
        solution_arcs = np.array(solution_arcs)

        self.optimal_hubs = solution_hubs
        self.optimal_arcs = solution_arcs

        return solution_hubs, solution_arcs

    def get_solution(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return lastly solved model's solution."""
        return self.optimal_hubs, self.optimal_arcs


class ServiceNetworkModel(gp.Model):
    """Model that solves a service coverage problem by finding the most optimal hubs."""

    def __init__(self, n_nodes: int, max_hubs: int = 10) -> None:
        """"""
        super().__init__("Hub network")
        self.n_nodes = n_nodes
        self.max_hubs = max_hubs
        self.optimal_hubs = None
        self.optimal_arcs = None
