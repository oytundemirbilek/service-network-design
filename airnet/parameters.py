"""Module to provide pre-defined parameters into the service network model."""

from typing import Any

import numpy as np


class ModelParameters:
    """Class to derive various parameter calculations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.base_price = kwargs.get("base_price", 45.0)  # dollar per seat
        self.price_per_km = kwargs.get("price_per_km", 3.0)  # dollar per km per seat
        self.fixed_cost_vertiport = kwargs.get(
            "fixed_cost_vertiport", 136.0
        )  # dollar daily
        self.fixed_cost_hub = kwargs.get("fixed_cost_hub", 264.0)  # dollar daily
        self.variable_cost_per_km = kwargs.get(
            "variable_cost_per_km", 0.95
        )  # dollar per km
        self.cap_vertiport = kwargs.get("cap_vertiport", 180)
        self.cap_hub = kwargs.get("cap_hub", 900)

    def get_total_price(self, distance: np.ndarray) -> np.ndarray:
        """Derive and return the price matrix from a given distance matrix."""
        distances = np.asarray(distance)

        # if there's no traveling, no base price
        total_price = np.where(
            distances == 0, 0, self.base_price + self.price_per_km * distances
        )
        total_price = np.round(total_price, 2)

        return total_price

    @property
    def get_fixed_cost_vertiport(self) -> float:
        """Return a pre-defined fixed cost for any vertiport."""
        return self.fixed_cost_vertiport

    @property
    def get_fixed_cost_hub(self) -> float:
        """Return a pre-defined fixed cost for any hub."""
        return self.fixed_cost_hub

    def get_variable_cost_per_km(self, distance: np.ndarray) -> np.ndarray:
        """Derive and return the variable cost matrix from a given distance matrix."""
        distances = np.asarray(distance)

        # if there's no traveling, no variable cost
        variable_cost = np.where(
            distances == 0, 0, self.variable_cost_per_km * distances
        )
        variable_cost = np.round(variable_cost, 2)

        return variable_cost
