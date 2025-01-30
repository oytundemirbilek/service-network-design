import numpy as np

class ModelParameters:
    def __init__(self):
        self.base_price = 45 #dollar per seat
        self.price_per_km = 3 #dollar per km per seat
        self.fixed_cost_vertiport = 56 #dollar daily
        self.fixed_cost_hub = 264 #dollar daily
        self.variable_cost_per_km = 0.95 #dollar per km

    def get_total_price(self, distance) -> np.ndarray:
        distances = np.asarray(distance)

        #if there's no traveling, no base price
        total_price = np.where(distances == 0, 0, self.base_price + self.price_per_km * distances)
        total_price = np.round(total_price, 2)

        return total_price

    @property
    def get_fixed_cost_vertiport(self):
        return self.fixed_cost_vertiport

    @property
    def get_fixed_cost_hub(self):
        return self.fixed_cost_hub

    def get_variable_cost_per_km(self, distance) -> np.ndarray:
        distances = np.asarray(distance)

        # if there's no traveling, no variable cost
        variable_cost = np.where(distances == 0, 0, self.variable_cost_per_km * distances)
        variable_cost = np.round(variable_cost, 2)

        return variable_cost

