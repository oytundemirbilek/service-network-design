"""Module for your custom dataset and how it should be treated."""

from __future__ import annotations

import math
import os

import geopandas
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import networkx as nx

FILE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "nyc-taxi-trip-duration-extended")
MAPDATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "neighborhoods-in-new-york")


class ServiceNetworkDataset:
    """"""

    def __init__(self, filename: str = "train_extended.csv") -> None:
        self.df: pd.DataFrame = pd.read_csv(os.path.join(DATA_PATH, filename))
        # self.nyc_info: pd.DataFrame = pd.read_csv(
        #     os.path.join(DATA_PATH, "nyc_additional_info.csv")
        # )
        self.nyc_map: pd.DataFrame = geopandas.read_file(
            os.path.join(MAPDATA_PATH, "ZillowNeighborhoods-NY.shp")
        )
        self.nyc_map = self.nyc_map.drop_duplicates("Name", keep="first")
        self.select_counties = [
            "Kings",
            "Queens",
            "Bronx",
            "Nassau",
            "Richmond",
            "Suffolk",
            "New York",
            # "Westchester",
            "Rockland",
            "Putnam",
        ]
        self.demand = None
        self.distances = None
        self.nodes = None
        self.edges = None
        self.locations = None
        self.nyc_neighborhoods = self.get_unique_neighborhoods()

        self.df = self.clean_data(self.df)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """"""
        rename_mapping = {
            "Battery Park City": "Battery Park",
            "Bay Terrace, Staten Island": "Bay Terrace",
            "Bedford-Stuyvesant": "Bedford Stuyvesant",
            "Bull's Head": "Bulls Head",
            "Mariners Harbor": "Mariner's Harbor",
            "Prospect-Lefferts Gardens": "Prospect Lefferts Gardens",
            "Richmondtown": "Richmond Town",
            "Hell's Kitchen": "Clinton",
            "Sea Gate": "Coney Island",
            "South Slope": "Greenwood",
            "Nolita": "Little Italy",
            "Long Island City": "Hunters Point",
            "Kips Bay": "Gramercy",
            "Howland Hook": "Port Ivory",
            "Edgemere": "Far Rockaway",
            "Bayswater": "Far Rockaway",
            "Cypress Hills": "East New York",
        }

        for k, v in rename_mapping.items():
            df.loc[df["pickup_neighbourhood"] == k, "pickup_neighbourhood"] = v
            df.loc[df["dropoff_neighbourhood"] == k, "dropoff_neighbourhood"] = v

        df = self.filter_neighborhoods(df, "pickup_neighbourhood")
        df = self.filter_neighborhoods(df, "dropoff_neighbourhood")

        def filter_out_same_neighborhood(row):
            return row["pickup_neighbourhood"] != row["dropoff_neighbourhood"]

        df_filter = df.apply(filter_out_same_neighborhood, axis=1)
        df = df[df_filter]

        return df

    def calculate_demand(self) -> NDArray[np.floating]:
        """"""
        demand = self.df.groupby(
            ["pickup_neighbourhood", "dropoff_neighbourhood"], as_index=False
        )["passenger_count"].sum()
        demand = pd.DataFrame(demand)

        size = len(self.nyc_neighborhoods)
        demand_mat = np.zeros((size, size))
        # TODO: maybe there is a more efficient way to derive this matrix.
        for i, src in enumerate(self.nyc_neighborhoods):
            for j, dest in enumerate(self.nyc_neighborhoods):
                pairs = demand[
                    (demand["pickup_neighbourhood"] == src)
                    & (demand["dropoff_neighbourhood"] == dest)
                ]
                if not pairs.empty:
                    demand_mat[i, j] = pairs["passenger_count"].item()
        np.savetxt("demand_matrix.txt", demand_mat)
        return demand_mat

    def get_demands(self) -> NDArray[np.floating]:
        """"""
        if self.demand is None:
            self.demand = self.calculate_demand()

        return self.demand

    def get_unique_neighborhoods(self) -> NDArray[np.object_]:
        """"""
        # info_uniq = set(self.nyc_info["neighbourhood"].unique())
        map_uniq = set(self.nyc_map["Name"].unique())
        pickup_uniq = set(self.df["pickup_neighbourhood"].unique())
        dropoff_uniq = set(self.df["dropoff_neighbourhood"].unique())
        union_uniq = dropoff_uniq.union(pickup_uniq)
        all_uniq = list(union_uniq.intersection(map_uniq))
        all_uniq.sort()
        return np.array(all_uniq)

    def filter_neighborhoods(self, df: pd.DataFrame, colname: str) -> pd.DataFrame:
        """"""

        def filters(row):
            return row in self.nyc_neighborhoods

        df_filter = df[colname].apply(filters)
        return df[df_filter]

    def calculate_locations(self) -> NDArray[np.floating]:
        """"""
        self.nyc_map["center"] = self.nyc_map["geometry"].centroid
        self.nyc_map["center_longitude"] = self.nyc_map["center"].x
        self.nyc_map["center_latitude"] = self.nyc_map["center"].y

        def filter_counties(row):
            return row in self.select_counties

        nyc_map_filter = self.nyc_map["County"].apply(filter_counties)
        self.nyc_map = self.nyc_map[nyc_map_filter]
        self.nyc_map = self.filter_neighborhoods(self.nyc_map, "Name")

        return np.array(
            [
                self.nyc_map["center_longitude"].to_numpy(),
                self.nyc_map["center_latitude"].to_numpy(),
            ]
        )

    def get_locations(self) -> NDArray[np.floating]:
        """"""
        if self.locations is None:
            self.locations = self.calculate_locations()
        return self.locations

    def calculate_haversine_distance(self, src: str, dest: str) -> float:
        """"""
        orig_long = self.nyc_map.loc[
            self.nyc_map["Name"] == src, "center_longitude"
        ].item()
        orig_lat = self.nyc_map.loc[
            self.nyc_map["Name"] == src, "center_latitude"
        ].item()

        dest_long = self.nyc_map.loc[
            self.nyc_map["Name"] == dest, "center_longitude"
        ].item()
        dest_lat = self.nyc_map.loc[
            self.nyc_map["Name"] == dest, "center_latitude"
        ].item()

        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(orig_lat)
        lon1_rad = math.radians(orig_long)
        lat2_rad = math.radians(dest_lat)
        lon2_rad = math.radians(dest_long)

        # Differences in coordinates
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        # Haversine formula
        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distance in kilometers
        distance = R * c

        return distance

    def calculate_distances(self) -> NDArray[np.floating]:
        """"""
        locations = self.get_locations()
        size = len(self.nyc_neighborhoods)
        dist_mat = np.zeros((size, size))
        for i, src in enumerate(self.nyc_neighborhoods):
            for j, dest in enumerate(self.nyc_neighborhoods):
                if dest != src:
                    dist = self.calculate_haversine_distance(src, dest)
                    # dist = self.calculate_distance(src, dest)
                    dist_mat[i, j] = dist
        np.savetxt("distance_matrix_haversine.txt", dist_mat)
        return dist_mat

    def get_distances(self) -> NDArray[np.floating]:
        """"""
        if self.distances is None:
            self.distances = self.calculate_distances()
        return self.distances

    def create_nodes(self) -> dict:
        """"""
        self.get_locations()
        # nodes = [(f"P{i+1}", row["Name"]) for i, row in self.nyc_map.iterrows()]
        nodes = {
            (f"V{i+1}", str(neigh)): np.array(
                (
                    self.nyc_map.loc[
                        self.nyc_map["Name"] == neigh, "center_longitude"
                    ].item(),
                    self.nyc_map.loc[
                        self.nyc_map["Name"] == neigh, "center_latitude"
                    ].item(),
                )
            )
            for i, neigh in enumerate(self.nyc_neighborhoods)
        }
        return nodes

    def get_nodes(self) -> dict:
        """"""
        if self.nodes is None:
            self.nodes = self.create_nodes()
        return self.nodes

    def create_edges(self, adj: NDArray[np.floating], verbose: bool = True) -> list:
        """"""
        edges = []
        for i, neigh_src in enumerate(self.nyc_neighborhoods):
            for j, neigh_dest in enumerate(self.nyc_neighborhoods):
                if adj[i, j] != 0:
                    # print(f"Total demand via {i} to {j}: {adj[i, j]}")
                    edges.append(
                        (
                            (f"V{i+1}", neigh_src),
                            (f"V{j+1}", neigh_dest),
                            {"demand": round(adj[i, j], 3)},
                        )
                    )
        return edges

    def get_edges(self, adj: NDArray[np.floating]) -> list:
        """"""
        if self.edges is None:
            self.edges = self.create_edges(adj)
        return self.edges

    def create_graph(self, adj: NDArray[np.floating]) -> nx.Graph:
        """"""
        graph = nx.DiGraph()

        graph.add_nodes_from(self.get_nodes())
        graph.add_edges_from(self.get_edges(adj))

        return graph

    def visualize_solution(self, solution: NDArray[np.floating]) -> None:
        """"""
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        graph = self.create_graph(solution)

        self.nyc_map.plot(ax=ax, color="white", edgecolor="grey")
        # # Draw and display plot
        nx.draw_networkx(
            graph,
            pos=self.nodes,
            # node_color=nx.get_node_attributes(graph, "color").values(),
            ax=ax,
            hide_ticks=False,
            node_size=50,
            font_size=6,
        )
        plt.show()

    def visualize(self, hubs: list[str] | None = None):
        """"""
        if hubs is None:
            hubs = []
        locations = self.get_locations()
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        self.nyc_map.plot(ax=ax, color="white", edgecolor="grey")

        # for location in locations.T:
        #     ax.plot(location[0], location[1], "go")
        for i, row in self.nyc_map.iterrows():
            mark = "r*" if row["Name"] in hubs else "go"
            ax.plot(row["center"].x, row["center"].y, mark)
            ax.annotate(row["Name"], (row["center"].x, row["center"].y))

        plt.show()


if __name__ == "__main__":
    data = ServiceNetworkDataset()
    hubs = [
        "Castleton Corners",
        "New Brighton",
        "Oakwood",
        "Prospect Heights",
        "Red Hook",
    ]
    data.visualize(hubs)
