"""Module for your custom dataset and how it should be treated."""

from __future__ import annotations

import math
import os
from typing import Any

import geopandas
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray

FILE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "nyc-taxi-trip-duration-extended")
MAPDATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "neighborhoods-in-new-york")


class ServiceNetworkDataset:
    """An utility class that performs various data operations."""

    def __init__(
        self, filename: str = "train_extended.csv", data_path: str | None = None
    ) -> None:
        if data_path is None:
            data_path = DATA_PATH
        self.data_path = data_path
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
        self.demand: NDArray[np.floating] | None = None
        self.distances: NDArray[np.floating] | None = None
        self.locations: NDArray[np.floating] | None = None
        self.nodes: dict[str, Any] | None = None
        self.edges: list[Any] | None = None
        self.df: pd.DataFrame = pd.read_csv(os.path.join(self.data_path, filename))
        self.nyc_neighborhoods = self.get_unique_neighborhoods()

        clean_data_path = os.path.join(
            self.data_path, filename.split(".")[0] + "_clean.csv"
        )
        if os.path.exists(clean_data_path):
            self.df = pd.read_csv(clean_data_path)
        else:
            self.df = self.clean_data(self.df)
            self.df.to_csv(clean_data_path, index=False)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the input dataframe.

        Some neighborhoods will be renamed; by removing the hyphens and apostrophes,
        standardize old/new names, very small neighborhoods will be made part of the bigger ones.
        Filter the data by detected neighborhoods. Remove inter-neighborhood trips.
        """
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

        def filter_out_same_neighborhood(row: pd.Series) -> bool:
            return row["pickup_neighbourhood"] != row["dropoff_neighbourhood"]

        df_filter = df.apply(filter_out_same_neighborhood, axis=1)
        df = df[df_filter]

        return df

    def calculate_demand(self) -> NDArray[np.floating]:
        """Calculate total demand per arc, by summing up the passenger counts per trip."""
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
        return demand_mat

    def get_demands(self) -> NDArray[np.floating]:
        """Get demand matrix either from saved file or defined attribute, otherwise calculate again."""
        if self.demand is None:
            demand_path = os.path.join(self.data_path, "demand_matrix.txt")
            if os.path.exists(demand_path):
                self.demand = np.loadtxt(demand_path)
            else:
                self.demand = self.calculate_demand()
                np.savetxt(demand_path, self.demand)

        return self.demand

    def get_unique_neighborhoods(self) -> NDArray[np.object_]:
        """Detect neighborhoods by intersection between the ones in the map and in the data."""
        # info_uniq = set(self.nyc_info["neighbourhood"].unique())
        map_uniq = set(self.nyc_map["Name"].unique())
        pickup_uniq = set(self.df["pickup_neighbourhood"].unique())
        dropoff_uniq = set(self.df["dropoff_neighbourhood"].unique())
        union_uniq = dropoff_uniq.union(pickup_uniq)
        all_uniq = list(union_uniq.intersection(map_uniq))
        all_uniq.sort()
        return np.array(all_uniq)

    def filter_neighborhoods(self, df: pd.DataFrame, colname: str) -> pd.DataFrame:
        """Filter dataframe by detected neighborhoods."""

        def filters(row: str) -> bool:
            return row in self.nyc_neighborhoods

        df_filter = df[colname].apply(filters)
        return df[df_filter]

    def calculate_locations(self) -> NDArray[np.floating]:
        """
        Process map data.

        Filter by counties and neighborhoods, also calculate center of the regions on the map.
        """
        self.nyc_map["center"] = self.nyc_map["geometry"].centroid
        self.nyc_map["center_longitude"] = self.nyc_map["center"].x
        self.nyc_map["center_latitude"] = self.nyc_map["center"].y

        def filter_counties(row: str) -> bool:
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
        """Get locations of nodes from defined attribute if available, otherwise calculate again."""
        if self.locations is None:
            self.locations = self.calculate_locations()
        return self.locations

    def calculate_haversine_distance(self, src: str, dest: str) -> float:
        """Return haversine distance between two neighborhoods (centers)."""
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
        radius = 6371.0

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
        distance = radius * c

        return distance

    def calculate_distances(self) -> NDArray[np.floating]:
        """Process map and calculate the distance matrix between each pair of neighborhoods."""
        self.get_locations()
        size = len(self.nyc_neighborhoods)
        dist_mat = np.zeros((size, size))
        for i, src in enumerate(self.nyc_neighborhoods):
            for j, dest in enumerate(self.nyc_neighborhoods):
                if dest != src:
                    dist = self.calculate_haversine_distance(src, dest)
                    # dist = self.calculate_distance(src, dest)
                    dist_mat[i, j] = dist
        return dist_mat

    def get_distances(self) -> NDArray[np.floating]:
        """Get distance matrix either from saved file or defined attribute, otherwise calculate again."""
        if self.distances is None:
            distances_path = os.path.join(
                self.data_path, "distance_matrix_haversine.txt"
            )
            if os.path.exists(distances_path):
                self.distances = np.loadtxt(distances_path)
            else:
                self.distances = self.calculate_distances()
                np.savetxt(distances_path, self.distances)
        return self.distances

    def create_nodes(self) -> dict:
        """Create graph nodes from neighborhoods to be used to create a graph."""
        self.get_locations()
        # nodes = [(f"P{i+1}", row["Name"]) for i, row in self.nyc_map.iterrows()]
        nodes = {
            (f"V{i}", str(neigh)): np.array(
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
        """Get nodes from defined attribute if available, otherwise calculate again."""
        if self.nodes is None:
            self.nodes = self.create_nodes()
        return self.nodes

    def create_edges(self, adj: NDArray[np.floating], verbose: bool = True) -> list:
        """Create graph edges from given adjacency matrix to be used to create a graph."""
        edges = []
        for i, neigh_src in enumerate(self.nyc_neighborhoods):
            for j, neigh_dest in enumerate(self.nyc_neighborhoods):
                if adj[i, j] != 0:
                    # print(f"Total demand via {i} to {j}: {adj[i, j]}")
                    edges.append(
                        (
                            (f"V{i}", neigh_src),
                            (f"V{j}", neigh_dest),
                            {"demand": round(adj[i, j], 3)},
                        )
                    )
        return edges

    def get_edges(self, adj: NDArray[np.floating]) -> list:
        """Get edges from defined attribute if available, otherwise calculate again."""
        if self.edges is None:
            self.edges = self.create_edges(adj)
        return self.create_edges(adj)

    def create_graph(self, adj: NDArray[np.floating]) -> nx.Graph:
        """Create a networkx graph from a given adjacency matrix for available neighborhoods."""
        graph = nx.DiGraph()

        graph.add_nodes_from(self.get_nodes())
        graph.add_edges_from(self.get_edges(adj))

        return graph

    def create_edge_labels(self, adj: NDArray[np.floating]) -> dict[Any, Any]:
        """Create edge labels from given adjacency matrix."""
        edges = {}
        for i, neigh_src in enumerate(self.nyc_neighborhoods):
            for j, neigh_dest in enumerate(self.nyc_neighborhoods):
                if adj[i, j] != 0:
                    # print(f"Total demand via {i} to {j}: {adj[i, j]}")
                    edges[((f"V{i}", neigh_src), (f"V{j}", neigh_dest))] = round(
                        adj[i, j], 3
                    )

        return edges

    def visualize_solution(
        self,
        solution: NDArray[np.floating],
        show_edges: bool = True,
        show: bool = True,
    ) -> None:
        """Plot the graph based on the given solution adjacency matrix."""
        _fig, ax = plt.subplots(1, 1, figsize=(9, 9))

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
        if show_edges and self.nodes is not None:
            edge_labels = self.create_edge_labels(solution)
            nx.draw_networkx_edge_labels(
                graph,
                pos=self.nodes,
                edge_labels=edge_labels,
                font_color="red",
                font_size=6,
            )

        if show:
            plt.show()

    def visualize_hubs(self, hubs: list[str] | None = None, show: bool = True) -> None:
        """Plot all neighborhood centers and mark the ones as hubs given in the list."""
        if hubs is None:
            hubs = []
        self.get_locations()
        _fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        self.nyc_map.plot(ax=ax, color="white", edgecolor="grey")

        for _, row in self.nyc_map.iterrows():
            mark = "r*" if row["Name"] in hubs else "go"
            ax.plot(row["center"].x, row["center"].y, mark)
            ax.annotate(row["Name"], (row["center"].x, row["center"].y))

        if show:
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
    data.visualize_hubs(hubs)
