"""Build mapped-canal graph position proxies for 250 m Bangkok cells.

OSM node connectivity and proximity to BMA equipment are map evidence only.
They do not establish hydraulic flow, an outfall, gate state, or pump capacity.

Run: .venv/bin/python flood_bangkok_canal_graph_features.py
"""

from __future__ import annotations

import json
import math

import networkx as nx
import numpy as np
import osmium
from scipy.spatial import cKDTree

from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import (
    OSM_PBF, OUTPUT, TO_LONLAT, TO_UTM, cell_box, grid_spec, pump_points,
)


CODES = ("1006", "1040")
FEATURES = ("mapped_canal_component_length_log", "path_to_mapped_gate_log",
            "path_to_mapped_pump_log", "cell_to_canal_node_m",
            "mapped_gate_reachable", "mapped_pump_reachable")
STATION_MAX_OFFSET_M = 150
CELL_MAX_OFFSET_M = 500


class CanalGraph(osmium.SimpleHandler):
    def __init__(self, bounds):
        super().__init__()
        self.graph = nx.Graph()
        self.coords = {}
        self.bounds = bounds
        self.lon0, self.lat0 = TO_LONLAT.transform(bounds[0], bounds[1])
        self.lon1, self.lat1 = TO_LONLAT.transform(bounds[2], bounds[3])
        self.ways = 0

    def way(self, way):
        if way.tags.get("waterway") != "canal":
            return
        nodes = list(way.nodes)
        if len(nodes) < 2 or any(not node.location.valid() for node in nodes):
            return
        lon = [node.lon for node in nodes]
        lat = [node.lat for node in nodes]
        if max(lon) < self.lon0 or min(lon) > self.lon1 or max(lat) < self.lat0 or min(lat) > self.lat1:
            return
        xs, ys = TO_UTM.transform(lon, lat)
        self.ways += 1
        for i, node in enumerate(nodes):
            self.coords[node.ref] = (xs[i], ys[i])
            if i == 0:
                continue
            first, second = nodes[i-1].ref, node.ref
            length = math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
            if length <= 0:
                continue
            old = self.graph.get_edge_data(first, second)
            if old is None or length < old["weight"]:
                self.graph.add_edge(first, second, weight=length)


def main() -> None:
    districts = {d["code"]: d for d in load_districts() if d["code"] in CODES}
    bounds = (min(d["polygon"].bounds[0] for d in districts.values()) - 5000,
              min(d["polygon"].bounds[1] for d in districts.values()) - 5000,
              max(d["polygon"].bounds[2] for d in districts.values()) + 5000,
              max(d["polygon"].bounds[3] for d in districts.values()) + 5000)
    handler = CanalGraph(bounds)
    handler.apply_file(str(OSM_PBF), locations=True)
    graph = handler.graph
    if graph.number_of_nodes() < 100 or graph.number_of_edges() < 100:
        raise ValueError("Too few mapped canal nodes")
    ids = np.fromiter(graph.nodes, dtype=np.int64)
    coordinates = np.asarray([handler.coords[int(i)] for i in ids], dtype=np.float64)
    tree = cKDTree(coordinates)
    stations = pump_points()
    sources = {"gate": set(), "pump": set()}
    matched = {"gate": 0, "pump": 0}
    total = {"gate": 0, "pump": 0}
    for station in stations:
        kinds = []
        if "ประตู" in station["type"] or "ประตู" in station["name"]:
            kinds.append("gate")
        if "สูบ" in station["type"] or "สูบ" in station["name"]:
            kinds.append("pump")
        distance, index = tree.query((station["x"], station["y"]))
        for kind in kinds:
            total[kind] += 1
            if distance <= STATION_MAX_OFFSET_M:
                matched[kind] += 1
                sources[kind].add(int(ids[index]))
    paths = {kind: nx.multi_source_dijkstra_path_length(graph, nodes,
              weight="weight") if nodes else {} for kind, nodes in sources.items()}
    component_length = {}
    components = list(nx.connected_components(graph))
    for component in components:
        length = sum(edge[2]["weight"] for edge in graph.subgraph(component).edges(data=True))
        for node in component:
            component_length[node] = length
    audit = {"osm_source": OSM_PBF.name, "snapshot_date": "2025-01-01",
             "waterway_tag": "canal", "ways": handler.ways,
             "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
             "components": len(components),
             "station_match_max_offset_m": STATION_MAX_OFFSET_M,
             "cell_match_max_offset_m": CELL_MAX_OFFSET_M,
             "station_inventory": total, "stations_matched_to_canal_node": matched,
             "feature_names": list(FEATURES),
             "connectivity_limit": "OSM shared nodes plus nearest-node attachments, not surveyed drainage topology or verified outfall; no flow direction, pipe, gate operation, or pump operation",
             "districts": {}}
    for code, district in districts.items():
        grid = grid_spec(district["polygon"])
        with np.load(OUTPUT / f"surface_shared_state_{code}_daily_cells.npz") as saved:
            rr, cc = saved["rows"], saved["cols"]
        values = np.full((len(rr), len(FEATURES)), np.nan, dtype=np.float32)
        for i, (row, col) in enumerate(zip(rr, cc)):
            centre = cell_box(grid, int(row), int(col)).centroid
            offset, index = tree.query((centre.x, centre.y))
            values[i, 3] = offset
            if offset > CELL_MAX_OFFSET_M:
                values[i, 4:] = 0
                continue
            node = int(ids[index])
            values[i, 0] = math.log1p(component_length[node])
            for j, kind in ((1, "gate"), (2, "pump")):
                distance = paths[kind].get(node)
                if distance is not None:
                    values[i, j] = math.log1p(distance + offset)
                    values[i, j + 3] = 1
                else:
                    values[i, j + 3] = 0
        np.savez_compressed(CACHE / f"canal_graph_{code}_250m.npz",
                            values=values, names=np.asarray(FEATURES),
                            rows=rr, cols=cc)
        audit["districts"][code] = {
            "cells": len(rr),
            "cells_attached_within_500m": int(np.count_nonzero(np.isfinite(values[:, 0]))),
            "cells_reaching_mapped_gate": int(np.count_nonzero(values[:, 4] == 1)),
            "cells_reaching_mapped_pump": int(np.count_nonzero(values[:, 5] == 1)),
            "median_offset_m": round(float(np.median(values[:, 3])), 1)}
        print(code, audit["districts"][code], flush=True)
    (CACHE / "canal_graph_250m_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
