"""Build candidate surface-flow and mapped-canal graphs on the citywide grid.

The surface links are inferred from 250 m terrain (steepest lower neighbour).
Canal links follow successive grid cells crossed by the same 2025 OSM way.
Neither is a verified pipe or cell-to-outfall drainage connection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

from flood_bangkok_canal_features import CanalWays, OSM_PBF
from flood_bangkok_spatial_cnn_citywide import FEATURE_CACHE
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import GRID_M, grid_spec


GRAPH_DIR = CACHE / "flow_graph_250m_v1"
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))
MIN_DROP_M = 0.05


def lines_in(geometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from lines_in(part)


def canal_edges(district: dict, lookup: np.ndarray, rr: np.ndarray,
                cc: np.ndarray, tree: STRtree, lines: list) -> np.ndarray:
    spec = grid_spec(district["polygon"])
    edges = set()
    # The 125 m buffer allows the same mapped way to cross a boundary cell.
    area = district["polygon"].buffer(GRID_M / 2)
    for index in tree.query(area):
        clipped = lines[int(index)].intersection(area)
        for line in lines_in(clipped):
            count = max(1, math.ceil(line.length / 50))
            previous = -1
            for i in range(count + 1):
                point = line.interpolate(i / count, normalized=True)
                col = math.floor((point.x - spec["west"]) / GRID_M)
                row = math.floor((spec["north"] - point.y) / GRID_M)
                node = (int(lookup[row, col]) if 0 <= row < lookup.shape[0]
                        and 0 <= col < lookup.shape[1] else -1)
                if node >= 0 and previous >= 0 and node != previous:
                    if abs(int(rr[node]) - int(rr[previous])) <= 1 and abs(
                            int(cc[node]) - int(cc[previous])) <= 1:
                        edges.add((previous, node))
                        edges.add((node, previous))
                previous = node
    return (np.asarray(sorted(edges), dtype=np.int32).reshape(-1, 2)
            if edges else np.empty((0, 2), dtype=np.int32))


def surface_edges(elevation: np.ndarray, rr: np.ndarray, cc: np.ndarray,
                  lookup: np.ndarray) -> np.ndarray:
    edges = []
    for node, (row, col) in enumerate(zip(rr, cc)):
        z = float(elevation[row, col])
        best_node, best_grade = -1, 0.0
        for dr, dc in OFFSETS:
            nr, nc = int(row) + dr, int(col) + dc
            if not (0 <= nr < lookup.shape[0] and 0 <= nc < lookup.shape[1]):
                continue
            next_node = int(lookup[nr, nc])
            if next_node < 0:
                continue
            drop = z - float(elevation[nr, nc])
            grade = drop / math.hypot(dr, dc)
            if drop >= MIN_DROP_M and grade > best_grade:
                best_node, best_grade = next_node, grade
        if best_node >= 0:
            edges.append((node, best_node))
    return np.asarray(edges, dtype=np.int32).reshape(-1, 2)


def build_one(district: dict, tree: STRtree, lines: list) -> dict:
    code = district["code"]
    meta = json.loads((FEATURE_CACHE / f"{code}_meta.json").read_text())
    rr = np.asarray(meta["valid_rows"], dtype=np.int32)
    cc = np.asarray(meta["valid_cols"], dtype=np.int32)
    h, w = meta["shape"][2:]
    lookup = np.full((h, w), -1, dtype=np.int32)
    lookup[rr, cc] = np.arange(len(rr))
    x = np.load(FEATURE_CACHE / f"{code}_x.npy", mmap_mode="r")
    elevation = np.asarray(x[0, 0], dtype=np.float32)
    surface = surface_edges(elevation, rr, cc, lookup)
    canal = canal_edges(district, lookup, rr, cc, tree, lines)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GRAPH_DIR / f"{code}.npz", rows=rr, cols=cc,
                        surface=surface, canal=canal, shape=np.asarray((h, w)))
    return {"nodes": len(rr), "surface_directed_edges": len(surface),
            "canal_directed_edges": len(canal),
            "surface_linked_nodes": len(np.unique(surface.ravel())) if len(surface) else 0,
            "canal_linked_nodes": len(np.unique(canal.ravel())) if len(canal) else 0}


def main() -> None:
    districts = load_districts()
    bounds = [d["polygon"].bounds for d in districts]
    bbox = (min(v[0] for v in bounds) - 5000,
            min(v[1] for v in bounds) - 5000,
            max(v[2] for v in bounds) + 5000,
            max(v[3] for v in bounds) + 5000)
    handler = CanalWays(bbox)
    handler.apply_file(str(OSM_PBF), locations=True)
    tree = STRtree(handler.lines)
    audit = {"terrain_rule": "one edge to steepest lower 8-neighbour, minimum 0.05 m drop",
             "canal_rule": "bidirectional between successive valid cells crossed by one OSM canal way",
             "osm_snapshot": "2025-01-01", "grid_metres": GRID_M,
             "limitations": ["Terrain links are inferred from 250 m mean elevation",
                             "Canal direction and hydraulic connection are unknown",
                             "District graphs omit cross-boundary links",
                             "No pipes, gates, pump state or tide-controlled direction"],
             "districts": {}}
    for district in districts:
        result = build_one(district, tree, handler.lines)
        audit["districts"][district["code"]] = result
        print(district["code"], result, flush=True)
    (GRAPH_DIR / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
