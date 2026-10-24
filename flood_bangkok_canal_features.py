"""Build 250 m canal geometry channels for the eleven-district CNN pilot.

Uses the local OpenStreetMap Thailand 2025-01-01 snapshot. Features are mapped
canal centreline length inside a cell and distance from its centre to the
nearest mapped canal. These describe geometry, not water level, canal capacity,
flow direction, or whether the canal was open during a historical storm.

Run: .venv/bin/python flood_bangkok_canal_features.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import osmium
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

from flood_bangkok_spatial_cnn_crossdistrict import DISTRICTS
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import GRID_M, TO_LONLAT, TO_UTM, cell_box, grid_spec


ROOT = Path(__file__).resolve().parent
OSM_PBF = ROOT / "data" / "thailand-250101.osm.pbf"
AUDIT = CACHE / "canal_2025_audit.json"
BUFFER_M = 5000


class CanalWays(osmium.SimpleHandler):
    def __init__(self, study_bounds: tuple[float, float, float, float]):
        super().__init__()
        self.clip = box(*study_bounds)
        self.lines = []
        self.classes = Counter()
        self.invalid_ways = 0
        x0, y0, x1, y1 = study_bounds
        self.lon0, self.lat0 = TO_LONLAT.transform(x0, y0)
        self.lon1, self.lat1 = TO_LONLAT.transform(x1, y1)

    def way(self, way) -> None:
        kind = way.tags.get("waterway")
        if not kind:
            return
        nodes = list(way.nodes)
        if len(nodes) < 2 or any(not node.location.valid() for node in nodes):
            self.invalid_ways += 1
            return
        lon = [node.lon for node in nodes]
        lat = [node.lat for node in nodes]
        if (max(lon) < self.lon0 or min(lon) > self.lon1 or
                max(lat) < self.lat0 or min(lat) > self.lat1):
            return
        xs, ys = TO_UTM.transform(lon, lat)
        line = LineString(zip(xs, ys))
        if line.is_empty or not line.intersects(self.clip):
            return
        self.classes[kind] += 1
        if kind == "canal":
            clipped = line.intersection(self.clip)
            if not clipped.is_empty and clipped.length > 0:
                self.lines.append(clipped)


def main() -> None:
    codes = DISTRICTS + MORE_CODES
    districts = {d["code"]: d for d in load_districts() if d["code"] in codes}
    x0 = min(d["polygon"].bounds[0] for d in districts.values()) - BUFFER_M
    y0 = min(d["polygon"].bounds[1] for d in districts.values()) - BUFFER_M
    x1 = max(d["polygon"].bounds[2] for d in districts.values()) + BUFFER_M
    y1 = max(d["polygon"].bounds[3] for d in districts.values()) + BUFFER_M
    if not OSM_PBF.exists():
        raise FileNotFoundError(OSM_PBF)
    handler = CanalWays((x0, y0, x1, y1))
    handler.apply_file(str(OSM_PBF), locations=True)
    if len(handler.lines) < 10:
        raise ValueError("Too few mapped canals in the study area")
    print("mapped canal ways", len(handler.lines), "classes", handler.classes, flush=True)
    tree = STRtree(handler.lines)
    audit = {"source": OSM_PBF.name, "snapshot_date": "2025-01-01",
             "tag_rule": "waterway=canal only", "study_buffer_m": BUFFER_M,
             "waterway_ways_in_study_bbox_by_tag": dict(handler.classes),
             "invalid_way_count": handler.invalid_ways,
             "mapped_canal_ways": len(handler.lines),
             "mapped_canal_length_km": round(sum(line.length for line in handler.lines) / 1000, 2),
             "districts": {}}
    for code in codes:
        polygon = districts[code]["polygon"]
        spec = grid_spec(polygon)
        shape = spec["rows"], spec["cols"]
        length = np.zeros(shape, dtype=np.float32)
        distance = np.full(shape, 10000, dtype=np.float32)
        valid = np.zeros(shape, dtype=np.uint8)
        for row in range(spec["rows"]):
            for col in range(spec["cols"]):
                cell = cell_box(spec, row, col)
                if not cell.intersects(polygon) or cell.intersection(polygon).area == 0:
                    continue
                valid[row, col] = 1
                centre = Point(cell.centroid)
                nearest = tree.nearest(centre)
                if nearest is not None:
                    distance[row, col] = min(10000, float(centre.distance(
                        handler.lines[int(nearest)])))
                for index in tree.query(cell):
                    length[row, col] += float(handler.lines[int(index)].intersection(cell).length)
        path = CACHE / f"canal_2025_{code}_250m.npz"
        np.savez_compressed(path, canal_length_m=length,
                            nearest_canal_m=distance, valid=valid)
        site_distances = distance[valid.astype(bool)]
        audit["districts"][code] = {
            "name": districts[code]["name_en"],
            "valid_cells": int(valid.sum()),
            "cells_with_canal_centreline": int(np.count_nonzero(length)),
            "canal_length_inside_grid_km": round(float(length.sum()) / 1000, 2),
            "median_cell_centre_distance_m": round(float(np.median(site_distances)), 1),
            "p90_cell_centre_distance_m": round(float(np.percentile(site_distances, 90)), 1)}
        print(code, audit["districts"][code], flush=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
