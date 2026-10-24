"""Build static drainage-infrastructure features on all 50 Bangkok 250 m grids.

Sources precede the October-December 2025 test: OSM 2025-01-01 canals,
BMA pump-station CSV (published 2023), 2021 retention-pond GIS, and a
2024 underpass CSV. Proximity is not evidence of hydraulic connectivity.

Run: .venv/bin/python flood_bangkok_infrastructure_features.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import requests
import shapefile
from shapely.geometry import Point
from shapely.strtree import STRtree

from flood_bangkok_canal_features import CanalWays, OSM_PBF
from flood_bangkok_spatial_cnn_citywide import FEATURE_CACHE
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import TO_UTM, cell_box, grid_spec


SOURCE_DIR = CACHE / "infrastructure_sources"
FEATURE_DIR = CACHE / "infrastructure_static_250m_v1"
PUMP_URL = ("https://data.bangkok.go.th/dataset/82cb9fcc-ddea-436a-abfa-"
            "eb5dc9fc85fc/resource/53205669-40f9-4cb6-b902-7b8fbc945919/"
            "download/water-pump-st")
POND_URL = ("https://data.bangkok.go.th/dataset/cbb01b56-48d5-4713-bc75-"
            "a440542f3668/resource/5391c6b0-1d36-4c7c-a452-da1a428a28b7/"
            "download/pond.zip")
UNDERPASS_URL = ("https://data.bangkok.go.th/dataset/070b94c0-e119-4f6e-"
                 "9d5c-97349d55bf37/resource/53379c72-a005-4d58-828d-"
                 "d4a4393803ed/download/tunnel.csv")
FEATURES = (
    "osm_canal_length_log_2025",
    "osm_canal_distance_log_2025",
    "bma_pump_capacity_within_1km_log",
    "bma_pond_volume_within_2km_log_2021",
    "bma_pond_distance_log_2021",
    "bma_underpass_within_250m_2024",
    "bma_underpass_distance_log_2024",
)


def source_bytes(name: str, url: str) -> bytes:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / name
    if not path.exists():
        response = requests.get(url, timeout=60,
                                headers={"User-Agent": "Bangkok open-data flood research"})
        response.raise_for_status()
        if len(response.content) < 100:
            raise ValueError(f"Unexpected short source {name}")
        path.write_bytes(response.content)
    return path.read_bytes()


def pump_stations(raw: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    stations: list[tuple[str, float, float, float]] = []
    invalid = 0
    duplicates = 0
    for row in rows:
        try:
            capacity = float(row["GP_CAPACITY"])
            lon, lat = float(row["COOR_X"]), float(row["COOR_Y"])
        except (ValueError, TypeError):
            invalid += 1
            continue
        if not (0 < capacity < 1000 and 100.2 < lon < 101 and 13.4 < lat < 14.2):
            invalid += 1
            continue
        x, y = TO_UTM.transform(lon, lat)
        name = row["PUMP_NAME"].strip()
        prior = next((i for i, (old_name, ox, oy, _) in enumerate(stations)
                      if old_name == name and math.hypot(x - ox, y - oy) <= 100), None)
        if prior is not None:
            old = stations[prior]
            stations[prior] = (name, old[1], old[2], max(old[3], capacity))
            duplicates += 1
        else:
            stations.append((name, x, y, capacity))
    if len(stations) < 100:
        raise ValueError("Too few geocoded pumps with usable reported capacity")
    points = np.asarray([(x, y) for _, x, y, _ in stations], dtype=np.float64)
    capacity = np.asarray([cap for *_, cap in stations], dtype=np.float64)
    return points, capacity, {"rows": len(rows), "invalid_or_missing": invalid,
                               "near_duplicate_name_points": duplicates,
                               "usable_unique_stations": len(stations),
                               "capacity_units": "as reported; catalog unit requires confirmation"}


def retention_ponds(raw: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        reader = shapefile.Reader(
            shp=io.BytesIO(archive.read("pond.shp")),
            shx=io.BytesIO(archive.read("pond.shx")),
            dbf=io.BytesIO(archive.read("pond.dbf")), encoding="cp874")
        if reader.shapeTypeName != "POINT":
            raise ValueError("Unexpected pond geometry")
        points, volumes = [], []
        for item in reader.shapeRecords():
            x, y = item.shape.points[0]
            volume = item.record.as_dict().get("volume")
            if volume is None or not (0 < float(volume) < 1e9):
                continue
            points.append((x, y))  # Source .prj: EPSG:32647.
            volumes.append(float(volume))
    if len(points) < 20:
        raise ValueError("Too few retention-pond points")
    return np.asarray(points), np.asarray(volumes), {
        "source_features": len(reader), "usable_points_with_volume": len(points),
        "geometry": "point location, not pond footprint or drainage connection"}


def underpasses(raw: bytes) -> tuple[np.ndarray, dict]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    points = []
    missing = 0
    for row in rows:
        try:
            lon, lat = float(row["longitude"]), float(row["latitude"])
        except (ValueError, TypeError):
            missing += 1
            continue
        if not (100.2 < lon < 101 and 13.4 < lat < 14.2):
            missing += 1
            continue
        points.append(TO_UTM.transform(lon, lat))
    if len(points) < 10:
        raise ValueError("Too few pretest geocoded underpasses")
    return np.asarray(points), {"catalog_rows": len(rows),
                                "missing_or_bad_coordinates": missing,
                                "usable_points": len(points),
                                "catalog_modified": "2024-06-07"}


def district_features(district: dict, rr: np.ndarray, cc: np.ndarray,
                      canal_tree: STRtree, canal_lines: list,
                      pumps: np.ndarray, capacities: np.ndarray,
                      ponds: np.ndarray, volumes: np.ndarray,
                      underpass: np.ndarray) -> tuple[np.ndarray, dict]:
    spec = grid_spec(district["polygon"])
    values = np.zeros((len(rr), len(FEATURES)), dtype=np.float32)
    centres = np.empty((len(rr), 2), dtype=np.float64)
    canal_length = np.zeros(len(rr), dtype=np.float64)
    canal_distance = np.full(len(rr), 10000.0, dtype=np.float64)
    for i, (r, c) in enumerate(zip(rr, cc)):
        cell = cell_box(spec, int(r), int(c))
        centre = cell.centroid
        centres[i] = centre.x, centre.y
        nearest = canal_tree.nearest(centre)
        if nearest is not None:
            canal_distance[i] = min(10000, centre.distance(canal_lines[int(nearest)]))
        canal_length[i] = sum(canal_lines[int(j)].intersection(cell).length
                              for j in canal_tree.query(cell))
    pump_dist = np.hypot(centres[:, None, 0] - pumps[None, :, 0],
                         centres[:, None, 1] - pumps[None, :, 1])
    pond_dist = np.hypot(centres[:, None, 0] - ponds[None, :, 0],
                         centres[:, None, 1] - ponds[None, :, 1])
    under_dist = np.hypot(centres[:, None, 0] - underpass[None, :, 0],
                          centres[:, None, 1] - underpass[None, :, 1])
    values[:, 0] = np.log1p(canal_length)
    values[:, 1] = np.log1p(canal_distance)
    values[:, 2] = np.log1p((pump_dist <= 1000) @ capacities)
    values[:, 3] = np.log1p((pond_dist <= 2000) @ volumes)
    values[:, 4] = np.log1p(np.minimum(10000, pond_dist.min(axis=1)))
    values[:, 5] = (under_dist.min(axis=1) <= 250).astype(np.float32)
    values[:, 6] = np.log1p(np.minimum(10000, under_dist.min(axis=1)))
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite infrastructure feature for {district['code']}")
    audit = {"cells": len(rr), "cells_with_mapped_canal_line": int(np.count_nonzero(canal_length)),
             "cells_with_capacity_within_1km": int(np.count_nonzero(values[:, 2])),
             "cells_with_pond_within_2km": int(np.count_nonzero(values[:, 3])),
             "cells_with_underpass_within_250m": int(np.count_nonzero(values[:, 5])),
             "median_canal_distance_m": round(float(np.median(canal_distance)), 1)}
    return values, audit


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    sources = {"pumps": ("pumpstation_2023.csv", PUMP_URL),
               "ponds": ("retention_ponds_2021.zip", POND_URL),
               "underpasses": ("underpasses_2024.csv", UNDERPASS_URL)}
    source_raw = {name: source_bytes(*spec) for name, spec in sources.items()}
    pump_xy, pump_capacity, pump_audit = pump_stations(source_raw["pumps"])
    pond_xy, pond_volume, pond_audit = retention_ponds(source_raw["ponds"])
    under_xy, under_audit = underpasses(source_raw["underpasses"])
    districts = load_districts()
    bounds = [d["polygon"].bounds for d in districts]
    bbox = (min(v[0] for v in bounds) - 5000, min(v[1] for v in bounds) - 5000,
            max(v[2] for v in bounds) + 5000, max(v[3] for v in bounds) + 5000)
    handler = CanalWays(bbox)
    handler.apply_file(str(OSM_PBF), locations=True)
    if len(handler.lines) < 100:
        raise ValueError("Too few mapped canals")
    canal_tree = STRtree(handler.lines)
    audit = {"features": list(FEATURES),
             "source_urls": {name: url for name, (_, url) in sources.items()},
             "source_sha256": {name: hashlib.sha256(raw).hexdigest()
                               for name, raw in source_raw.items()},
             "osm_source": OSM_PBF.name, "osm_snapshot": "2025-01-01",
             "osm_sha256": hashlib.sha256(OSM_PBF.read_bytes()).hexdigest(),
             "osm_canal_ways_in_city_bbox": len(handler.lines),
             "pumps": pump_audit, "ponds": pond_audit, "underpasses": under_audit,
             "limitations": ["asset proximity does not establish drainage connectivity",
                             "static inventories may not reflect earlier operating status",
                             "pump capacity units need source confirmation",
                             "2024 underpass catalog has nine rows without coordinates"],
             "districts": {}}
    for district in districts:
        code = district["code"]
        meta = json.loads((FEATURE_CACHE / f"{code}_meta.json").read_text())
        rr, cc = np.asarray(meta["valid_rows"]), np.asarray(meta["valid_cols"])
        spec = grid_spec(district["polygon"])
        if list(spec[k] for k in ("rows", "cols")) != meta["shape"][2:]:
            raise ValueError(f"Grid changed for {code}")
        values, district_audit = district_features(
            district, rr, cc, canal_tree, handler.lines,
            pump_xy, pump_capacity, pond_xy, pond_volume, under_xy)
        path = FEATURE_DIR / f"{code}.npz"
        np.savez_compressed(path, values=values, rows=rr, cols=cc,
                            feature_names=np.asarray(FEATURES))
        audit["districts"][code] = district_audit
        print(code, district["name_en"], district_audit, flush=True)
    (FEATURE_DIR / "audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
