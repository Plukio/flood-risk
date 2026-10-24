"""Build 250 m low-point and surface-cover features for two study districts.

Sources: BMA 10 m LiDAR DTM, ESA WorldCover 2021, OSM 2025 ways. The D8
accumulation is terrain-only and does not represent engineered drainage.

Run: .venv/bin/python flood_bangkok_new_static_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.ndimage import uniform_filter

from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_surface_grid import (
    GRID_M, OSM_PBF, OUTPUT, SurfaceOSM, grid_spec, osm_arrays,
)


CODES = ("1006", "1040")
WORLD_COVER_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
FEATURES = (
    "lidar_p10_m", "lidar_p50_p10_m", "lidar_low_depth_500m_m",
    "lidar_low_fraction_500m", "terrain_d8_upstream_cells_log",
    "worldcover_built_fraction_2021", "worldcover_vegetated_fraction_2021",
    "worldcover_water_fraction_2021", "worldcover_valid_fraction_2021",
    "osm_building_fraction_2025", "osm_road_length_log_2025",
)


def reduce_10m(data: np.ndarray, rows: int, cols: int, func) -> np.ndarray:
    if data.shape != (rows * 25, cols * 25):
        raise ValueError(f"Expected 25 terrain pixels per 250 m cell; got {data.shape}")
    blocks = data.reshape(rows, 25, cols, 25)
    return func(blocks)


def d8_upstream(median: np.ndarray) -> tuple[np.ndarray, int]:
    rows, cols = median.shape
    downstream = np.full(rows * cols, -1, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            best_slope, best = 0.0, -1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < rows and 0 <= cc < cols):
                        continue
                    slope = (median[r, c] - median[rr, cc]) / (dr * dr + dc * dc) ** 0.5
                    if slope > best_slope:
                        best_slope, best = slope, rr * cols + cc
            downstream[r * cols + c] = best
    accumulated = np.ones(rows * cols, dtype=np.float32)
    for index in np.argsort(median.ravel())[::-1]:
        next_index = downstream[index]
        if next_index >= 0:
            accumulated[next_index] += accumulated[index]
    return accumulated.reshape(rows, cols) - 1, int(np.count_nonzero(downstream < 0))


def worldcover_url() -> tuple[str, str]:
    result = requests.post(WORLD_COVER_SEARCH, json={
        "collections": ["esa-worldcover"],
        "bbox": [100.35, 13.60, 100.73, 13.86], "limit": 10}, timeout=30).json()
    matches = [item for item in result.get("features", [])
               if item["id"] == "ESA_WorldCover_10m_2021_v200_N12E099"]
    if len(matches) != 1:
        raise ValueError("ESA WorldCover 2021 Bangkok tile not found uniquely")
    item = planetary_computer.sign(matches[0])
    return item["assets"]["map"]["href"], matches[0]["id"]


def main() -> None:
    districts = {d["code"]: d for d in load_districts() if d["code"] in CODES}
    world_url, world_id = worldcover_url()
    audit = {"bma_terrain_source": "https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer",
             "worldcover_item": world_id,
             "osm_source": OSM_PBF.name, "osm_snapshot_date": "2025-01-01",
             "feature_names": list(FEATURES),
             "d8_limit": "Steepest strictly downhill neighbor on 250 m cell medians inside each district bbox; no pipe or gate flow and boundary-truncated catchments",
             "districts": {}}
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
        with rasterio.open(world_url) as world:
            for code in CODES:
                district = districts[code]
                grid = grid_spec(district["polygon"])
                path = (Path(__file__).resolve().parent / "data/flood_bangkapi_surface/dtm_bangkapi_10m.tif"
                        if code == "1006" else CACHE / "dtm_1040_10m.tif")
                with rasterio.open(path) as src:
                    if (src.crs.to_epsg() != 32647 or src.shape != (
                            grid["height"], grid["width"]) or
                            any(abs(a - b) > 1 for a, b in zip(src.res, (10, 10)))):
                        raise ValueError(f"BMA terrain grid mismatch for {code}")
                    terrain = src.read(1).astype(np.float32)
                good = np.isfinite(terrain) & (terrain > -20) & (terrain < 60)
                if good.mean() < 0.99:
                    raise ValueError(f"Too many invalid BMA terrain pixels for {code}")
                clean = np.where(good, terrain, float(np.median(terrain[good])))
                rows, cols = grid["rows"], grid["cols"]
                p10 = reduce_10m(clean, rows, cols,
                                 lambda b: np.percentile(b, 10, axis=(1, 3)))
                p50 = reduce_10m(clean, rows, cols,
                                 lambda b: np.percentile(b, 50, axis=(1, 3)))
                local_mean = uniform_filter(clean, size=51, mode="nearest")
                low_fraction = reduce_10m((clean < local_mean - 0.3).astype(np.float32),
                                          rows, cols, lambda b: b.mean(axis=(1, 3)))
                neighborhood = reduce_10m(local_mean, rows, cols,
                                          lambda b: b.mean(axis=(1, 3)))
                upstream, sinks = d8_upstream(p50)
                with WarpedVRT(world, crs="EPSG:32647", transform=grid["transform"],
                               width=grid["width"], height=grid["height"],
                               resampling=Resampling.nearest) as vrt:
                    land = vrt.read(1)
                valid = (land > 0).astype(np.float32)
                denominator = reduce_10m(valid, rows, cols,
                                         lambda b: b.sum(axis=(1, 3)))
                def fraction(classes):
                    numerator = reduce_10m(np.isin(land, classes).astype(np.float32),
                                           rows, cols, lambda b: b.sum(axis=(1, 3)))
                    return np.divide(numerator, denominator,
                                     out=np.full((rows, cols), np.nan, dtype=np.float32),
                                     where=denominator > 0)
                if code == "1006":
                    osm, _ = osm_arrays(grid)
                else:
                    handler = SurfaceOSM(grid)
                    handler.apply_file(str(OSM_PBF), locations=True)
                    osm = {"building_area_m2": handler.building_area_m2,
                           "road_m": handler.road_m}
                layers = {
                    "lidar_p10_m": p10,
                    "lidar_p50_p10_m": p50 - p10,
                    "lidar_low_depth_500m_m": np.maximum(0, neighborhood - p10),
                    "lidar_low_fraction_500m": low_fraction,
                    "terrain_d8_upstream_cells_log": np.log1p(upstream),
                    "worldcover_built_fraction_2021": fraction((50,)),
                    "worldcover_vegetated_fraction_2021": fraction((10, 20, 30, 40, 90, 95)),
                    "worldcover_water_fraction_2021": fraction((80,)),
                    "worldcover_valid_fraction_2021": denominator / 625,
                    "osm_building_fraction_2025": np.clip(osm["building_area_m2"] / (GRID_M ** 2), 0, 1),
                    "osm_road_length_log_2025": np.log1p(osm["road_m"]),
                }
                with np.load(OUTPUT / f"surface_shared_state_{code}_daily_cells.npz") as saved:
                    rr, cc = saved["rows"], saved["cols"]
                values = np.column_stack([layers[name][rr, cc] for name in FEATURES]).astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite static features in {code}")
                np.savez_compressed(CACHE / f"new_static_{code}_250m.npz",
                                    values=values, names=np.asarray(FEATURES),
                                    rows=rr, cols=cc)
                audit["districts"][code] = {
                    "cells": len(rr), "terrain_valid_fraction": round(float(good.mean()), 4),
                    "worldcover_valid_median": round(float(np.median(layers[
                        "worldcover_valid_fraction_2021"][rr, cc])), 4),
                    "terrain_d8_sink_cells_in_bbox": sinks,
                    "osm_building_coverage_cells": int(np.count_nonzero(
                        layers["osm_building_fraction_2025"][rr, cc])),
                    "worldcover_built_coverage_cells": int(np.count_nonzero(
                        layers["worldcover_built_fraction_2021"][rr, cc]))}
                print(code, audit["districts"][code], flush=True)
    (CACHE / "new_static_250m_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
