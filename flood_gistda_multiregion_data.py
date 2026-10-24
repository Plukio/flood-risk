"""Build a transferable 250 m positive-water dataset in six Thai regions.

GISTDA supplies mapped-water polygons, not a verified dry footprint. Zero
stored wet area therefore means unobserved/unknown outside positive polygons.
"""

from __future__ import annotations

import gzip
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import osmium
import rasterio
import requests
import shapely
from pyproj import Transformer
from scipy.ndimage import uniform_filter
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from flood_bangkok_gistda_extent_audit import COLLECTIONS, public_example_key
from flood_bangkok_transfer import RAIN_URL
from flood_bangkapi_surface_grid import OSM_PBF
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "flood_gistda_multiregion"
OUTPUT = ROOT / "artifacts" / "flood_gistda_multiregion"
GRID_M = 250
SIZE = 80  # each fixed patch is 20 km x 20 km
CENTERS = {
    "nong_chok": (100.86, 13.84),
    "pathum_thani": (100.64, 14.09),
    "chachoengsao": (101.19, 13.79),
    "ayutthaya_holdout": (100.57, 14.38),
    "nakhon_sawan": (100.14, 15.70),
    "phitsanulok": (100.26, 16.82),
}
TO_UTM = Transformer.from_crs(4326, 32647, always_xy=True)
TO_LONLAT = Transformer.from_crs(32647, 4326, always_xy=True)
SOURCE_DEM = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"


def specs() -> dict:
    out = {}
    for name, (lon, lat) in CENTERS.items():
        cx, cy = TO_UTM.transform(lon, lat)
        west = round(cx / GRID_M) * GRID_M - SIZE * GRID_M / 2
        north = round(cy / GRID_M) * GRID_M + SIZE * GRID_M / 2
        east, south = west + SIZE * GRID_M, north - SIZE * GRID_M
        corners = [TO_LONLAT.transform(x, y) for x in (west, east)
                   for y in (south, north)]
        out[name] = {"center_lonlat": [lon, lat],
                     "utm_bounds": [west, south, east, north],
                     "query_bbox": [min(x for x, y in corners) - .003,
                                    min(y for x, y in corners) - .003,
                                    max(x for x, y in corners) + .003,
                                    max(y for x, y in corners) + .003]}
    return out


def boxes(spec: dict) -> np.ndarray:
    west, _, _, north = spec["utm_bounds"]
    col, row = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    x0 = west + col.ravel() * GRID_M
    y1 = north - row.ravel() * GRID_M
    return shapely.box(x0, y1 - GRID_M, x0 + GRID_M, y1)


def request_json(url: str, params: dict) -> dict:
    last = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=80)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last = error
    raise RuntimeError(f"GISTDA request failed after retries: {type(last).__name__}")


def item_assets(days: list[date], key: str) -> dict:
    result = {}
    for day in days:
        collection = f"flood_r2_{day:%Y%m%d}"
        item = request_json(f"{COLLECTIONS}/{collection}/items",
                            {"api_key": key, "limit": 20})["features"]
        if len(item) != 1 or "data" not in item[0]["assets"]:
            raise ValueError(f"Missing GISTDA data asset on {day}")
        result[day] = item[0]["assets"]["data"]["href"]
    return result


def one_label(job: tuple) -> tuple[str, date, np.ndarray, dict]:
    name, spec, day, asset = job
    bbox = ",".join(str(x) for x in spec["query_bbox"])
    features = []
    offset = 0
    while True:
        data = request_json(asset, {"bbox": bbox, "limit": 10000,
                                    "offset": offset})
        batch = data.get("features", [])
        features.extend(batch)
        matched = data.get("numberMatched")
        if matched is None:
            raise ValueError("GISTDA asset lacks numberMatched")
        offset += len(batch)
        if offset >= matched:
            break
        if not batch:
            raise ValueError("GISTDA pagination stalled")
    to_utm = TO_UTM.transform
    patch = shapely.box(*spec["utm_bounds"])
    polygons = []
    for feature in features:
        geometry = shapely.transform(shape(feature["geometry"]), to_utm,
                                     interleaved=False)
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
        if geometry.intersects(patch):
            clipped = geometry.intersection(patch)
            if clipped.area > 1:
                polygons.append(clipped)
    wet = np.zeros(SIZE * SIZE, dtype=np.float32)
    if polygons:
        merged = unary_union(polygons)
        parts = list(shapely.get_parts(merged))
        tree = STRtree(parts)
        cells = boxes(spec)
        for i, cell in enumerate(cells):
            nearby = tree.query(cell)
            if len(nearby):
                wet[i] = sum(cell.intersection(parts[int(j)]).area for j in nearby)
        if abs(float(wet.sum()) - merged.area) / merged.area > .01:
            raise ValueError(f"Cell area differs from clipped polygon area: {name} {day}")
    return name, day, wet.reshape(SIZE, SIZE), {
        "bbox_matched_polygons": matched, "clipped_polygons": len(polygons),
        "mapped_wet_km2": round(float(wet.sum()) / 1e6, 5),
        "mapped_wet_cells": int(np.count_nonzero(wet > 1))}


def labels(all_specs: dict, days: list[date]) -> dict:
    previous_path = OUTPUT / "multiregion_data_audit.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    previous_coverage = previous.get("coverage", {})
    pending = {}
    for name, spec in all_specs.items():
        path = CACHE / f"{name}_wet_250m.npz"
        if path.exists() and name in previous_coverage:
            with np.load(path) as saved:
                if list(saved["dates"].astype(str)) == [str(d) for d in days]:
                    print("GISTDA cached", name, flush=True)
                    continue
        pending[name] = spec
    if not pending:
        return {name: previous_coverage[name] for name in all_specs}
    key = public_example_key()
    assets = item_assets(days, key)
    jobs = [(name, spec, day, assets[day]) for name, spec in pending.items()
            for day in days]
    arrays = {name: np.zeros((len(days), SIZE, SIZE), dtype=np.float32)
              for name in pending}
    audits = {name: previous_coverage[name] for name in all_specs if name not in pending}
    audits.update({name: {} for name in pending})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one_label, job) for job in jobs]
        for future in as_completed(futures):
            name, day, wet, audit = future.result()
            arrays[name][days.index(day)] = wet
            audits[name][str(day)] = audit
            print("GISTDA", name, day, audit, flush=True)
    for name, wet in arrays.items():
        np.savez_compressed(CACHE / f"{name}_wet_250m.npz",
                            wet_area_m2=wet, dates=np.asarray([str(d) for d in days]))
    return audits


def dem_tile(lat: int, lon: int) -> Path:
    name = f"N{lat:02d}E{lon:03d}"
    path = CACHE / "dem" / f"{name}.hgt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{SOURCE_DEM}/N{lat:02d}/{name}.hgt.gz"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path.write_bytes(gzip.decompress(response.content))
    return path


def terrain(spec: dict) -> np.ndarray:
    west, _, _, north = spec["utm_bounds"]
    col, row = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    x = west + (col + .5) * GRID_M
    y = north - (row + .5) * GRID_M
    lon, lat = TO_LONLAT.transform(x, y)
    elev = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
    for tile_lat, tile_lon in sorted(set(zip(np.floor(lat).astype(int).ravel(),
                                          np.floor(lon).astype(int).ravel()))):
        mask = (np.floor(lat) == tile_lat) & (np.floor(lon) == tile_lon)
        with rasterio.open(dem_tile(tile_lat, tile_lon)) as source:
            values = source.read(1)
            rr, cc = rasterio.transform.rowcol(source.transform, lon[mask], lat[mask])
            elev[mask] = values[np.clip(rr, 0, values.shape[0]-1),
                                np.clip(cc, 0, values.shape[1]-1)]
    valid = np.isfinite(elev) & (elev > -1000)
    if valid.mean() < .99:
        raise ValueError("SRTM terrain missing across patch")
    local = uniform_filter(elev, size=5, mode="nearest")
    dy, dx = np.gradient(elev, GRID_M)
    return np.stack((elev, elev - local, np.hypot(dx, dy))).astype(np.float32)


class OSMWays(osmium.SimpleHandler):
    def __init__(self, all_specs: dict):
        super().__init__()
        b = [s["query_bbox"] for s in all_specs.values()]
        self.bounds = (min(x[0] for x in b), min(x[1] for x in b),
                       max(x[2] for x in b), max(x[3] for x in b))
        self.canal = []
        self.road = []

    def way(self, way):
        tags = way.tags
        kind = "canal" if tags.get("waterway") == "canal" else (
            "road" if tags.get("highway") in {
                "motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service", "living_street",
                "motorway_link", "trunk_link", "primary_link", "secondary_link",
                "tertiary_link"} else None)
        if kind is None:
            return
        nodes = [n for n in way.nodes if n.location.valid()]
        if len(nodes) < 2:
            return
        west, south, east, north = self.bounds
        lon = [n.lon for n in nodes]
        lat = [n.lat for n in nodes]
        if max(lon) < west or min(lon) > east or max(lat) < south or min(lat) > north:
            return
        xs, ys = TO_UTM.transform(lon, lat)
        line = LineString(zip(xs, ys))
        if line.length > 0:
            getattr(self, kind).append(line)


def osm_arrays(spec: dict, lines: list) -> tuple[np.ndarray, np.ndarray]:
    cells = boxes(spec)
    tree = STRtree(lines)
    length = np.zeros(len(cells), dtype=np.float32)
    distance = np.full(len(cells), 10000.0, dtype=np.float32)
    for i, cell in enumerate(cells):
        near = tree.nearest(cell.centroid)
        if near is not None:
            distance[i] = min(10000.0, cell.centroid.distance(lines[int(near)]))
        length[i] = sum(cell.intersection(lines[int(j)]).length
                        for j in tree.query(cell))
    return length.reshape(SIZE, SIZE), distance.reshape(SIZE, SIZE)


def static_features(all_specs: dict) -> dict:
    previous_path = OUTPUT / "multiregion_data_audit.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    previous_audit = previous.get("static_audit", {})
    pending = {name: spec for name, spec in all_specs.items()
               if not (CACHE / f"{name}_static_250m.npz").exists()
               or name not in previous_audit}
    if not pending:
        return {name: previous_audit[name] for name in all_specs}
    handler = OSMWays(pending)
    handler.apply_file(str(OSM_PBF), locations=True)
    if len(handler.canal) < 100 or len(handler.road) < 100:
        raise ValueError("Too few OSM canal or road ways")
    print("OSM loaded", len(handler.canal), "canals", len(handler.road), "roads", flush=True)
    audit = {name: previous_audit[name] for name in all_specs if name not in pending}
    for name, spec in pending.items():
        elevation = terrain(spec)
        canal_len, canal_dist = osm_arrays(spec, handler.canal)
        road_len, road_dist = osm_arrays(spec, handler.road)
        static = np.stack((elevation[0], elevation[1], elevation[2],
                           np.log1p(canal_len), np.log1p(canal_dist),
                           np.log1p(road_len), np.log1p(road_dist))).astype(np.float32)
        np.savez_compressed(CACHE / f"{name}_static_250m.npz", values=static)
        audit[name] = {"elev_min_m": round(float(elevation[0].min()), 2),
                       "elev_max_m": round(float(elevation[0].max()), 2),
                       "cells_with_canal": int(np.count_nonzero(canal_len)),
                       "cells_with_road": int(np.count_nonzero(road_len))}
        print("static", name, audit[name], flush=True)
    return audit


def rain_features(all_specs: dict, days: list[date]) -> dict:
    audit = {}
    for name, spec in all_specs.items():
        lon, lat = spec["center_lonlat"]
        path = CACHE / f"{name}_rain_2026.json"
        if not path.exists():
            params = {"latitude": lat, "longitude": lon,
                      "start_date": str(date.fromordinal(days[0].toordinal() - 3)),
                      "end_date": str(days[-1]), "hourly": "precipitation",
                      "timezone": "Asia/Bangkok"}
            response = requests.get(RAIN_URL, params=params, timeout=50)
            response.raise_for_status()
            path.write_text(response.text)
        values = rainfall_by_date(days[0], days[-1], path)
        matrix = np.asarray([[math.log1p(max(0, values[d][key]))
                              for key in RAIN_FEATURES] for d in days], dtype=np.float32)
        np.savez_compressed(CACHE / f"{name}_rain_features_2026.npz",
                            values=matrix, dates=np.asarray([str(d) for d in days]))
        source = json.loads(path.read_text())
        audit[name] = {"returned_grid_lonlat": [source["longitude"], source["latitude"]],
                       "daily_rain_mm": {str(d): round(values[d]["rain_today_mm"], 2)
                                         for d in days}}
        print("rain", name, audit[name]["daily_rain_mm"], flush=True)
    return audit


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scan = json.loads((ROOT / "artifacts/flood_bangkapi_surface/gistda_2026_bma_daily_scan.json").read_text())
    days = [date.fromisoformat(x["day"]) for x in scan["rows"]
            if x["polygons_intersecting_bma"] > 0]
    all_specs = specs()
    coverage = labels(all_specs, days)
    static = static_features(all_specs)
    rain = rain_features(all_specs, days)
    audit = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source_gistda": "https://data.go.th/dataset/flood-disaster-data",
             "source_dem": SOURCE_DEM, "source_osm_snapshot": OSM_PBF.name,
             "source_rain": RAIN_URL, "grid_m": GRID_M,
             "patch_size_cells": [SIZE, SIZE], "patches": all_specs,
             "days": [str(d) for d in days],
             "label_semantics": "positive GISTDA mapped-water area only; zero stored area means unobserved, not verified dry",
             "coverage": coverage, "static_audit": static, "rain_audit": rain}
    (OUTPUT / "multiregion_data_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")
    print("saved", OUTPUT / "multiregion_data_audit.json", flush=True)


if __name__ == "__main__":
    main()
