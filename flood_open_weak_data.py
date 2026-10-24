"""Build additional Thai flood training scenes from open, weakly labeled maps.

The output preserves three states: mapped water, analyzed non-water, and
unobserved. EOS-RS and GISTDA transparent/empty cells are unobserved, not dry.
The weather block is retrospective daily exposure because exact satellite
acquisition hours are missing for several products.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import osmium
import rasterio
import requests
import shapefile
import shapely
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import LineString, shape

from flood_bangkok_transfer import RAIN_URL
from flood_bangkapi_surface_grid import OSM_PBF
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date
from flood_gistda_multiregion_data import (
    CACHE as STATIC_CACHE, GRID_M, SIZE, TO_UTM, osm_arrays, specs, terrain,
)
from flood_gistda_multiregion_worldcover import CHANNELS as COVER_NAMES
from flood_gistda_province_expand_data import patch_spec, worldcover


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "artifacts" / "open_flood_data"
CACHE = ROOT / "data" / "flood_open_weak_train"
OUTPUT = ROOT / "artifacts" / "flood_open_weak_train"
SUBPIXEL_M = 50
SUBSIZE = SIZE * GRID_M // SUBPIXEL_M
NEW_PATCHES = {
    "bang_kapi_core": (100.64, 13.77),
    "chiang_mai_city": (98.985, 18.79),
    "hat_yai_city": (100.47, 7.01),
}
UNOSAT_2021_DATES = ("2021-10-02", "2021-10-10", "2021-10-20", "2021-11-01")
STATIC_NAMES = ("srtm_elevation_m", "srtm_relative_1250m_m", "srtm_slope",
                "osm_canal_length_log", "osm_canal_distance_log",
                "osm_road_length_log", "osm_road_distance_log")
FEATURE_NAMES = STATIC_NAMES + COVER_NAMES
WEATHER_NAMES = tuple("retrospective_" + x for x in RAIN_FEATURES) + ("month_sin", "month_cos")


def all_specs() -> dict:
    old = specs()
    province = json.loads((ROOT / "artifacts/flood_gistda_multiregion/"
                           "province_expansion_data_audit.json").read_text())["patches"]
    return {**old, **province,
            **{name: patch_spec(*center) for name, center in NEW_PATCHES.items()}}


class LocalWays(osmium.SimpleHandler):
    """Read one Thailand snapshot while retaining only the three new patches."""

    def __init__(self, patches: dict):
        super().__init__()
        self.bounds = [s["query_bbox"] for s in patches.values()]
        self.canal: list = []
        self.road: list = []

    def way(self, way) -> None:
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
        lon = [n.lon for n in nodes]
        lat = [n.lat for n in nodes]
        if not any(min(lon) <= b[2] and max(lon) >= b[0]
                   and min(lat) <= b[3] and max(lat) >= b[1] for b in self.bounds):
            return
        x, y = TO_UTM.transform(lon, lat)
        line = LineString(zip(x, y))
        if line.length > 0:
            getattr(self, kind).append(line)


def make_static(patches: dict) -> dict:
    pending = {n: patches[n] for n in NEW_PATCHES
               if not (STATIC_CACHE / f"{n}_static_250m.npz").exists()}
    audit = {}
    if pending:
        handler = LocalWays(pending)
        handler.apply_file(str(OSM_PBF), locations=True)
        if not handler.road or not handler.canal:
            raise ValueError("No OSM road or canal lines for new patches")
        for name, spec in pending.items():
            west, south, east, north = spec["utm_bounds"]
            # Querying only local lines avoids building an all-Thailand tree.
            def nearby(lines: list) -> list:
                result = []
                for line in lines:
                    x0, y0, x1, y1 = line.bounds
                    if x0 <= east + 1000 and x1 >= west - 1000 \
                            and y0 <= north + 1000 and y1 >= south - 1000:
                        result.append(line)
                return result
            canal = nearby(handler.canal)
            road = nearby(handler.road)
            if not canal or not road:
                raise ValueError(f"Missing OSM lines in {name}")
            elev = terrain(spec)
            canal_len, canal_dist = osm_arrays(spec, canal)
            road_len, road_dist = osm_arrays(spec, road)
            x = np.stack((elev[0], elev[1], elev[2],
                          np.log1p(canal_len), np.log1p(canal_dist),
                          np.log1p(road_len), np.log1p(road_dist))).astype(np.float32)
            np.savez_compressed(STATIC_CACHE / f"{name}_static_250m.npz", values=x)
            audit[name] = {"roads": len(road), "canals": len(canal)}
            print("static", name, audit[name], flush=True)
    worldcover({n: patches[n] for n in NEW_PATCHES})
    return audit


def load_static(name: str) -> np.ndarray:
    with np.load(STATIC_CACHE / f"{name}_static_250m.npz") as z:
        static = z["values"].astype(np.float32)
    with np.load(STATIC_CACHE / f"{name}_worldcover_2021_250m.npz") as z:
        if tuple(z["names"].astype(str)) != COVER_NAMES:
            raise ValueError(f"WorldCover channels changed for {name}")
        cover = z["values"].astype(np.float32)
    if static.shape != (7, SIZE, SIZE) or cover.shape != (6, SIZE, SIZE):
        raise ValueError(f"Static feature shape changed for {name}")
    return np.concatenate((static, cover))


def weather(name: str, day: date, spec: dict) -> np.ndarray:
    path = CACHE / f"rain_{name}_{day}.json"
    if not path.exists():
        lon, lat = spec["center_lonlat"]
        response = requests.get(RAIN_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(day - timedelta(days=3)), "end_date": str(day),
            "hourly": "precipitation", "timezone": "Asia/Bangkok"}, timeout=60)
        response.raise_for_status()
        path.write_text(response.text)
    values = rainfall_by_date(day, day, path)[day]
    rain = [math.log1p(max(0.0, values[key])) for key in RAIN_FEATURES]
    return np.asarray(rain + [math.sin(2 * math.pi * day.month / 12),
                              math.cos(2 * math.pi * day.month / 12)], np.float32)


def partial_shape(whole: shapefile.Shape, bbox: list[float]) -> dict | None:
    """Select complete polygon rings touching a patch bbox, preserving holes."""
    subset = shapefile.Shape(shapeType=whole.shapeType)
    stops = list(whole.parts) + [len(whole.points)]
    for first, last in zip(stops[:-1], stops[1:]):
        points = whole.points[first:last]
        if len(points) < 4:
            continue
        xs, ys = zip(*points)
        if min(xs) > bbox[2] or max(xs) < bbox[0] or \
                min(ys) > bbox[3] or max(ys) < bbox[1]:
            continue
        subset.parts.append(len(subset.points))
        subset.points.extend(points)
    if not subset.parts:
        return None
    geom = shape(subset.__geo_interface__)
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return geom.__geo_interface__


def shp_shapes(archive: zipfile.ZipFile, ending: str) -> list[shapefile.Shape]:
    name = next(n for n in archive.namelist() if n.endswith(ending + ".shp"))
    stem = name[:-4]
    reader = shapefile.Reader(shp=io.BytesIO(archive.read(stem + ".shp")),
                              shx=io.BytesIO(archive.read(stem + ".shx")),
                              dbf=io.BytesIO(archive.read(stem + ".dbf")))
    return list(reader.iterShapes())


def shp_geometry(shapes: list[shapefile.Shape],
                 bbox: list[float]) -> dict | None:
    geoms = [partial_shape(s, bbox) for s in shapes]
    geoms = [g for g in geoms if g is not None]
    if not geoms:
        return None
    return geoms[0] if len(geoms) == 1 else {
        "type": "GeometryCollection", "geometries": geoms}


def fraction(geom: dict | None, spec: dict) -> np.ndarray:
    if geom is None:
        return np.zeros((SIZE, SIZE), np.float32)
    west, _, _, north = spec["utm_bounds"]
    source = shape(geom)
    if not source.is_valid:
        source = shapely.make_valid(source)
    to_utm = Transformer.from_crs(4326, 32647, always_xy=True).transform
    projected = shapely.transform(source, to_utm, interleaved=False)
    hi = rasterize([(projected, 1)], out_shape=(SUBSIZE, SUBSIZE),
                   transform=from_origin(west, north, SUBPIXEL_M, SUBPIXEL_M),
                   dtype="uint8", all_touched=False)
    return hi.reshape(SIZE, 5, SIZE, 5).mean(axis=(1, 3)).astype(np.float32)


def tif_positive(tif: rasterio.io.DatasetReader, spec: dict) -> np.ndarray:
    west, _, _, north = spec["utm_bounds"]
    dest = np.zeros((SIZE, SIZE), np.float32)
    reproject((tif.read(4) > 0).astype(np.float32), dest,
              src_transform=tif.transform, src_crs=tif.crs,
              dst_transform=from_origin(west, north, GRID_M, GRID_M),
              dst_crs="EPSG:32647", resampling=Resampling.average)
    return dest


def save_scene(source: str, name: str, day: date, wet: np.ndarray,
               observed: np.ndarray, spec: dict, note: str,
               weather_override: np.ndarray | None = None) -> dict | None:
    static = load_static(name)
    water = static[FEATURE_NAMES.index("worldcover_water_fraction_2021")]
    # A 2021 permanent-water class is neither flood nor likely dry ground.
    usable = (water < .2).astype(np.float32)
    wet = np.clip(wet, 0, 1) * usable
    observed = np.clip(observed, 0, 1) * usable
    wet = np.minimum(wet, observed) if source.startswith("unosat") else wet
    if float(wet.sum()) < 1 and (not source.startswith("unosat")
                                 or float(observed.sum()) < 50):
        return None
    path = CACHE / f"{source}_{name}_{day}.npz"
    event_weather = weather(name, day, spec) if weather_override is None else weather_override
    np.savez_compressed(path, wet=wet, observed=observed,
                        static=static, weather=event_weather)
    return {"source": source, "region": name, "day": str(day),
            "path": str(path.relative_to(ROOT)),
            "wet_proxy_km2": round(float(wet.sum()) * .0625, 4),
            "observed_candidate_km2": round(float(observed.sum()) * .0625, 4),
            "note": note}


def build() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    patches = all_specs()
    static_audit = make_static(patches)
    scenes = []
    with zipfile.ZipFile(RAW / "unosat_central_2021_shp.zip") as z:
        names = [n for n in patches if n not in ("chiang_mai_city", "hat_yai_city")]
        for stamp in UNOSAT_2021_DATES:
            key = stamp.replace("-", "")
            stem = next(n[:-4] for n in z.namelist() if n.endswith(
                f"S1_{key}_FloodExtent_CentralRegion.shp") or n.endswith(
                f"S1_{key}_FloodExtent_Central_WesternRegions.shp"))
            flood_end = stem.split("/")[-1]
            analysis_end = flood_end.replace("FloodExtent", "AnalysisExtent")
            flood_shapes = shp_shapes(z, flood_end)
            analysis_shapes = shp_shapes(z, analysis_end)
            for name in names:
                spec = patches[name]
                bbox = spec["query_bbox"]
                observed = fraction(shp_geometry(analysis_shapes, bbox), spec)
                if observed.mean() < .15:
                    continue
                wet = fraction(shp_geometry(flood_shapes, bbox), spec)
                row = save_scene("unosat2021", name, date.fromisoformat(stamp),
                                 wet, observed, spec, "Sentinel-1; analyzed nonwater is weak dry")
                if row:
                    scenes.append(row)
            print("UNOSAT", stamp, "scenes", len(scenes), flush=True)
    with zipfile.ZipFile(RAW / "unosat_chiangmai_2024_shp.zip") as z:
        name = "chiang_mai_city"
        spec = patches[name]
        bbox = spec["query_bbox"]
        stem = "CHIANGMAI_20241005_20241010_"
        observed = fraction(shp_geometry(shp_shapes(z, stem + "AnalysisExtent"), bbox), spec)
        cloud = fraction(shp_geometry(shp_shapes(z, stem + "CloudObstruction"), bbox), spec)
        observed = np.clip(observed - cloud, 0, 1)
        wet = fraction(shp_geometry(shp_shapes(z, stem + "AccumulatedFlood"), bbox), spec)
        row = save_scene("unosat2024", name, date(2024, 10, 10), wet,
                         observed, spec, "6–10 Oct cumulative GF-1/ST-2 extent; rain day is approximate")
        if row:
            scenes.append(row)
    with rasterio.open(RAW / "eosrs_2026_bangkok.tif") as tif:
        for name in ("nong_chok", "bang_kapi_core"):
            spec = patches[name]
            wet = tif_positive(tif, spec)
            row = save_scene("eos2026", name, date(2026, 9, 27), wet,
                             np.zeros_like(wet), spec,
                             "Sentinel-1 likely water; transparent pixels unobserved")
            if row:
                scenes.append(row)
    with zipfile.ZipFile(RAW / "eosrs_2025_hatyai.zip") as z:
        tif_name = next(n for n in z.namelist() if n.endswith(".tif"))
        with rasterio.MemoryFile(z.read(tif_name)) as mem:
            with mem.open() as tif:
                name = "hat_yai_city"
                spec = patches[name]
                wet = tif_positive(tif, spec)
                row = save_scene("eos2025", name, date(2025, 11, 23), wet,
                                 np.zeros_like(wet), spec,
                                 "Sentinel-1 likely water; transparent pixels unobserved")
                if row:
                    scenes.append(row)
    # Retain the original multi-region positive-only training maps alongside
    # the added open sources; do not mark GISTDA zero-polygon cells dry.
    for name, spec in patches.items():
        if name in NEW_PATCHES or name == "nong_chok":
            continue
        wet_path = STATIC_CACHE / f"{name}_wet_250m.npz"
        rain_path = STATIC_CACHE / f"{name}_rain_features_2026.npz"
        if not wet_path.exists() or not rain_path.exists():
            continue
        with np.load(wet_path) as z:
            mapped = z["wet_area_m2"].astype(np.float32)
            dates = list(z["dates"].astype(str))
        with np.load(rain_path) as z:
            rain = z["values"].astype(np.float32)
            if list(z["dates"].astype(str)) != dates:
                raise ValueError(f"GISTDA rain dates differ for {name}")
        for i, stamp in enumerate(dates):
            day = date.fromisoformat(stamp)
            global_weather = np.concatenate((rain[i], [
                math.sin(2 * math.pi * day.month / 12),
                math.cos(2 * math.pi * day.month / 12)])).astype(np.float32)
            row = save_scene("gistda2026", name, day,
                             np.clip(mapped[i] / (GRID_M * GRID_M), 0, 1),
                             np.zeros((SIZE, SIZE), np.float32), spec,
                             "Daily GISTDA mapped water; outside polygons unobserved",
                             weather_override=global_weather)
            if row:
                scenes.append(row)
    if not any(s["source"] == "unosat2021" for s in scenes):
        raise ValueError("No 2021 UNOSAT training scenes")
    if not any(s["source"] == "unosat2024" for s in scenes):
        raise ValueError("No 2024 Chiang Mai training scene")
    if not any(s["source"] == "eos2025" for s in scenes):
        raise ValueError("No 2025 Hat Yai training scene")
    if not any(s["source"] == "eos2026" for s in scenes):
        raise ValueError("No 2026 Bangkok training scene")
    if not any(s["source"] == "gistda2026" for s in scenes):
        raise ValueError("No existing GISTDA training scenes")
    audit = {"features": FEATURE_NAMES, "weather": WEATHER_NAMES,
             "patch_size": [SIZE, SIZE], "cell_m": GRID_M,
             "fraction_subpixel_m": SUBPIXEL_M,
             "new_static": static_audit, "scenes": scenes,
             "label_semantics": "UNOSAT observed nonwater is weak dry; EOS transparent and GISTDA outside polygons are unknown",
             "rain_limit": "Open-Meteo full local calendar-day rain may include hours after the satellite pass; retrospective exposure only",
             "source_catalogs": {
                "unosat2021": "https://data.humdata.org/dataset/water-extents-in-central-and-western-parts-thailand-as-of-21-october-2021",
                "unosat2024": "https://data.humdata.org/dataset/water-extents-from-5-to-10-october-2024-over-chiang-mai-and-lamphun-provinces",
                "eos2025": "https://data.humdata.org/dataset/eos-rs-thailand-hat-yai-floods-nov-2025",
                "eos2026": "https://data.humdata.org/dataset/eos-rs-thailand-flood-bangkok-sep-2026",
                "gistda2026": "https://data.go.th/dataset/flood-disaster-data"}}
    (OUTPUT / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("saved", len(scenes), "weak-label scenes", flush=True)
    return audit


if __name__ == "__main__":
    build()
