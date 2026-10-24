"""Build a place-wide Bang Kapi flood-model feature grid from open data.

Run: .venv/bin/python flood_bangkapi_surface_grid.py

The grid covers road and off-road surface. It does not invent flood
probabilities: the available BMA outcomes observe major roads only.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import osmium
import planetary_computer
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy.ndimage import uniform_filter
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import transform as shapely_transform

from flood_bangkok_transfer import load_districts


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "flood_bangkapi_surface"
OUTPUT = ROOT / "artifacts" / "flood_bangkapi_surface"
OSM_PBF = ROOT / "data" / "thailand-250101.osm.pbf"
S2_STAC_CACHE = ROOT / "data" / "flood_bangkapi_imagery_audit" / "sentinel-2-l2a_2025_bangkapi.json"
ANCHORS = ROOT / "artifacts" / "flood_bangkapi_site_ranking" / "site_ranking_2025.csv"
TRAFFY = CACHE / "traffy_flood_2025_minimal.csv"
TRAFFY_ACTIVITY = CACHE / "traffy_all_activity_2025_cell_month.csv"
DTM_URL = "https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer/exportImage"
PUMP_URL = ("https://data.bangkok.go.th/dataset/22475fbb-18ef-4705-9ce2-"
            "53f4907394e7/resource/0d645a7e-50f4-4c4f-ad1c-19db560b29c8/download/-.csv")
S2_SCENE_PREFIX = "S2A_MSIL2A_20250115T034111"
GRID_M = 250
PIXEL_M = 10
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "living_street", "service",
                "motorway_link", "trunk_link", "primary_link", "secondary_link",
                "tertiary_link"}
ESSENTIAL_AMENITIES = {"school", "kindergarten", "university", "college",
                       "hospital", "clinic", "pharmacy", "police", "fire_station",
                       "bus_station", "marketplace"}
TO_UTM = Transformer.from_crs(4326, 32647, always_xy=True)
TO_LONLAT = Transformer.from_crs(32647, 4326, always_xy=True)


def fetch(path: Path, url: str, params: dict | None = None) -> bytes:
    if path.exists():
        return path.read_bytes()
    response = requests.get(url, params=params, timeout=120,
                            headers={"User-Agent": "Bangkok open-data flood surface pilot"})
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def district_polygon() -> Polygon:
    districts = [d for d in load_districts() if d["name_en"] == "Bang Kapi"]
    if len(districts) != 1:
        raise ValueError("BMA Bang Kapi boundary was not found uniquely")
    return districts[0]["polygon"]


def grid_spec(polygon: Polygon) -> dict:
    x0, y0, x1, y1 = polygon.bounds
    west = math.floor(x0 / GRID_M) * GRID_M
    south = math.floor(y0 / GRID_M) * GRID_M
    east = math.ceil(x1 / GRID_M) * GRID_M
    north = math.ceil(y1 / GRID_M) * GRID_M
    return {"west": west, "south": south, "east": east, "north": north,
            "rows": int((north - south) / GRID_M),
            "cols": int((east - west) / GRID_M),
            "height": int((north - south) / PIXEL_M),
            "width": int((east - west) / PIXEL_M),
            "transform": from_origin(west, north, PIXEL_M, PIXEL_M)}


def get_dtm(grid: dict) -> np.ndarray:
    path = CACHE / "dtm_bangkapi_10m.tif"
    fetch(path, DTM_URL, {"bbox": f"{grid['west']},{grid['south']},{grid['east']},{grid['north']}",
                          "bboxSR": 32647, "imageSR": 32647,
                          "size": f"{grid['width']},{grid['height']}",
                          "format": "tiff", "pixelType": "F32", "f": "image"})
    with rasterio.open(path) as source:
        if (source.crs.to_epsg() != 32647 or source.width != grid["width"]
                or source.height != grid["height"] or abs(source.res[0] - PIXEL_M) > 0.01):
            raise ValueError("Unexpected BMA terrain export grid")
        data = source.read(1).astype(np.float32)
    valid = np.isfinite(data) & (data > -20) & (data < 60)
    if np.mean(valid) < 0.99:
        raise ValueError("BMA terrain has too many invalid pixels")
    return data


def s2_scene() -> dict:
    if not S2_STAC_CACHE.exists():
        raise FileNotFoundError("Run flood_bangkapi_imagery_audit.py first")
    items = json.loads(S2_STAC_CACHE.read_text())["features"]
    matches = [item for item in items if item["id"].startswith(S2_SCENE_PREFIX)]
    if len(matches) != 1:
        raise ValueError("The pre-season Sentinel-2 scene was not found uniquely")
    return matches[0]


def satellite_arrays(grid: dict) -> dict[str, np.ndarray]:
    path = CACHE / "sentinel2_2025_01_15_bangkapi_10m.npz"
    if path.exists():
        with np.load(path) as saved:
            return {name: saved[name] for name in saved.files}
    scene = s2_scene()
    arrays = {}
    for name in ("B04", "B08", "B11", "SCL"):
        href = planetary_computer.sign(scene["assets"][name]["href"])
        resampling = Resampling.nearest if name == "SCL" else Resampling.bilinear
        with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=1):
            with rasterio.open(href) as source:
                with WarpedVRT(source, crs="EPSG:32647", transform=grid["transform"],
                               width=grid["width"], height=grid["height"],
                               resampling=resampling) as warped:
                    arrays[name] = warped.read(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def cell_box(grid: dict, row: int, col: int) -> Polygon:
    left = grid["west"] + col * GRID_M
    top = grid["north"] - row * GRID_M
    return box(left, top - GRID_M, left + GRID_M, top)


class SurfaceOSM(osmium.SimpleHandler):
    """Summarize January 2025 OSM ways/nodes into all 250 m grid cells."""

    def __init__(self, grid: dict):
        super().__init__()
        self.grid = grid
        shape = (grid["rows"], grid["cols"])
        self.road_m = np.zeros(shape, dtype=np.float32)
        self.waterway_m = np.zeros(shape, dtype=np.float32)
        self.building_area_m2 = np.zeros(shape, dtype=np.float32)
        self.amenity_count = np.zeros(shape, dtype=np.int32)
        self.essential_count = np.zeros(shape, dtype=np.int32)
        self.features_seen = Counter()
        self.lon_min, self.lat_min = TO_LONLAT.transform(grid["west"], grid["south"])
        self.lon_max, self.lat_max = TO_LONLAT.transform(grid["east"], grid["north"])

    def _index(self, x: float, y: float) -> tuple[int, int] | None:
        col = math.floor((x - self.grid["west"]) / GRID_M)
        row = math.floor((self.grid["north"] - y) / GRID_M)
        if 0 <= row < self.grid["rows"] and 0 <= col < self.grid["cols"]:
            return row, col
        return None

    def _add_amenity(self, x: float, y: float, tag: str) -> None:
        index = self._index(x, y)
        if index is None:
            return
        self.amenity_count[index] += 1
        if tag in ESSENTIAL_AMENITIES:
            self.essential_count[index] += 1
        self.features_seen["amenities"] += 1

    def _add_geometry(self, geometry, output: np.ndarray, measure: str) -> None:
        left, bottom, right, top = geometry.bounds
        col0 = max(0, math.floor((left - self.grid["west"]) / GRID_M))
        col1 = min(self.grid["cols"] - 1, math.floor((right - self.grid["west"]) / GRID_M))
        row0 = max(0, math.floor((self.grid["north"] - top) / GRID_M))
        row1 = min(self.grid["rows"] - 1, math.floor((self.grid["north"] - bottom) / GRID_M))
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                part = geometry.intersection(cell_box(self.grid, row, col))
                output[row, col] += part.length if measure == "length" else part.area

    def node(self, node) -> None:
        amenity = node.tags.get("amenity")
        if not amenity or not node.location.valid():
            return
        if not (self.lon_min <= node.lon <= self.lon_max
                and self.lat_min <= node.lat <= self.lat_max):
            return
        self._add_amenity(*TO_UTM.transform(node.lon, node.lat), amenity)

    def way(self, way) -> None:
        highway = way.tags.get("highway")
        waterway = way.tags.get("waterway")
        building = way.tags.get("building")
        amenity = way.tags.get("amenity")
        if highway not in ROAD_CLASSES and not waterway and not building and not amenity:
            return
        nodes = [(node.lon, node.lat) for node in way.nodes if node.location.valid()]
        if len(nodes) < 2:
            return
        if (max(p[0] for p in nodes) < self.lon_min
                or min(p[0] for p in nodes) > self.lon_max
                or max(p[1] for p in nodes) < self.lat_min
                or min(p[1] for p in nodes) > self.lat_max):
            return
        xs, ys = TO_UTM.transform(*zip(*nodes))
        coords = list(zip(xs, ys))
        if highway in ROAD_CLASSES:
            self._add_geometry(LineString(coords), self.road_m, "length")
            self.features_seen["roads"] += 1
        if waterway:
            self._add_geometry(LineString(coords), self.waterway_m, "length")
            self.features_seen["waterways"] += 1
        if building and building != "no" and len(coords) >= 4 and coords[0] == coords[-1]:
            polygon = Polygon(coords)
            if polygon.is_valid and polygon.area > 0:
                self._add_geometry(polygon, self.building_area_m2, "area")
                self.features_seen["buildings"] += 1
        if amenity:
            centre = LineString(coords).centroid
            self._add_amenity(centre.x, centre.y, amenity)


def osm_arrays(grid: dict) -> tuple[dict[str, np.ndarray], dict]:
    path = CACHE / "osm_surface_features_250m.npz"
    audit_path = CACHE / "osm_surface_features_250m_audit.json"
    if path.exists() and audit_path.exists():
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        if all(array.shape == (grid["rows"], grid["cols"]) for array in arrays.values()):
            return arrays, json.loads(audit_path.read_text())
    handler = SurfaceOSM(grid)
    handler.apply_file(str(OSM_PBF), locations=True)
    arrays = {"road_m": handler.road_m, "waterway_m": handler.waterway_m,
              "building_area_m2": handler.building_area_m2,
              "amenity_count": handler.amenity_count,
              "essential_amenity_count": handler.essential_count}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    audit = {"osm_pbf": OSM_PBF.name, "features_seen_in_grid_bbox": dict(handler.features_seen)}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return arrays, audit


def pump_points() -> list[dict]:
    raw = fetch(CACHE / "bma_pump_gate_stations.csv", PUMP_URL)
    stations = []
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        try:
            lon, lat = float(row["gp_long"]), float(row["gp_lat"])
        except ValueError:
            continue
        if not (99 < lon < 102 and 12 < lat < 15):
            continue
        x, y = TO_UTM.transform(lon, lat)
        stations.append({"x": x, "y": y, "name": row["gp_name"],
                         "type": row["gp_type"]})
    return stations


def traffy_cell_reports(grid: dict, polygon: Polygon) -> tuple[dict[tuple[int, int], dict], dict]:
    """Positive complaint locations; missing complaints never imply dry cells."""
    if not TRAFFY.exists():
        raise FileNotFoundError("Run flood_bangkapi_traffy_extract.py first")
    # Requests for pump work or preventive sandbags need not mean water was
    # present at the complaint timestamp, so keep the more direct categories.
    direct_types = {"น้ำท่วม", "BEST -> น้ำท่วมขัง",
                    "น้ำท่วม -> ขอให้แก้ไขปัญหาน้ำท่วม",
                    "น้ำท่วม -> ขอให้สูบน้ำออก"}
    result: dict[tuple[int, int], dict] = {}
    audit = Counter()
    with TRAFFY.open(encoding="utf-8-sig", newline="") as file:
        for report in csv.DictReader(file):
            if report["type"] not in direct_types:
                continue
            x, y = TO_UTM.transform(float(report["lon"]), float(report["lat"]))
            if not polygon.covers(Point(x, y)):
                continue
            col = math.floor((x - grid["west"]) / GRID_M)
            row = math.floor((grid["north"] - y) / GRID_M)
            if not (0 <= row < grid["rows"] and 0 <= col < grid["cols"]):
                continue
            day = report["timestamp"][:10]
            cell = result.setdefault((row, col), {"reports": 0, "dates": set(),
                                                   "may_aug": set(), "sep_dec": set()})
            cell["reports"] += 1
            cell["dates"].add(day)
            if "2025-05-01" <= day <= "2025-08-31":
                cell["may_aug"].add(day)
            elif "2025-09-01" <= day <= "2025-12-31":
                cell["sep_dec"].add(day)
            audit["reports"] += 1
    audit["cells_with_report"] = len(result)
    audit["cell_report_days"] = sum(len(cell["dates"]) for cell in result.values())
    return result, dict(audit)


def traffy_activity() -> dict[tuple[int, int], Counter]:
    if not TRAFFY_ACTIVITY.exists():
        raise FileNotFoundError("Run flood_bangkapi_traffy_extract.py to get all-complaint activity")
    by_cell: dict[tuple[int, int], Counter] = {}
    with TRAFFY_ACTIVITY.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            index = int(row["grid_row"]), int(row["grid_col"])
            by_cell.setdefault(index, Counter())[row["month"]] += int(row["all_complaints"])
    return by_cell


def distance_features(x: float, y: float, stations: list[dict]) -> dict:
    pumps = [math.hypot(s["x"] - x, s["y"] - y) for s in stations
             if "สูบ" in s["type"] or "สูบ" in s["name"]]
    gates = [math.hypot(s["x"] - x, s["y"] - y) for s in stations
             if "ประตู" in s["type"] or "ประตู" in s["name"]]
    return {"nearest_pump_m": round(min(pumps), 1) if pumps else None,
            "pumps_within_1km": sum(distance <= 1000 for distance in pumps),
            "nearest_gate_m": round(min(gates), 1) if gates else None}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report_anchors() -> None:
    features = []
    with ANCHORS.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            props = {key: row[key] for key in ("site", "label", "report_days_2021_2024",
                                                "report_days_2025", "inventory_points")}
            features.append({"type": "Feature", "geometry": mapping(Point(
                float(row["lon"]), float(row["lat"]))), "properties": props})
    (OUTPUT / "bma_report_anchors.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False))


def write_feature_map(grid_geojson: dict) -> None:
    anchors = json.loads((OUTPUT / "bma_report_anchors.geojson").read_text())
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bang Kapi surface feature grid</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
body{margin:0;font:14px system-ui,sans-serif;color:#192a38;background:#f4f7f9}
header{padding:12px 18px;background:#12344b;color:white;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
header strong{font-size:17px}header small{opacity:.88}select{font:inherit;padding:5px;background:white;border:0;border-radius:4px}
#map{height:calc(100vh - 74px);min-height:480px}.legend{background:white;padding:8px 10px;line-height:1.5;box-shadow:0 1px 8px #5555;border-radius:5px;max-width:250px}
.legend .bar{height:12px;background:linear-gradient(90deg,#d5e6fa,#377fb6,#083b65);margin:5px 0}
</style></head><body>
<header><strong>Bang Kapi surface cells · 250 m</strong><label>Show <select id="layer">
<option value="elev_median_m">Ground elevation (m)</option>
<option value="s2_ndvi_median">Pre-storm vegetation index</option>
<option value="s2_low_veg_land_fraction">Low-vegetation land fraction</option>
<option value="osm_road_m">OSM road length (m)</option>
<option value="nearest_pump_m">Distance to nearest pump (m)</option>
<option value="traffy_2025_flood_reports">Observed flood complaints (2025)</option>
<option value="traffy_all_activity_jan_aug">All complaint activity (Jan–Aug 2025)</option>
</select></label><small>Inputs across all surface cells. Flood probability is not calibrated.</small></header>
<div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const grid=__GRID_DATA__,anchors=__ANCHORS_DATA__;
const map=L.map('map',{preferCanvas:true});
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
let active='elev_median_m';
const ranges={elev_median_m:[-0.6,1.7],s2_ndvi_median:[0,0.5],s2_low_veg_land_fraction:[0,1],osm_road_m:[0,3000],nearest_pump_m:[0,3000],traffy_2025_flood_reports:[0,12],traffy_all_activity_jan_aug:[0,50]};
function shade(feature){const value=Number(feature.properties[active]);if(!Number.isFinite(value))return '#ddd';const [lo,hi]=ranges[active];const t=Math.max(0,Math.min(1,(value-lo)/(hi-lo)));return `hsl(207,${43+23*t}%,${87-60*t}%)`;}
const cells=L.geoJSON(grid,{style:f=>({color:'#35566e',weight:.25,fillColor:shade(f),fillOpacity:.75}),onEachFeature:(f,l)=>{
 const p=f.properties;
 l.bindPopup(`<b>${p.cell_id}</b><br>Elevation: ${p.elev_median_m} m<br>Vegetation index: ${p.s2_ndvi_median}<br>Low-vegetation land: ${p.s2_low_veg_land_fraction}<br>Road length: ${p.osm_road_m} m<br>Buildings mapped: ${p.osm_building_footprint_fraction}<br>Amenities: ${p.osm_amenity_count}<br>Nearest pump: ${p.nearest_pump_m} m<br>2025 flood complaints: ${p.traffy_2025_flood_reports}<br>Prior all-complaint activity: ${p.traffy_all_activity_jan_aug}<br><b>Flood probability: unavailable</b>`);
}}).addTo(map);
const reportPoints=L.geoJSON(anchors,{pointToLayer:(f,latlng)=>L.circleMarker(latlng,{radius:5+Math.min(8,Number(f.properties.report_days_2025)),color:'#852226',fillColor:'#d54e45',fillOpacity:.9,weight:1.5}),onEachFeature:(f,l)=>l.bindPopup(`<b>${f.properties.label}</b><br>2025 BMA report days: ${f.properties.report_days_2025}<br>Point inventory published in 2026.`)}).addTo(map);
map.fitBounds(cells.getBounds());
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const div=L.DomUtil.create('div','legend');div.innerHTML='<b>Raw feature value</b><div class="bar"></div><span id="legend-range"></span><br><small>Red circles: BMA road-report locations</small>';return div};legend.addTo(map);
function refresh(){active=document.getElementById('layer').value;cells.setStyle(f=>({fillColor:shade(f)}));const [lo,hi]=ranges[active];document.getElementById('legend-range').textContent=`${lo} → ${hi}`;}
document.getElementById('layer').addEventListener('change',refresh);refresh();
</script></body></html>"""
    html = html.replace("__GRID_DATA__", json.dumps(grid_geojson, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__ANCHORS_DATA__", json.dumps(anchors, ensure_ascii=False).replace("</", "<\\/"))
    (OUTPUT / "surface_feature_map.html").write_text(html)


def main() -> None:
    polygon = district_polygon()
    grid = grid_spec(polygon)
    terrain = get_dtm(grid)
    s2 = satellite_arrays(grid)
    osm, osm_audit = osm_arrays(grid)
    stations = pump_points()
    complaint_cells, complaint_audit = traffy_cell_reports(grid, polygon)
    activity = traffy_activity()
    inside = geometry_mask([polygon.__geo_interface__], terrain.shape,
                           grid["transform"], invert=True)
    b04 = s2["B04"].astype(float)
    b08 = s2["B08"].astype(float)
    b11 = s2["B11"].astype(float)
    scl = s2["SCL"]
    clear = np.isin(scl, [4, 5, 6]) & (b04 > 0) & (b08 > 0) & (b11 > 0)
    land = np.isin(scl, [4, 5]) & clear
    ndvi = (b08 - b04) / np.maximum(1, b08 + b04)
    ndbi = (b11 - b08) / np.maximum(1, b11 + b08)
    # Neighbourhood mean is a surface-context feature, not a drainage model.
    neighborhood_elev = uniform_filter(terrain, size=51, mode="nearest")
    pixels = GRID_M // PIXEL_M
    rows, features = [], []
    for row in range(grid["rows"]):
        for col in range(grid["cols"]):
            rr = slice(row * pixels, (row + 1) * pixels)
            cc = slice(col * pixels, (col + 1) * pixels)
            mask = inside[rr, cc]
            fraction = float(np.mean(mask))
            if not np.any(mask):
                continue
            valid = clear[rr, cc] & mask
            valid_land = land[rr, cc] & mask
            elevations = terrain[rr, cc][mask]
            median = float(np.median(elevations))
            box_utm = cell_box(grid, row, col)
            centre = box_utm.centroid
            lon, lat = TO_LONLAT.transform(centre.x, centre.y)
            complaints = complaint_cells.get((row, col), {"reports": 0, "dates": set(),
                                                      "may_aug": set(), "sep_dec": set()})
            activity_months = activity.get((row, col), Counter())
            item = {"cell_id": f"bk:{row:02d}:{col:02d}",
                    "centre_lon": round(lon, 6), "centre_lat": round(lat, 6),
                    "district_coverage_fraction": round(fraction, 3),
                    "boundary_sliver": int(fraction < 0.25),
                    "elev_p10_m": round(float(np.percentile(elevations, 10)), 3),
                    "elev_median_m": round(median, 3),
                    "relative_to_500m_mean_m": round(
                        median - float(np.mean(neighborhood_elev[rr, cc][mask])), 3),
                    "s2_clear_fraction": round(float(np.sum(valid) / np.sum(mask)), 3),
                    "s2_ndvi_median": round(float(np.median(ndvi[rr, cc][valid])), 3)
                    if np.any(valid) else None,
                    "s2_ndbi_median": round(float(np.median(ndbi[rr, cc][valid])), 3)
                    if np.any(valid) else None,
                    "s2_low_veg_land_fraction": round(float(np.mean(
                        ndvi[rr, cc][valid_land] < 0.25)), 3)
                    if np.any(valid_land) else None,
                    "osm_road_m": round(float(osm["road_m"][row, col]), 1),
                    "osm_waterway_m": round(float(osm["waterway_m"][row, col]), 1),
                    "osm_building_footprint_fraction": round(min(1.0,
                        float(osm["building_area_m2"][row, col]) / (GRID_M ** 2)), 3),
                    "osm_amenity_count": int(osm["amenity_count"][row, col]),
                    "osm_essential_amenity_count": int(osm["essential_amenity_count"][row, col]),
                    **distance_features(centre.x, centre.y, stations),
                    "traffy_2025_flood_reports": complaints["reports"],
                    "traffy_2025_report_days": len(complaints["dates"]),
                    "traffy_may_aug_report_days": len(complaints["may_aug"]),
                    "traffy_sep_dec_report_days": len(complaints["sep_dec"]),
                    "traffy_all_activity_jan_aug": sum(
                        count for month, count in activity_months.items() if month <= "2025-08"),
                    "traffy_all_activity_may_aug": sum(
                        count for month, count in activity_months.items()
                        if "2025-05" <= month <= "2025-08"),
                    "traffy_all_activity_sep_dec": sum(
                        count for month, count in activity_months.items() if month >= "2025-09"),
                    "flood_probability": None,
                    "outcome_status": "flood_complaint_observed" if complaints["reports"]
                    else "unobserved"}
            rows.append(item)
            geographic = shapely_transform(TO_LONLAT.transform, box_utm)
            features.append({"type": "Feature", "geometry": mapping(geographic),
                             "properties": item})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "surface_cells_250m.csv", rows)
    grid_geojson = {"type": "FeatureCollection", "features": features}
    (OUTPUT / "surface_cells_250m.geojson").write_text(json.dumps(
        grid_geojson, ensure_ascii=False))
    write_report_anchors()
    write_feature_map(grid_geojson)
    audit = {"unit": "250 m surface cell, road and off-road",
             "grid_cell_count": len(rows),
             "district_area_km2": round(polygon.area / 1e6, 2),
             "terrain_resolution_m": PIXEL_M,
             "satellite_scene": S2_SCENE_PREFIX,
             "satellite_pre_event_date": "2025-01-15",
             "clear_fraction_median": round(float(np.median(
                 [r["s2_clear_fraction"] for r in rows])), 3),
             "cells_with_road": sum(r["osm_road_m"] > 0 for r in rows),
             "cells_without_road": sum(r["osm_road_m"] == 0 for r in rows),
             "cells_with_amenity": sum(r["osm_amenity_count"] > 0 for r in rows),
             "pump_gate_station_rows": len(stations),
             "traffy": complaint_audit,
             "traffy_all_category_prior_activity": sum(
                 r["traffy_all_activity_jan_aug"] for r in rows),
             "osm": osm_audit,
             "probability_status": "unset: Traffy supplies spatial positives, but unreported cells are not verified dry and reporting propensity is unknown"}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
