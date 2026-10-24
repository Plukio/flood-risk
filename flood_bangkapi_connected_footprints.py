"""Build event-linked, LiDAR-constrained flood-report proxy footprints.

Reports connect when within 3 hours and 300 m and a local 10 m LiDAR
low-elevation path joins their points. A connected group's footprint is its
seed-connected low terrain inside the report convex hull plus 100 m. Groups
without a spatial spread of 50 m use a 100 m point-buffer fallback.

These are reproducible *report-derived proxies*, not observed flood edges.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
import shapely
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window
from scipy.ndimage import label
from shapely.geometry import LineString, MultiPoint, Point, mapping, shape
from shapely.ops import unary_union

from flood_bangkapi_surface_grid import GRID_M, TO_LONLAT, TO_UTM, cell_box, district_polygon, grid_spec
from flood_bangkapi_surface_model import DIRECT_TYPES, OUTPUT, TRAFFY


ROOT = Path(__file__).resolve().parent
DTM = ROOT / "data" / "flood_bangkapi_surface" / "dtm_bangkapi_10m.tif"
MAX_LINK_TIME_H = 3
MAX_LINK_DISTANCE_M = 300
MAX_CLUSTER_SPAN_H = 6
MAX_SEED_ELEV_DIFF_M = 0.75
MAX_SADDLE_RISE_M = 0.25
CORRIDOR_M = 75
FALLBACK_RADIUS_M = 100
MIN_CLUSTER_SPREAD_M = 50


def reports(district, dataset, source_path: Path = TRAFFY) -> list[dict]:
    result = []
    for item in csv.DictReader(source_path.open(encoding="utf-8-sig", newline="")):
        if item["type"] not in DIRECT_TYPES:
            continue
        x, y = TO_UTM.transform(float(item["lon"]), float(item["lat"]))
        if not district.covers(Point(x, y)):
            continue
        row, col = dataset.index(x, y)
        result.append({"time": datetime.fromisoformat(item["timestamp"]),
                       "day": date.fromisoformat(item["timestamp"][:10]),
                       "x": x, "y": y, "row": row, "col": col,
                       "elevation": float(dataset.read(1, window=Window(col, row, 1, 1))[0, 0])})
    return result


def corridor_connected(a: dict, b: dict, terrain: np.ndarray, dataset) -> bool:
    if abs(a["elevation"] - b["elevation"]) > MAX_SEED_ELEV_DIFF_M:
        return False
    pad = math.ceil(CORRIDOR_M / 10) + 2
    r0 = max(0, min(a["row"], b["row"]) - pad)
    r1 = min(terrain.shape[0] - 1, max(a["row"], b["row"]) + pad)
    c0 = max(0, min(a["col"], b["col"]) - pad)
    c1 = min(terrain.shape[1] - 1, max(a["col"], b["col"]) + pad)
    window = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    subset = terrain[r0:r1 + 1, c0:c1 + 1]
    corridor = LineString([(a["x"], a["y"]), (b["x"], b["y"])]).buffer(CORRIDOR_M)
    allowed = geometry_mask([mapping(corridor)], subset.shape,
                            dataset.window_transform(window), invert=True, all_touched=True)
    threshold = max(a["elevation"], b["elevation"]) + MAX_SADDLE_RISE_M
    components, _ = label(allowed & np.isfinite(subset) & (subset <= threshold),
                          structure=np.ones((3, 3), dtype=np.uint8))
    first = components[a["row"] - r0, a["col"] - c0]
    second = components[b["row"] - r0, b["col"] - c0]
    return bool(first and first == second)


def components(reports_: list[dict], terrain: np.ndarray, dataset) -> tuple[list[list[int]], dict]:
    n = len(reports_)
    parent = list(range(n))
    earliest = [item["time"] for item in reports_]
    latest = earliest.copy()

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = []
    for i, a in enumerate(reports_):
        for j in range(i + 1, n):
            b = reports_[j]
            hours = abs((a["time"] - b["time"]).total_seconds()) / 3600
            distance = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if hours <= MAX_LINK_TIME_H and distance <= MAX_LINK_DISTANCE_M:
                candidates.append((hours, distance, i, j))
    candidates.sort()
    linked = 0
    for _, _, i, j in candidates:
        a, b = find(i), find(j)
        if a == b:
            continue
        start, end = min(earliest[a], earliest[b]), max(latest[a], latest[b])
        if (end - start).total_seconds() > MAX_CLUSTER_SPAN_H * 3600:
            continue
        if not corridor_connected(reports_[i], reports_[j], terrain, dataset):
            continue
        parent[b] = a
        earliest[a], latest[a] = start, end
        linked += 1
    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        grouped[find(i)].append(i)
    return list(grouped.values()), {"candidate_pairs": len(candidates),
                                    "accepted_union_links": linked}


def terrain_footprint(group: list[dict], terrain: np.ndarray, dataset, district) -> object | None:
    points = [Point(item["x"], item["y"]) for item in group]
    hull = MultiPoint(points).convex_hull.buffer(FALLBACK_RADIUS_M)
    pad = math.ceil(FALLBACK_RADIUS_M / 10) + 2
    r0 = max(0, min(item["row"] for item in group) - pad)
    r1 = min(terrain.shape[0] - 1, max(item["row"] for item in group) + pad)
    c0 = max(0, min(item["col"] for item in group) - pad)
    c1 = min(terrain.shape[1] - 1, max(item["col"] for item in group) + pad)
    window = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    subset = terrain[r0:r1 + 1, c0:c1 + 1]
    candidate = geometry_mask([mapping(hull)], subset.shape,
                              dataset.window_transform(window), invert=True, all_touched=True)
    threshold = max(item["elevation"] for item in group) + MAX_SADDLE_RISE_M
    components_, _ = label(candidate & np.isfinite(subset) & (subset <= threshold),
                           structure=np.ones((3, 3), dtype=np.uint8))
    seed_ids = {int(components_[item["row"] - r0, item["col"] - c0]) for item in group}
    if len(seed_ids) != 1 or 0 in seed_ids:
        return None
    selected = components_ == seed_ids.pop()
    polygon_parts = [shape(geom) for geom, value in shapes(selected.astype(np.uint8),
                     mask=selected, transform=dataset.window_transform(window)) if value == 1]
    if not polygon_parts:
        return None
    geometry = unary_union(polygon_parts).intersection(hull).intersection(district)
    return geometry if not geometry.is_empty else None


def write_footprint_map(features: list[dict], filename: str = "connected_report_footprint_map.html") -> None:
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bang Kapi connected flood-report footprints</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,sans-serif}header{padding:12px 16px;background:#173b54;color:white}header strong{font-size:17px}header small{display:block;margin-top:4px}#map{height:calc(100vh - 75px);min-height:480px}.legend{padding:10px;background:white;box-shadow:0 1px 8px #5555;border-radius:5px;line-height:1.5}</style></head><body><header><strong>Bang Kapi · inferred report footprints</strong><small>Blue: nearby, similar-time reports connected through low LiDAR terrain. Orange: 100 m fallback. These are proxy labels, not observed water edges.</small></header><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const data=__DATA__;const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
const layer=L.geoJSON(data,{style:f=>({color:f.properties.method==='connected_low_terrain'?'#1d5d90':'#ad642d',weight:1,fillColor:f.properties.method==='connected_low_terrain'?'#54a0c8':'#efa363',fillOpacity:.43}),onEachFeature:(f,l)=>{const p=f.properties;l.bindPopup(`<b>${p.method==='connected_low_terrain'?'Terrain-connected':'100 m fallback'}</b><br>Reports: ${p.report_count}<br>Report dates: ${p.report_dates.join(', ')}<br>Area: ${Math.round(p.area_m2).toLocaleString()} m²<br>Terrain ceiling: ${p.terrain_threshold_m===null?'not used':p.terrain_threshold_m+' m'}`);}}).addTo(map);map.fitBounds(layer.getBounds());
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Proxy footprint method</b><br><span style="color:#1d5d90">■</span> Connected low terrain<br><span style="color:#ad642d">■</span> 100 m report buffer';return d};legend.addTo(map);
</script></body></html>"""
    html = html.replace("__DATA__", json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False).replace("</", "<\\/"))
    (OUTPUT / filename).write_text(html)


def build(source_path: Path = TRAFFY, year_tag: str = "2025") -> tuple[dict[str, set[date]], dict]:
    district = district_polygon()
    spec = grid_spec(district)
    grid = json.loads((OUTPUT / "surface_cells_250m.geojson").read_text())
    ids = {item["properties"]["cell_id"] for item in grid["features"]}
    cell_shapes = {}
    for cid in ids:
        _, row, col = cid.split(":")
        cell_shapes[cid] = cell_box(spec, int(row), int(col)).intersection(district)
    with rasterio.open(DTM) as dataset:
        if dataset.crs.to_epsg() != 32647 or abs(dataset.res[0] - 10) > 0.01:
            raise ValueError("Unexpected BMA LiDAR grid")
        terrain = dataset.read(1)
        source = reports(district, dataset, source_path)
        groups, graph_audit = components(source, terrain, dataset)
        features = []
        labels: dict[str, set[date]] = defaultdict(set)
        method_counts = Counter()
        method_report_counts = Counter()
        fallback_reasons = Counter()
        cluster_sizes = Counter()
        areas = []
        spans = []
        for cluster_no, indices in enumerate(groups, 1):
            group = [source[i] for i in indices]
            cluster_sizes[len(group)] += 1
            spans.append((max(item["time"] for item in group) -
                          min(item["time"] for item in group)).total_seconds() / 3600)
            spread = max((math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                          for a in group for b in group), default=0.0)
            footprint = None
            fallback_reason = "singleton" if len(group) < 2 else (
                "under_50m_spread" if spread < MIN_CLUSTER_SPREAD_M else "terrain_mask_disconnected")
            if len(group) >= 2 and spread >= MIN_CLUSTER_SPREAD_M:
                footprint = terrain_footprint(group, terrain, dataset, district)
            if footprint is None:
                method = "100m_point_buffer_fallback"
                fallback_reasons[fallback_reason] += 1
                footprint = unary_union([Point(item["x"], item["y"]).buffer(
                    FALLBACK_RADIUS_M) for item in group]).intersection(district)
            else:
                method = "connected_low_terrain"
            method_counts[method] += 1
            method_report_counts[method] += len(group)
            areas.append(float(footprint.area))
            dates = sorted({item["day"] for item in group})
            # Limit intersection tests to grid cells near the footprint.
            minx, miny, maxx, maxy = footprint.bounds
            left = max(0, math.floor((minx - spec["west"]) / GRID_M))
            right = min(spec["cols"] - 1, math.floor((maxx - spec["west"]) / GRID_M))
            top = max(0, math.floor((spec["north"] - maxy) / GRID_M))
            bottom = min(spec["rows"] - 1, math.floor((spec["north"] - miny) / GRID_M))
            for rr in range(top, bottom + 1):
                for cc in range(left, right + 1):
                    cid = f"bk:{rr:02d}:{cc:02d}"
                    if cid in cell_shapes and cell_shapes[cid].intersects(footprint):
                        labels[cid].update(dates)
            geographic = shapely.transform(footprint, TO_LONLAT.transform, interleaved=False)
            features.append({"type": "Feature", "geometry": mapping(geographic),
                             "properties": {"cluster_id": cluster_no, "method": method,
                                            "report_count": len(group),
                                            "start_local": min(item["time"] for item in group).isoformat(),
                                            "end_local": max(item["time"] for item in group).isoformat(),
                                            "report_dates": [str(day) for day in dates],
                                            "area_m2": round(float(footprint.area), 1),
                                            "terrain_threshold_m": round(max(
                                                item["elevation"] for item in group) + MAX_SADDLE_RISE_M, 3)
                                            if method == "connected_low_terrain" else None}})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"connected_report_footprints_{year_tag}.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False))
    write_footprint_map(features, "connected_report_footprint_map.html" if year_tag == "2025"
                        else f"connected_report_footprint_map_{year_tag}.html")
    audit = {"source": f"BMA Traffy Fondue {year_tag} direct-category geocoded flood reports",
             "reports": len(source), "clusters": len(groups),
             "time_window_h": MAX_LINK_TIME_H, "max_link_distance_m": MAX_LINK_DISTANCE_M,
             "max_cluster_span_h": MAX_CLUSTER_SPAN_H,
             "max_seed_elevation_difference_m": MAX_SEED_ELEV_DIFF_M,
             "max_saddle_rise_above_higher_seed_m": MAX_SADDLE_RISE_M,
             "path_corridor_width_m": CORRIDOR_M * 2,
             "footprint_hull_margin_m": FALLBACK_RADIUS_M,
             "fallback_radius_m": FALLBACK_RADIUS_M,
             "minimum_spread_for_terrain_footprint_m": MIN_CLUSTER_SPREAD_M,
             **graph_audit, "cluster_method_counts": dict(method_counts),
             "report_method_counts": dict(method_report_counts),
             "fallback_reason_counts": dict(fallback_reasons),
             "cluster_size_counts": dict(cluster_sizes),
             "max_cluster_span_observed_h": round(max(spans), 2),
             "footprint_area_m2_median": round(float(np.median(areas)), 1),
             "footprint_area_m2_max": round(float(max(areas)), 1),
             f"positive_cells_any_{year_tag}": sum(bool(days) for days in labels.values()),
             f"positive_cell_days_{year_tag}": sum(len(days) for days in labels.values()),
             "interpretation": "terrain-constrained report proxy, not verified flood extent"}
    audit_path = (OUTPUT / "connected_report_footprints_audit.json" if year_tag == "2025"
                  else OUTPUT / f"connected_report_footprints_audit_{year_tag}.json")
    audit_path.write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")
    return labels, audit


if __name__ == "__main__":
    _, result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
