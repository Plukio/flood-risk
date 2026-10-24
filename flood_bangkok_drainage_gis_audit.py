"""Audit BMA's public ArcGIS drainage network around Bang Kapi.

Outputs a source-geometry snapshot, a proximity-only canal-connection layer,
an audit JSON, and a Leaflet inspection map. None of the inferred points is a
verified operating outfall.

Run: .venv/bin/python flood_bangkok_drainage_gis_audit.py
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from flood_multi_target_citywide_data import OUT

SERVICE = "https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer"
PIPE_WHERE = "DISTRICT_I='1006'"  # Bang Kapi
AUDIT = OUT / "bangkapi_drainage_gis_2021_audit.json"
SOURCE_GEOJSON = OUT / "bangkapi_drainage_gis_2021_source.geojson"
CONNECTION_GEOJSON = OUT / "bangkapi_candidate_pipe_canal_connections.geojson"
MAP_HTML = OUT / "bangkapi_drainage_gis_2021_map.html"
TO_LONLAT = Transformer.from_crs(32647, 4326, always_xy=True)
FIELDS = {
    5: "OBJECTID_1,DISTRICT_I,NAME,F_ROAD,TO_ROAD,F_EL_BOTT,T_EL_BOTT,"
       "DIMENSION,LENGTH,PIPE_TYPE,SOURCE,YEAR",
    6: "OBJECTID,KHONG_NAME,WIDTH_M,DEEP_M",
    7: "OBJECTID,ชื่ออาคารบังคับน้ำ,ชนิด,POWER",
    1: "OBJECTID,NAME,TYPE_GATE,NO_GATE,SIZE_",
    3: "OBJECTID,NAME",
}
OID = {5: "OBJECTID_1", 6: "OBJECTID", 7: "OBJECTID",
       1: "OBJECTID", 3: "OBJECTID"}


def query(session: requests.Session, layer: int, where: str,
          envelope: str | None = None) -> list[dict]:
    result = []
    offset = 0
    while True:
        params = {"f": "json", "where": where, "outFields": FIELDS[layer],
                  "returnGeometry": "true", "outSR": "32647",
                  "resultOffset": offset, "resultRecordCount": 1000,
                  "orderByFields": OID[layer]}
        if envelope:
            params.update(geometry=envelope,
                          geometryType="esriGeometryEnvelope", inSR=32647,
                          spatialRel="esriSpatialRelIntersects")
        response = session.get(f"{SERVICE}/{layer}/query", params=params,
                               timeout=35)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"BMA layer {layer}: {data['error']}")
        page = data.get("features", [])
        result.extend(page)
        if len(page) < 1000:
            break
        offset += len(page)
    return result


def count(session: requests.Session, layer: int, where: str = "1=1") -> int:
    response = session.get(f"{SERVICE}/{layer}/query", params={
        "f": "json", "where": where, "returnCountOnly": "true"}, timeout=25)
    response.raise_for_status()
    data = response.json()
    if "error" in data or "count" not in data:
        raise ValueError(f"BMA layer {layer} count: {data}")
    return int(data["count"])


def name(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"[^\wก-๙]", "", value)


def number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def wgs_point(xy) -> list[float]:
    lon, lat = TO_LONLAT.transform(float(xy[0]), float(xy[1]))
    return [round(lon, 7), round(lat, 7)]


def source_feature(row: dict, kind: str) -> dict:
    geom = row["geometry"]
    if "paths" in geom:
        paths = [[wgs_point(xy) for xy in path] for path in geom["paths"]]
        shape = ({"type": "LineString", "coordinates": paths[0]} if
                 len(paths) == 1 else
                 {"type": "MultiLineString", "coordinates": paths})
    else:
        shape = {"type": "Point", "coordinates": wgs_point((geom["x"], geom["y"]))}
    return {"type": "Feature", "geometry": shape,
            "properties": {"kind": kind, **row["attributes"]}}


def nearest_other_pipe(endpoints: np.ndarray) -> np.ndarray:
    owner = np.repeat(np.arange(len(endpoints)//2), 2)
    distance, index = cKDTree(endpoints).query(
        endpoints, k=min(16, len(endpoints)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    return np.asarray([
        next((float(d) for d, j in zip(ds, js) if owner[j] != owner[i]),
             math.inf)
        for i, (ds, js) in enumerate(zip(distance, index))])


def components(points: np.ndarray, radius: float) -> list[list[int]]:
    if not len(points):
        return []
    parent = list(range(len(points)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(points).query_pairs(radius):
        pa, pb = root(a), root(b)
        if pa != pb:
            parent[pb] = pa
    groups = {}
    for i in range(len(points)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def build():
    session = requests.Session()
    root = session.get(SERVICE, params={"f": "json"}, timeout=25).json()
    if "error" in root or "Query" not in root.get("capabilities", ""):
        raise ValueError("BMA drainage service is not publicly queryable")
    layer_counts = {str(layer): count(session, layer)
                    for layer in (0, 1, 2, 3, 5, 6, 7, 9)}
    citywide_both_invert = count(
        session, 5, "F_EL_BOTT <> '-' AND T_EL_BOTT <> '-'")
    citywide_2021 = count(session, 5, "YEAR='2564'")
    pipes = query(session, 5, PIPE_WHERE)
    extent = session.get(f"{SERVICE}/5/query", params={
        "f": "json", "where": PIPE_WHERE, "returnExtentOnly": "true"},
        timeout=25).json()["extent"]
    envelope = ",".join(str(extent[key]+delta) for key, delta in
                        (("xmin", -500), ("ymin", -500),
                         ("xmax", 500), ("ymax", 500)))
    canals = query(session, 6, "1=1", envelope)
    machines = query(session, 7, "1=1", envelope)
    gates = query(session, 1, "1=1", envelope)
    tunnels = query(session, 3, "1=1", envelope)
    if len(pipes) != 2454 or not canals:
        raise ValueError("Bang Kapi source coverage changed; review the audit")

    canal_lines = [LineString(path) for feature in canals
                   for path in feature["geometry"].get("paths", [])
                   if len(path) >= 2]
    canal_names = [name(feature["attributes"].get("KHONG_NAME"))
                   for feature in canals
                   for path in feature["geometry"].get("paths", [])
                   if len(path) >= 2]
    canal_tree = STRtree(canal_lines)
    endpoints = np.asarray([
        [feature["geometry"]["paths"][0][0],
         feature["geometry"]["paths"][-1][-1]]
        for feature in pipes], dtype=float).reshape(-1, 2)
    other_distance = nearest_other_pipe(endpoints)
    both_invert, drops = 0, []
    named, near_named, exact_name, no_other, candidate = 0, 0, 0, 0, []
    for i, feature in enumerate(pipes):
        attrs = feature["attributes"]
        first, last = number(attrs.get("F_EL_BOTT")), number(attrs.get("T_EL_BOTT"))
        if math.isfinite(first) and math.isfinite(last):
            both_invert += 1
            drops.append(first-last)
        target = name(attrs.get("TO_ROAD"))
        if "คลอง" not in target and "แม่น้ำ" not in target:
            continue
        named += 1
        point = Point(endpoints[2*i+1])
        j = int(canal_tree.nearest(point))
        distance = float(point.distance(canal_lines[j]))
        if distance > 20:
            continue
        near_named += 1
        same_name = target == canal_names[j]
        if same_name:
            exact_name += 1
        if other_distance[2*i+1] > 5:
            no_other += 1
        if same_name:
            candidate.append({"pipe_id": attrs["OBJECTID_1"],
                              "xy": endpoints[2*i+1],
                              "to_canal": attrs.get("TO_ROAD"),
                              "canal": canals[j]["attributes"].get("KHONG_NAME"),
                              "distance_m": distance,
                              "other_pipe_m": float(other_distance[2*i+1]),
                              "pipe_type": attrs.get("PIPE_TYPE"),
                              "to_invert_m": last})

    clusters = components(np.asarray([c["xy"] for c in candidate]), 20.)
    connection_features = []
    for group in clusters:
        rows = [candidate[i] for i in group]
        representative = min(rows, key=lambda row: row["distance_m"])
        connection_features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": wgs_point(representative["xy"])},
            "properties": {
                "classification": "candidate pipe-to-canal connection; unverified outfall",
                "pipe_ends_in_cluster": len(rows),
                "pipe_ids": [int(r["pipe_id"]) for r in rows],
                "canal": representative["canal"],
                "closest_canal_m": round(min(r["distance_m"] for r in rows), 2),
                "pipe_type": representative["pipe_type"],
                "endpoint_has_no_other_pipe_within_5m": bool(
                    representative["other_pipe_m"] > 5),
                "to_invert_m": (round(representative["to_invert_m"], 3)
                                if math.isfinite(representative["to_invert_m"])
                                else None),
            },
        })

    arrays = [("pipe", pipes), ("canal", canals),
              ("pump_equipment", machines), ("gate", gates),
              ("tunnel", tunnels)]
    source_features = [source_feature(feature, kind)
                       for kind, rows in arrays for feature in rows]
    source = {"type": "FeatureCollection", "features": source_features}
    connections = {"type": "FeatureCollection", "features": connection_features}
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_GEOJSON.write_text(json.dumps(source, ensure_ascii=False)+"\n")
    CONNECTION_GEOJSON.write_text(json.dumps(connections, ensure_ascii=False)+"\n")

    drops = np.asarray(drops)
    audit = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service": SERVICE,
        "scope": "BMA pipe district code 1006 (Bang Kapi); other layers within pipe bbox +500m",
        "service_snapshot": "Pipe layer description says B.E. 2564 (2021), scale 1:4,000, WGS84 UTM zone 47N; service has no timeInfo",
        "citywide_layer_counts": layer_counts,
        "citywide_pipes_with_both_nonplaceholder_invert_fields": citywide_both_invert,
        "citywide_pipes_with_year_2564": citywide_2021,
        "pipe_records": len(pipes),
        "pipe_types": dict(Counter(f["attributes"].get("PIPE_TYPE") for f in pipes)),
        "pipe_sources": dict(Counter(f["attributes"].get("SOURCE") for f in pipes)),
        "pipes_with_both_numeric_end_inverts": both_invert,
        "numeric_invert_drop_from_minus_to_p05_m": round(float(np.quantile(drops, .05)), 3),
        "numeric_invert_drop_from_minus_to_median_m": round(float(np.median(drops)), 3),
        "numeric_invert_drop_from_minus_to_p95_m": round(float(np.quantile(drops, .95)), 3),
        "pipe_endpoints": len(endpoints),
        "endpoints_with_other_pipe_endpoint_within_5m": int((other_distance <= 5).sum()),
        "pipe_to_field_names_canal_or_river": named,
        "named_pipe_to_endpoint_within_20m_of_any_bma_canal": near_named,
        "near_endpoint_nearest_canal_name_matches_to_field": exact_name,
        "near_named_endpoints_without_other_pipe_within_5m": no_other,
        "candidate_connection_clusters_20m": len(clusters),
        "nearby_canal_lines": len(canals),
        "nearby_pump_equipment_points": len(machines),
        "nearby_gate_points": len(gates),
        "nearby_tunnel_lines": len(tunnels),
        "connection_rule": "TO_ROAD names canal/river; last polyline endpoint <=20m from BMA canal; nearest canal normalized name matches; cluster endpoints at 20m",
        "caveats": ["A candidate connection is not a surveyed or operating outfall",
                    "No street inlet/manhole point layer or verified pipe-to-pump/gate edge was found in this service",
                    "Pipe endpoints near each other may be unconnected or differ vertically",
                    "Digital lines come from map scans/CAD and may have positional error",
                    "No 2022-2025 pump/gate operation or time-matched outlet level is supplied"],
        "source_geojson": str(SOURCE_GEOJSON),
        "candidate_connections_geojson": str(CONNECTION_GEOJSON),
        "inspection_map": str(MAP_HTML),
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n")
    make_map(source, connections)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def make_map(source: dict, connections: dict):
    pipe = [f for f in source["features"] if f["properties"]["kind"] == "pipe"]
    canal = [f for f in source["features"] if f["properties"]["kind"] == "canal"]
    assets = [f for f in source["features"] if f["properties"]["kind"] not in
              ("pipe", "canal")]
    embedded = json.dumps({"pipes": {"type": "FeatureCollection", "features": pipe},
                           "canals": {"type": "FeatureCollection", "features": canal},
                           "assets": {"type": "FeatureCollection", "features": assets},
                           "connections": connections}, ensure_ascii=False)
    template = """<!doctype html><html><head><meta charset="utf-8"><title>Bang Kapi drainage GIS audit</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{margin:0;font:14px system-ui}#map{height:100vh}.note{position:absolute;z-index:1000;top:12px;left:55px;background:white;padding:12px;max-width:350px;box-shadow:0 1px 8px #777}.note b{display:block;margin-bottom:5px}</style></head><body>
<div class="note"><b>Bang Kapi drainage GIS audit</b>BMA 2021 pipe geometry. Green points are <b>candidate pipe–canal connections</b>, not verified outfalls or operating routes. Toggle layers at upper right.</div><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>const D=__DATA__;
const map=L.map('map').setView([13.756,100.61],12);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors; source geometry © BMA'}).addTo(map);
function clean(x){return String(x??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));}
const pipes=L.geoJSON(D.pipes,{style:f=>({color:f.properties.PIPE_TYPE==='Main'?'#555':'#aaa',weight:f.properties.PIPE_TYPE==='Main'?2:1,opacity:.7}),onEachFeature:(f,l)=>l.bindPopup('Pipe '+clean(f.properties.OBJECTID_1)+'<br>'+clean(f.properties.NAME)+'<br>To: '+clean(f.properties.TO_ROAD))});
const canals=L.geoJSON(D.canals,{style:{color:'#0876b8',weight:3,opacity:.8},onEachFeature:(f,l)=>l.bindPopup('Canal: '+clean(f.properties.KHONG_NAME))}).addTo(map);
const assets=L.geoJSON(D.assets,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:5,color:f.properties.kind==='gate'?'#e87500':'#d12a55',fillOpacity:.9}),style:{color:'#6f42a0',weight:4},onEachFeature:(f,l)=>l.bindPopup(clean(f.properties.kind)+'<br>'+clean(f.properties.NAME||f.properties['ชื่ออาคารบังคับน้ำ']))}).addTo(map);
const links=L.geoJSON(D.connections,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:5,color:'#168140',fillOpacity:.75}),onEachFeature:(f,l)=>l.bindPopup('Candidate connection<br>Canal: '+clean(f.properties.canal)+'<br>Pipe ends: '+clean(f.properties.pipe_ends_in_cluster)+'<br>Closest line: '+clean(f.properties.closest_canal_m)+' m')}).addTo(map);
L.control.layers(null,{'BMA canals':canals,'BMA pipes':pipes,'Pumps, gates, tunnels':assets,'Candidate canal connections':links},{collapsed:false}).addTo(map);map.fitBounds(canals.getBounds());</script></body></html>"""
    MAP_HTML.write_text(template.replace("__DATA__", embedded))


if __name__ == "__main__":
    build()
