"""Audit open wet/dry evidence before publishing cell flood probabilities.

Run: .venv/bin/python flood_bangkapi_probability_evidence.py

This intentionally never converts complaint absence, an offline camera's
is_flooded=False default, or a canal reading into a dry surface observation.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from shapely.geometry import Point

from flood_bangkapi_surface_grid import GRID_M, TO_UTM, district_polygon, grid_spec


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "flood_bangkapi_surface"
SITES = ROOT / "artifacts" / "flood_bangkapi_site_ranking" / "site_ranking_2025.csv"
GRID = OUT / "surface_cells_250m.geojson"
CCTV_URL = "https://floodcheck.gistda.or.th/cctv"
FLOODROAD_URL = "https://floodcheck.gistda.or.th/floodroad"


def cell_id(lon: float, lat: float, polygon, spec: dict, valid: set[str]) -> str | None:
    x, y = TO_UTM.transform(lon, lat)
    if not polygon.covers(Point(x, y)):
        return None
    row = math.floor((spec["north"] - y) / GRID_M)
    col = math.floor((x - spec["west"]) / GRID_M)
    result = f"bk:{row:02d}:{col:02d}"
    return result if result in valid else None


def bma_anchors(polygon, spec: dict, valid: set[str]) -> tuple[dict, list[dict], list[dict]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    features = []
    wet_site_days = []
    with SITES.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            lon, lat = float(row["lon"]), float(row["lat"])
            cid = cell_id(lon, lat, polygon, spec, valid)
            days = int(row["report_days_2025"])
            if cid:
                counts[cid]["approximate_wet_site_days"] += days
                counts[cid]["site_anchors"] += 1
            for day in row["report_dates_2025"].split(";"):
                if day:
                    wet_site_days.append({"date_bangkok": day, "site": row["site"],
                                          "site_label": row["label"],
                                          "approximate_cell_id": cid or "",
                                          "site_lon": lon, "site_lat": lat,
                                          "spatial_status": "2026 inventory anchor, not verified 2025 cell"})
            features.append({"type": "Feature", "geometry": {"type": "Point",
                             "coordinates": [lon, lat]}, "properties": {
                                 "site": row["site"], "label": row["label"],
                                 "report_days_2025": days, "assigned_cell": cid,
                                 "location_warning": "2026 inventory coordinate; approximate 2025 site anchor"}})
    return counts, features, wet_site_days


def camera_snapshot(polygon, spec: dict, valid: set[str]) -> tuple[dict, dict, list[dict]]:
    audit = {"source": CCTV_URL, "http_status": None, "citywide_cameras": None,
             "citywide_online": None, "bang_kapi_cameras": 0,
             "bang_kapi_online": 0, "bang_kapi_flooded_flag_true": 0,
             "warning": "An offline camera's is_flooded=false is not a dry label; "
                        "the API response has no observation timestamp or archived frame."}
    counts: dict[str, Counter] = defaultdict(Counter)
    features = []
    try:
        response = requests.get(CCTV_URL, timeout=25)
        audit["http_status"] = response.status_code
        response.raise_for_status()
        cameras = response.json()["data"]
        audit["citywide_cameras"] = len(cameras)
        audit["citywide_online"] = sum(camera.get("is_online") is True for camera in cameras)
        for camera in cameras:
            try:
                lon, lat = float(camera["longitude"]), float(camera["latitude"])
                cid = cell_id(lon, lat, polygon, spec, valid)
            except (TypeError, ValueError, KeyError):
                continue
            if not cid:
                continue
            online = camera.get("is_online") is True
            flag = camera.get("is_flooded") is True
            audit["bang_kapi_cameras"] += 1
            audit["bang_kapi_online"] += int(online)
            audit["bang_kapi_flooded_flag_true"] += int(flag)
            counts[cid]["camera_locations"] += 1
            counts[cid]["online_camera_locations"] += int(online)
            features.append({"type": "Feature", "geometry": {"type": "Point",
                             "coordinates": [lon, lat]}, "properties": {
                                 "camera_id": str(camera.get("cameraId", "")),
                                 "cell_id": cid, "is_online": online,
                                 "is_flooded_api_flag": flag,
                                 "verified_wet_or_dry": False}})
    except (requests.RequestException, ValueError, KeyError) as exc:
        audit["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return audit, counts, features


def floodroad_access() -> dict:
    try:
        response = requests.get(FLOODROAD_URL, timeout=20)
        return {"source": FLOODROAD_URL, "http_status": response.status_code,
                "usable_without_credentials": response.ok,
                "note": "Current access check only; historical records and dry-state semantics not established."}
    except requests.RequestException as exc:
        return {"source": FLOODROAD_URL, "http_status": None,
                "usable_without_credentials": False,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def write_map(grid: dict, anchors: list[dict], cameras: list[dict], captured: str) -> None:
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bang Kapi flood-probability evidence audit</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{margin:0;font:14px system-ui,sans-serif;color:#20313b}header{background:#153b51;color:white;padding:12px 17px}header strong{font-size:17px}header span{display:block;margin-top:3px}#map{height:calc(100vh - 83px);min-height:480px}.legend{background:white;padding:10px;max-width:280px;line-height:1.45;box-shadow:0 1px 8px #5555;border-radius:5px}.sw{display:inline-block;width:11px;height:11px;margin-right:6px}</style></head><body>
<header><strong>Bang Kapi · evidence for cell flood probabilities</strong><span>All 527 flood probabilities are unavailable: no time-matched, verified wet/dry cell sample. GISTDA CCTV status captured __CAPTURED__.</span></header>
<div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const grid=__GRID__, anchors=__ANCHORS__, cameras=__CAMERAS__;
const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
const cells=L.geoJSON(grid,{style:f=>({color:'#526471',weight:.35,fillColor:f.properties.traffy_2025_flood_reports>0?'#f1c28b':'#dce3e7',fillOpacity:.77}),onEachFeature:(f,l)=>{const p=f.properties;l.bindPopup(`<b>${p.cell_id}</b><br>Actual flood probability: <b>unavailable</b><br>Traffy flood-category complaints in 2025: ${p.traffy_2025_flood_reports}<br>Approximate BMA road-report site-days: ${p.approximate_bma_wet_site_days}<br>GISTDA camera locations: ${p.gistda_camera_locations}, online: ${p.gistda_online_camera_locations}<br>Verified dry observations: 0`);}}).addTo(map);
L.geoJSON(anchors,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:4,color:'#3b3b77',fillColor:'#655ab0',fillOpacity:.9,weight:1}),onEachFeature:(f,l)=>l.bindPopup(`<b>${f.properties.label}</b><br>BMA 2025 road-report days: ${f.properties.report_days_2025}<br>Approximate site coordinate from 2026 inventory`)}).addTo(map);
L.geoJSON(cameras,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:6,color:'#aa3030',fillColor:'#e96557',fillOpacity:.9,weight:1}),onEachFeature:(f,l)=>l.bindPopup(`GISTDA camera ${f.properties.camera_id}<br>Online: ${f.properties.is_online}<br>API flooded flag: ${f.properties.is_flooded_api_flag}<br><b>No verified wet/dry frame</b>`)}).addTo(map);
map.fitBounds(cells.getBounds());const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Evidence coverage, not flood risk</b><br><span class="sw" style="background:#f1c28b"></span>Cell with a Traffy flood complaint<br><span class="sw" style="background:#dce3e7"></span>No complaint observed<br><span class="sw" style="background:#655ab0"></span>BMA road-report anchor<br><span class="sw" style="background:#e96557"></span>GISTDA camera location<br><small>Complaint absence and offline camera flags are not dry observations.</small>';return d};legend.addTo(map);
</script></body></html>"""
    html = html.replace("__GRID__", json.dumps(grid, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__ANCHORS__", json.dumps({"type": "FeatureCollection", "features": anchors}, ensure_ascii=False))
    html = html.replace("__CAMERAS__", json.dumps({"type": "FeatureCollection", "features": cameras}, ensure_ascii=False))
    html = html.replace("__CAPTURED__", captured)
    (OUT / "surface_probability_evidence_map.html").write_text(html)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = json.loads(GRID.read_text())
    valid = {feature["properties"]["cell_id"] for feature in grid["features"]}
    polygon = district_polygon()
    spec = grid_spec(polygon)
    captured = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
    bma_counts, anchors, wet_site_days = bma_anchors(polygon, spec, valid)
    with (OUT / "approximate_bma_wet_site_days_2025.csv").open(
            "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(wet_site_days[0]))
        writer.writeheader()
        writer.writerows(wet_site_days)
    camera_audit, camera_counts, cameras = camera_snapshot(polygon, spec, valid)
    road_audit = floodroad_access()
    for feature in grid["features"]:
        cid = feature["properties"]["cell_id"]
        feature["properties"].update({
            "flood_probability": None,
            "probability_status": "unavailable_no_verified_wet_dry_cell_sample",
            "approximate_bma_wet_site_days": bma_counts[cid]["approximate_wet_site_days"],
            "bma_site_anchors": bma_counts[cid]["site_anchors"],
            "gistda_camera_locations": camera_counts[cid]["camera_locations"],
            "gistda_online_camera_locations": camera_counts[cid]["online_camera_locations"],
            "verified_dry_observations": 0})
    (OUT / "surface_probability_evidence.geojson").write_text(json.dumps(grid, ensure_ascii=False))
    (OUT / "gistda_bangkapi_camera_snapshot.json").write_text(json.dumps(
        {"captured_bangkok_time": captured, "source": CCTV_URL, "cameras": [
            feature["properties"] | {"coordinates": feature["geometry"]["coordinates"]}
            for feature in cameras]}, ensure_ascii=False, indent=2) + "\n")
    audit = {"captured_bangkok_time": captured, "cell_count": len(grid["features"]),
             "cells_with_flood_probability": 0,
             "cells_with_traffy_flood_complaint": sum(
                 feature["properties"]["traffy_2025_flood_reports"] > 0
                 for feature in grid["features"]),
             "bma_2025_road_report_site_days": sum(
                 feature["properties"]["report_days_2025"] for feature in anchors),
             "bma_anchor_sites_inside_district": sum(
                 feature["properties"]["assigned_cell"] is not None for feature in anchors),
             "bma_site_days_with_anchor_inside_district": sum(
                 bool(row["approximate_cell_id"]) for row in wet_site_days),
             "bma_anchor_cells": sum(bool(count["site_anchors"]) for count in bma_counts.values()),
             "bma_anchor_warning": "Coordinates from a 2026 inventory; 2025 named-site reports are wet evidence with spatial uncertainty, not a sampled cell wet/dry panel.",
             "gistda_cctv": camera_audit, "gistda_floodroad": road_audit,
             "verified_dry_cell_events": 0,
             "probability_gate_passed": False,
             "reason": "No independently verified dry cell-events paired with wet cell-events across storms; complaint absence and offline camera defaults cannot supply them."}
    (OUT / "surface_probability_evidence_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")
    write_map(grid, anchors, cameras, captured)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
