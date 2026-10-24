"""Turn exploratory rain-stress rankings into a reviewable inspection queue.

The queue is an as-of-1-Oct-2025 demonstration, not a current deployment plan.
It reserves up to five cells per district for Jan–Sep Traffy history and fills
the remaining ten slots by agreement across two models and four rain scenes.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from flood_bangkok_spatial_cnn_citywide import CAL_N, FEATURE_CACHE
from flood_bangkok_transfer import load_districts
from flood_bangkapi_surface_grid import TO_LONLAT, grid_spec
from flood_bangkapi_surface_model import OUTPUT, write_csv


PREFIX = "surface_citywide_rain_scenario_stress_v1"
SCENES = ("20mm_2h", "40mm_2h", "40mm_4h", "20mm_2h_wet_prior")
MODELS = ("canal_cnn", "cnn_gnn")


def read_scores() -> dict:
    result = defaultdict(lambda: defaultdict(dict))
    with (OUTPUT / f"{PREFIX}_cells.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            code = row["district_code"]
            cell = (int(row["row"]), int(row["col"]))
            if cell not in result[code]:
                result[code][cell] = {"district": row["district"],
                                      "lon": float(row["lon"]), "lat": float(row["lat"]),
                                      "canal_distance_m": float(row["canal_distance_m"]),
                                      "pump_capacity_within_1km_reported_units": float(
                                          row["pump_capacity_within_1km_reported_units"]),
                                      "underpass_within_250m": row["underpass_within_250m"] == "True"}
            result[code][cell][(row["model"], row["scenario"])] = {
                "rank": float(row["district_rank_percentile"]),
                "score": float(row["proxy_score"])}
    return result


def inspection_action(info: dict) -> str:
    if info["underpass_within_250m"]:
        return "Check underpass drainage, pump readiness, and closure trigger"
    if info["canal_distance_m"] <= 250:
        return "Check nearby canal-side inlets and obstructions; verify actual drain connection"
    if info["pump_capacity_within_1km_reported_units"] > 0:
        return "Check street drains and verify nearby pump operating status"
    return "Check street drains, access, and observed pooling with a field visit"


def queue_for(code: str, cells: dict) -> list[dict]:
    meta = json.loads((FEATURE_CACHE / f"{code}_meta.json").read_text())
    rr, cc = np.asarray(meta["valid_rows"]), np.asarray(meta["valid_cols"])
    y = np.load(FEATURE_CACHE / f"{code}_y.npy", mmap_mode="r")
    history = np.asarray(y[1096:CAL_N, 0, rr, cc]).sum(axis=0)
    if len(cells) != len(rr):
        raise ValueError(f"Scenario/grid cell count differs in {code}")
    for r, c, n in zip(rr, cc, history):
        info = cells[(int(r), int(c))]
        info["jan_sep_2025_complaint_positive_days"] = int(n)
        info["mean_model_rank_percentile"] = float(np.mean([
            info[(model, scenario)]["rank"]
            for model in MODELS for scenario in SCENES]))
    ordered_history = sorted((key for key in cells if cells[key]["jan_sep_2025_complaint_positive_days"]),
                             key=lambda key: (-cells[key]["jan_sep_2025_complaint_positive_days"],
                                              -cells[key]["mean_model_rank_percentile"], key))
    chosen = [(key, "prior_complaint_history") for key in ordered_history[:5]]
    taken = {key for key, _ in chosen}
    ordered_model = sorted((key for key in cells if key not in taken),
                           key=lambda key: (-cells[key]["mean_model_rank_percentile"],
                                            -cells[key]["jan_sep_2025_complaint_positive_days"], key))
    chosen.extend((key, "scenario_model_consensus") for key in ordered_model[:10 - len(chosen)])
    rows = []
    for slot, (key, reason) in enumerate(chosen, 1):
        info = cells[key]
        rows.append({"district_code": code, "district": info["district"],
                     "queue_slot": slot, "selection_reason": reason,
                     "row": key[0], "col": key[1],
                     "lon": info["lon"], "lat": info["lat"],
                     "jan_sep_2025_complaint_positive_days": info["jan_sep_2025_complaint_positive_days"],
                     "mean_model_rank_percentile": round(info["mean_model_rank_percentile"], 5),
                     "canal_distance_m": info["canal_distance_m"],
                     "pump_capacity_within_1km_reported_units": info["pump_capacity_within_1km_reported_units"],
                     "underpass_within_250m": info["underpass_within_250m"],
                     "suggested_field_check": inspection_action(info)})
    return rows


def make_bang_kapi_map(cells: dict, queue: list[dict], district: dict) -> None:
    spec = grid_spec(district["polygon"])
    features = []
    for (row, col), info in cells.items():
        x0 = spec["west"] + col * 250
        x1 = x0 + 250
        y1 = spec["north"] - row * 250
        y0 = y1 - 250
        corners = [TO_LONLAT.transform(x, y) for x, y in
                   ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [corners]},
                         "properties": {"row": row, "col": col,
                                        "history": info["jan_sep_2025_complaint_positive_days"],
                                        "canal_m": info["canal_distance_m"],
                                        "underpass": info["underpass_within_250m"],
                                        "rank": {model: {name: info[(model, name)]["rank"]
                                                         for name in SCENES + ("80mm_4h",)}
                                                 for model in MODELS}}})
    payload = json.dumps({"features": features, "queue": queue}, ensure_ascii=False,
                         separators=(",", ":"))
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bang Kapi rain stress test</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,sans-serif;color:#173044}header{padding:12px 16px;background:#12344b;color:white;display:flex;gap:14px;align-items:center;flex-wrap:wrap}header strong{font-size:17px}header label{white-space:nowrap}select{font:inherit;padding:4px}header small{max-width:700px;line-height:1.4}#map{height:calc(100vh - 116px);min-height:470px}.legend{background:white;padding:10px;border-radius:4px;box-shadow:0 1px 8px #5555;line-height:1.45}.swatch{height:12px;background:linear-gradient(90deg,#e7f2fa,#94c7df,#276b9c,#083e64);margin:7px 0}.queue{color:#a83b20;font-weight:bold}</style></head><body><header><strong>Bang Kapi · rain-input stress test</strong><label>Model <select id="model"><option value="canal_cnn">Canal CNN</option><option value="cnn_gnn">CNN–GNN</option></select></label><label>Rain scene <select id="scenario"><option value="20mm_2h">20 mm / 2 h</option><option value="40mm_2h">40 mm / 2 h, harder rain</option><option value="40mm_4h">40 mm / 4 h, longer rain</option><option value="80mm_4h">80 mm / 4 h, beyond training</option><option value="20mm_2h_wet_prior">20 mm / 2 h + wet prior days</option></select></label><small>Fixed inputs as of 1 Oct 2025. Colour shows rank within Bang Kapi, not flood probability or water depth. Red circles are a ten-cell preparation shortlist.</small></header><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>const data=__DATA__;const map=L.map('map');L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);function color(v){return `hsl(203 72% ${96-64*v}%)`};function selected(){return [document.getElementById('model').value,document.getElementById('scenario').value]};function rank(p){const [m,s]=selected();return p.rank[m][s]};const layer=L.geoJSON({type:'FeatureCollection',features:data.features},{style:f=>({color:'#8299a7',weight:.45,fillColor:color(rank(f.properties)),fillOpacity:.68}),onEachFeature:(f,l)=>{l.on('click',()=>{const p=f.properties;l.bindPopup(`<b>Cell ${p.row}:${p.col}</b><br>District rank percentile: ${rank(p).toFixed(3)}<br>Jan–Sep complaint-positive days: ${p.history}<br>Mapped canal distance: ${Math.round(p.canal_m)} m<br>Mapped underpass nearby: ${p.underpass?'yes':'no'}<br><small>Rank is a complaint proxy; inspect on site before action.</small>`).openPopup()})}}).addTo(map);map.fitBounds(layer.getBounds());const q=L.layerGroup();data.queue.forEach(r=>L.circleMarker([r.lat,r.lon],{radius:6,color:'#a83b20',fillColor:'#e66b44',fillOpacity:.9,weight:1.5}).bindPopup(`<b>Preparation slot ${r.queue_slot}</b><br>${r.selection_reason.replaceAll('_',' ')}<br>${r.suggested_field_check}`).addTo(q));q.addTo(map);L.control.layers(null,{'Preparation shortlist':q},{collapsed:false}).addTo(map);const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Within-district rank</b><div class="swatch"></div><small>Lower → higher score<br>Red = field-review shortlist<br>80 mm scene is extrapolation</small>';return d};legend.addTo(map);function redraw(){layer.setStyle(f=>({fillColor:color(rank(f.properties))}));layer.eachLayer(l=>l.closePopup())};document.getElementById('model').addEventListener('change',redraw);document.getElementById('scenario').addEventListener('change',redraw);</script></body></html>""".replace("__DATA__", payload)
    (OUTPUT / f"{PREFIX}_bang_kapi_map.html").write_text(html)


def main() -> None:
    districts = {d["code"]: d for d in load_districts()}
    scores = read_scores()
    queue = []
    for code in districts:
        queue.extend(queue_for(code, scores[code]))
    if len(queue) != 500:
        raise ValueError("Expected ten preparation cells in each of 50 districts")
    write_csv(OUTPUT / f"{PREFIX}_preparation_queue.csv", queue)
    make_bang_kapi_map(scores["1006"], [r for r in queue if r["district_code"] == "1006"],
                       districts["1006"])
    overlap = {}
    for model in MODELS:
        overlap[model] = {}
        for scenario in ("40mm_2h", "40mm_4h", "80mm_4h", "20mm_2h_wet_prior"):
            counts = []
            for code, cells in scores.items():
                def top(name):
                    return set(sorted(cells, key=lambda key: cells[key][(model, name)]["score"],
                                      reverse=True)[:10])
                counts.append(len(top("20mm_2h") & top(scenario)))
            overlap[model][scenario] = round(float(np.mean(counts)), 3)
    bma_path = OUTPUT / "surface_bma_positive_site_check_citywide_v1_site_days.csv"
    bma_capture = None
    if bma_path.exists():
        with bma_path.open(encoding="utf-8-sig", newline="") as file:
            bma = list(csv.DictReader(file))
        chosen = {(r["district_code"], int(r["row"]), int(r["col"])) for r in queue}
        def captured(rows):
            return {"site_days": len(rows), "selected_site_days": sum(
                (r["district_code"], int(r["grid_row"]), int(r["grid_col"])) in chosen
                for r in rows)}
        bma_capture = {"all_exact_matched": captured(bma),
                       "geographic_holdouts": captured([
                           r for r in bma if r["geographic_holdout"] == "True"]),
                       "location_caveat": "2026 BMA inventory anchors approximate past road-report locations"}
    audit = {"as_of_date": "2025-10-01", "districts": len(districts),
             "queue_cells": len(queue), "history_slots": sum(
                 r["selection_reason"] == "prior_complaint_history" for r in queue),
             "model_slots": sum(r["selection_reason"] == "scenario_model_consensus" for r in queue),
             "mean_top10_overlap_with_20mm_2h": overlap,
             "independent_bma_positive_capture": bma_capture,
             "limitation": "Review list based on complaint rankings, not verified flooded/dry cells"}
    (OUTPUT / f"{PREFIX}_preparation_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")
    print("saved",len(queue),"queue cells and Bang Kapi map")


if __name__ == "__main__":
    main()
