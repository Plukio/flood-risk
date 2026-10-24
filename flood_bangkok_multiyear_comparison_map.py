"""Map Bang Kapi cell ranks from historical, per-cell, and expanded CNN models.

Run after flood_bangkok_spatial_cnn_multiyear.py --expanded:
    .venv/bin/python flood_bangkok_multiyear_comparison_map.py
"""

from __future__ import annotations

import csv
import json

import numpy as np

from flood_bangkapi_spatial_cnn import percentiles
from flood_bangkapi_surface_model import OUTPUT


PREFIX = "surface_crossdistrict_terrain_cnn_multiyear_eleven"


def main() -> None:
    with (OUTPUT / f"{PREFIX}_cell_comparison.csv").open(
            encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    with (OUTPUT / "surface_proxy_historical_comparison.csv").open(
            encoding="utf-8-sig", newline="") as file:
        tree = {row["cell_id"]: row for row in csv.DictReader(file)}
    if len(rows) != 527 or set(tree) != {row["cell_id"] for row in rows}:
        raise ValueError("Comparison cells differ")
    for row in rows:
        old = tree[row["cell_id"]]
        if int(old["observed_positive_days_oct_dec"]) != int(row["observed_days_oct_dec"]):
            raise ValueError("Per-cell tree uses different test labels")
        row["per_cell_tree_expected_days"] = float(old["model_expected_positive_days_oct_dec"])
    score_columns = {
        "historical": "historical_positive_days_jan_sep",
        "per_cell_tree": "per_cell_tree_expected_days",
        "six_districts_2025": "six_districts_2025_only_expected_days",
        "bang_kapi_multiyear": "bk_multiyear_expected_days",
        "six_districts_multiyear": "six_districts_multiyear_expected_days",
        "eleven_districts_multiyear": "eleven_districts_multiyear_expected_days",
        "observed": "observed_days_oct_dec",
    }
    for name, column in score_columns.items():
        ranks = percentiles(np.asarray([float(row[column]) for row in rows]))
        for row, rank in zip(rows, ranks):
            row[f"{name}_rank"] = round(float(rank), 2)
    geojson = json.loads((OUTPUT / "surface_cells_250m.geojson").read_text())
    by_id = {row["cell_id"]: row for row in rows}
    for feature in geojson["features"]:
        feature["properties"].update(by_id[feature["properties"]["cell_id"]])
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bang Kapi training-data comparison</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,sans-serif;color:#20313b}header{background:#12344b;color:white;padding:12px 16px;display:flex;gap:15px;align-items:center;flex-wrap:wrap}header strong{font-size:17px}header small{max-width:690px}select{font:inherit;padding:5px;border:0;border-radius:3px}#map{height:calc(100vh - 84px);min-height:480px}.legend{background:white;padding:10px;box-shadow:0 1px 8px #5555;border-radius:5px;line-height:1.45}.bar{height:12px;background:linear-gradient(90deg,#fff1bd,#e3944a,#9b2429);margin:7px 0}</style></head><body><header><strong>Bang Kapi · added years and districts</strong><label>Show <select id="view"><option value="historical_rank">Historical count</option><option value="per_cell_tree_rank">Per-cell tree</option><option value="six_districts_2025_rank">Six districts · 2025</option><option value="bang_kapi_multiyear_rank">Bang Kapi · four years</option><option value="six_districts_multiyear_rank">Six districts · four years</option><option value="eleven_districts_multiyear_rank">Eleven districts · four years</option><option value="observed_rank">Observed Oct–Dec</option></select></label><small>Rank among 527 cells for 100 m flood-report-buffer-positive days. The October–December 2025 outcomes are shown only for retrospective evaluation. Ranks are not verified flood probabilities.</small></header><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const grid=__GRID__;const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);let active='bang_kapi_multiyear_rank';
function color(p){const t=Math.max(0,Math.min(1,Number(p)/100));return `hsl(${48-46*t},${85-20*t}%,${93-56*t}%)`}
function popup(p){return `<b>${p.cell_id}</b><br>Historical Jan–Sep positive days: <b>${p.historical_positive_days_jan_sep}</b><br>Bang Kapi four-year expected Oct–Dec days: <b>${p.bk_multiyear_expected_days}</b><br>Six-district four-year expected days: <b>${p.six_districts_multiyear_expected_days}</b><br>Eleven-district four-year expected days: <b>${p.eleven_districts_multiyear_expected_days}</b><br>Observed Oct–Dec positive days: <b>${p.observed_days_oct_dec}</b><br>Displayed rank: <b>${Number(p[active]).toFixed(1)}</b> percentile<br><small>Complaint-derived 100 m buffer, not verified water extent.</small>`}
const cells=L.geoJSON(grid,{style:f=>({color:'#5a5050',weight:.25,fillColor:color(f.properties[active]),fillOpacity:.83}),onEachFeature:(f,l)=>l.bindPopup(popup(f.properties))}).addTo(map);map.fitBounds(cells.getBounds());
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Cell rank percentile</b><div class="bar"></div><small>0 → 100; darker means higher rank.<br>Zero-day cells are rank 0; ties share a rank.</small>';return d};legend.addTo(map);
document.getElementById('view').addEventListener('change',e=>{active=e.target.value;cells.setStyle(f=>({fillColor:color(f.properties[active])}));cells.eachLayer(l=>l.setPopupContent(popup(l.feature.properties)));});
</script></body></html>"""
    (OUTPUT / f"{PREFIX}_comparison_map.html").write_text(
        html.replace("__GRID__", json.dumps(geojson, ensure_ascii=False).replace("</", "<\\/")))


if __name__ == "__main__":
    main()
