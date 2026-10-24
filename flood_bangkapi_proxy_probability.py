"""Calibrated probability map for report-defined flood footprints.

Run: .venv/bin/python flood_bangkapi_proxy_probability.py [--mode connected|buffer]

Default connected mode links similar-time, nearby reports through low LiDAR
terrain; reports without such an area use a 100 m radius. Buffer mode retains
the original 100 m-only comparison. No report footprint means negative by the
user's chosen proxy definition, not verified dry physical surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import rankdata
from shapely.geometry import Point
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from flood_bangkapi_surface_grid import GRID_M, TO_UTM, cell_box, district_polygon, grid_spec
from flood_bangkapi_connected_footprints import build as connected_labels
from flood_bangkapi_surface_model import (
    DIRECT_TYPES, OUTPUT, TRAFFY, activity_by_cell, cells, model,
    rainfall_by_date, rank_metrics, scenario_rain, vector, write_csv,
)


RADIUS_M = 100
TRAIN_END = date(2025, 7, 31)
CAL_END = date(2025, 9, 30)
TEST_END = date(2025, 12, 31)
SCENARIO_DAY = date(2025, 10, 15)
SCENARIOS = {
    "4mm_h_2h": (4.0, 2),
    "8mm_h_2h": (8.0, 2),
    "4mm_h_4h": (4.0, 4),
    "8mm_h_4h": (8.0, 4),
}


def proxy_labels(valid_ids: set[str]) -> tuple[dict[str, set[date]], dict]:
    district = district_polygon()
    spec = grid_spec(district)
    cell_shapes = {}
    for cid in valid_ids:
        _, row, col = cid.split(":")
        cell_shapes[cid] = cell_box(spec, int(row), int(col)).intersection(district)
    labels: dict[str, set[date]] = defaultdict(set)
    unique_reports = set()
    report_count = 0
    for row in csv.DictReader(TRAFFY.open(encoding="utf-8-sig", newline="")):
        if row["type"] not in DIRECT_TYPES:
            continue
        x, y = TO_UTM.transform(float(row["lon"]), float(row["lat"]))
        if not district.covers(Point(x, y)):
            continue
        report_day = date.fromisoformat(row["timestamp"][:10])
        report_count += 1
        unique_reports.add(row["ticket_hash"])
        footprint = Point(x, y).buffer(RADIUS_M)
        left = max(0, math.floor((x - RADIUS_M - spec["west"]) / GRID_M))
        right = min(spec["cols"] - 1, math.floor((x + RADIUS_M - spec["west"]) / GRID_M))
        top = max(0, math.floor((spec["north"] - (y + RADIUS_M)) / GRID_M))
        bottom = min(spec["rows"] - 1, math.floor((spec["north"] - (y - RADIUS_M)) / GRID_M))
        for rr in range(top, bottom + 1):
            for cc in range(left, right + 1):
                cid = f"bk:{rr:02d}:{cc:02d}"
                if cid in cell_shapes and cell_shapes[cid].intersects(footprint):
                    labels[cid].add(report_day)
    audit = {"source": "BMA Traffy Fondue 2025 direct flood-category reports",
             "report_rows_in_bang_kapi": report_count,
             "unique_hashed_tickets": len(unique_reports),
             "buffer_radius_m": RADIUS_M,
             "positive_cells_any_2025": sum(bool(x) for x in labels.values()),
             "positive_cell_days_2025": sum(len(x) for x in labels.values())}
    return labels, audit


def calibrate_intercept(raw: np.ndarray, target: np.ndarray) -> float:
    if target.sum() < 1 or target.sum() == len(target):
        raise ValueError("Calibration period needs both positive and negative proxy labels")
    logits = logit(np.clip(raw, 1e-10, 1 - 1e-10))
    observed = float(target.mean())
    return float(brentq(lambda bias: float(expit(logits + bias).mean()) - observed,
                        -30.0, 30.0))


def apply_calibration(raw: np.ndarray, bias: float) -> np.ndarray:
    return expit(logit(np.clip(raw, 1e-10, 1 - 1e-10)) + bias)


def evaluation(p: np.ndarray, y: np.ndarray) -> dict:
    # Equal-sized probability bins expose whether the printed percentages
    # match held-out report-proxy frequencies; no bin is silently discarded.
    order = np.argsort(p)
    bins = []
    for indices in np.array_split(order, 10):
        bins.append({"n": len(indices), "predicted_mean": round(float(p[indices].mean()), 6),
                     "observed_rate": round(float(y[indices].mean()), 6),
                     "positives": int(y[indices].sum())})
    return {"n": len(y), "positives": int(y.sum()),
            "observed_rate": round(float(y.mean()), 6),
            "predicted_positive_cell_days": round(float(p.sum()), 1),
            "mean_predicted_probability": round(float(p.mean()), 6),
            "average_precision": round(float(average_precision_score(y, p)), 5),
            "brier": round(float(brier_score_loss(y, p)), 6),
            "log_loss": round(float(log_loss(y, p)), 6),
            "reliability_deciles": bins}


def write_probability_map(rows: list[dict], scale_max: float, prefix: str,
                          description: str) -> None:
    geojson = json.loads((OUTPUT / "surface_cells_250m.geojson").read_text())
    by_cell = {row["cell_id"]: row for row in rows}
    for feature in geojson["features"]:
        feature["properties"].update(by_cell[feature["properties"]["cell_id"]])
    (OUTPUT / f"{prefix}_probability_scenarios.geojson").write_text(json.dumps(
        geojson, ensure_ascii=False))
    footprint_path = OUTPUT / "connected_report_footprints_2025.geojson"
    footprints = json.loads(footprint_path.read_text()) if prefix.startswith("surface_connected") else {
        "type": "FeatureCollection", "features": []}
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bang Kapi report-footprint probability</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,sans-serif;color:#20313b}header{background:#12344b;color:white;padding:12px 16px;display:flex;gap:15px;align-items:center;flex-wrap:wrap}header strong{font-size:17px}header small{max-width:610px}select{font:inherit;padding:5px;border:0;border-radius:3px}#map{height:calc(100vh - 84px);min-height:480px}.legend{background:white;padding:10px;box-shadow:0 1px 8px #5555;border-radius:5px;max-width:265px;line-height:1.45}.bar{height:12px;background:linear-gradient(90deg,#fff1bd,#e3944a,#9b2429);margin:7px 0}</style></head><body>
<header><strong>Bang Kapi · 250 m cells</strong><label>Rain scenario <select id="scenario"><option value="4mm_h_2h">4 mm/h for 2 h</option><option value="8mm_h_2h">8 mm/h for 2 h</option><option value="4mm_h_4h">4 mm/h for 4 h</option><option value="8mm_h_4h">8 mm/h for 4 h</option></select></label><small>Chance of __DESCRIPTION__ touching a cell on one October day. No report footprint = dry by the chosen proxy definition. __SCENARIO_WARNING__This is not independently observed inundation.</small></header>
<div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const grid=__GRID__,footprints=__FOOTPRINTS__,scaleMax=__SCALE_MAX__;const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);let active='4mm_h_2h';
function key(){return 'proxy_flood_probability_'+active}function color(p){const t=Math.max(0,Math.min(1,Number(p)/scaleMax));return `hsl(${48-46*t},${85-20*t}%,${93-56*t}%)`}
function popup(p){return `<b>${p.cell_id}</b><br>Report-footprint proxy probability: <b>${(100*Number(p[key()])).toFixed(2)}%</b><br>Footprint rule: ${p.proxy_definition}<br>2025 positive proxy days: ${p.proxy_positive_days_2025}<br>Elevation: ${p.elev_median_m} m<br>Road length: ${p.osm_road_m} m<br><b>Actual inundation probability: unmeasured</b>`}
const cells=L.geoJSON(grid,{style:f=>({color:'#5a5050',weight:.25,fillColor:color(f.properties[key()]),fillOpacity:.83}),onEachFeature:(f,l)=>l.bindPopup(popup(f.properties))}).addTo(map);map.fitBounds(cells.getBounds());
const outlines=L.geoJSON(footprints,{style:f=>({color:f.properties.method==='connected_low_terrain'?'#17588b':'#ad642d',weight:1.5,fillOpacity:0}),onEachFeature:(f,l)=>l.bindPopup(`${f.properties.method}<br>${f.properties.report_count} reports<br>${f.properties.report_dates.join(', ')}`)});if(footprints.features.length)L.control.layers(null,{'2025 inferred report areas':outlines}).addTo(map);
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Report-proxy probability</b><div class="bar"></div><small>0% → __SCALE_LABEL__% (darker). Same scale for every scenario.<br>Percentages are for the defined report proxy. Check held-out calibration before operational use.</small>';return d};legend.addTo(map);
document.getElementById('scenario').addEventListener('change',e=>{active=e.target.value;cells.setStyle(f=>({fillColor:color(f.properties[key()])}));cells.eachLayer(l=>l.setPopupContent(popup(l.feature.properties)));});
</script></body></html>"""
    html = html.replace("__GRID__", json.dumps(geojson, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__FOOTPRINTS__", json.dumps(footprints, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__DESCRIPTION__", description)
    html = html.replace("__SCENARIO_WARNING__", "CNN rain responses can be non-monotonic. "
                        if "_cnn" in prefix else "")
    html = html.replace("__SCALE_MAX__", repr(scale_max))
    html = html.replace("__SCALE_LABEL__", f"{100 * scale_max:.1f}")
    (OUTPUT / f"{prefix}_probability_map.html").write_text(html)


def write_historical_comparison_map(rows: list[dict], prefix: str) -> None:
    geojson = json.loads((OUTPUT / "surface_cells_250m.geojson").read_text())
    by_cell = {row["cell_id"]: row for row in rows}
    for feature in geojson["features"]:
        feature["properties"].update(by_cell[feature["properties"]["cell_id"]])
    (OUTPUT / f"{prefix}_historical_comparison.geojson").write_text(json.dumps(
        geojson, ensure_ascii=False))
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bang Kapi historical flood-report baseline</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,sans-serif;color:#20313b}header{background:#12344b;color:white;padding:12px 16px;display:flex;gap:15px;align-items:center;flex-wrap:wrap}header strong{font-size:17px}header small{max-width:690px}select{font:inherit;padding:5px;border:0;border-radius:3px}#map{height:calc(100vh - 84px);min-height:480px}.legend{background:white;padding:10px;box-shadow:0 1px 8px #5555;border-radius:5px;line-height:1.45}.bar{height:12px;background:linear-gradient(90deg,#fff1bd,#e3944a,#9b2429);margin:7px 0}</style></head><body><header><strong>Bang Kapi · historical baseline versus model</strong><label>Show <select id="view"><option value="historical_rank_percentile">Historical Jan–Sep rank</option><option value="model_rank_percentile">Model Oct–Dec rank</option><option value="observed_rank_percentile">Observed Oct–Dec rank</option></select></label><small>Rank among 527 cells for report-footprint-positive days. Model uses held-out Oct–Dec weather; observed is shown only for retrospective evaluation. Zero-day cells are rank 0.</small></header><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const grid=__GRID__;const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);let active='historical_rank_percentile';
function color(p){const t=Math.max(0,Math.min(1,Number(p)/100));return `hsl(${48-46*t},${85-20*t}%,${93-56*t}%)`}
function popup(p){return `<b>${p.cell_id}</b><br>Historical positive days, Jan–Sep: <b>${p.historical_positive_days_jan_sep}</b><br>Model expected positive days, Oct–Dec: <b>${p.model_expected_positive_days_oct_dec}</b><br>Observed positive days, Oct–Dec: <b>${p.observed_positive_days_oct_dec}</b><br>Current map rank: <b>${Number(p[active]).toFixed(1)}</b> percentile<br><small>These are report-footprint labels, not verified surface flooding.</small>`}
const cells=L.geoJSON(grid,{style:f=>({color:'#5a5050',weight:.25,fillColor:color(f.properties[active]),fillOpacity:.83}),onEachFeature:(f,l)=>l.bindPopup(popup(f.properties))}).addTo(map);map.fitBounds(cells.getBounds());
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Cell rank percentile</b><div class="bar"></div><small>0 → 100; darker means higher rank.<br>Compare ranks because Jan–Sep and Oct–Dec have different lengths. Tied counts share a rank.</small>';return d};legend.addTo(map);
document.getElementById('view').addEventListener('change',e=>{active=e.target.value;cells.setStyle(f=>({fillColor:color(f.properties[active])}));cells.eachLayer(l=>l.setPopupContent(popup(l.feature.properties)));});
</script></body></html>"""
    (OUTPUT / f"{prefix}_historical_comparison_map.html").write_text(
        html.replace("__GRID__", json.dumps(geojson, ensure_ascii=False).replace("</", "<\\/")))


def main(mode: str = "connected") -> None:
    sites = cells()
    ids = [row["cell_id"] for row in sites]
    if mode == "connected":
        labels, label_audit = connected_labels()
        prefix = "surface_connected"
        description = "a terrain-connected report area or 100 m fallback"
    else:
        labels, label_audit = proxy_labels(set(ids))
        prefix = "surface_proxy"
        description = "a 100 m flood-report buffer"
    weather = rainfall_by_date()
    activity = activity_by_cell()
    days = sorted(weather)
    n = len(sites)
    x_train, y_train, x_cal, y_cal, x_test, y_test = [], [], [], [], [], []
    prior = Counter()
    # Freeze historical labels and activity at the end of calibration for the
    # held-out test, so no October/November outcome enters later test features.
    for day in days:
        month = day.strftime("%Y-%m")
        for cell in sites:
            cid = cell["cell_id"]
            activity_count = sum(count for label, count in activity.get(cid, {}).items()
                                 if label < month and label <= "2025-09")
            features = vector(cell, day, weather[day], prior[cid], activity_count, False)
            target = int(day in labels.get(cid, set()))
            if day <= TRAIN_END:
                x_train.append(features); y_train.append(target)
            elif day <= CAL_END:
                x_cal.append(features); y_cal.append(target)
            elif day <= TEST_END:
                x_test.append(features); y_test.append(target)
        if day <= CAL_END:
            for cid in ids:
                prior[cid] += int(day in labels.get(cid, set()))
    X_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.uint8)
    X_cal = np.asarray(x_cal, dtype=np.float32)
    y_cal = np.asarray(y_cal, dtype=np.uint8)
    X_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.uint8)
    fitted = model(X_train.shape[1]).fit(X_train, y_train)
    raw_cal = fitted.predict_proba(X_cal)[:, 1]
    bias = calibrate_intercept(raw_cal, y_cal)
    raw_test = fitted.predict_proba(X_test)[:, 1]
    p_test = apply_calibration(raw_test, bias)
    test_days = [day for day in days if CAL_END < day <= TEST_END]
    observed_by_cell = y_test.reshape(len(test_days), n).sum(axis=0)
    expected_by_cell = p_test.reshape(len(test_days), n).sum(axis=0)
    historical_by_cell = np.asarray([prior[cid] for cid in ids])
    spatial_bins = []
    for indices in np.array_split(np.argsort(expected_by_cell), 10):
        spatial_bins.append({"cells": len(indices),
                             "predicted_positive_cell_days": round(float(expected_by_cell[indices].sum()), 1),
                             "observed_positive_cell_days": int(observed_by_cell[indices].sum())})
    daily_test = {}
    for month in sorted({day.strftime("%Y-%m") for day in test_days}):
        indices = [i for i, day in enumerate(test_days) if day.strftime("%Y-%m") == month]
        yy = y_test.reshape(len(test_days), n)[indices].ravel()
        pp = p_test.reshape(len(test_days), n)[indices].ravel()
        daily_test[month] = {"positives": int(yy.sum()),
                             "predicted_positive_cell_days": round(float(pp.sum()), 1),
                             "brier": round(float(brier_score_loss(yy, pp)), 6)}
    rain_strata = {}
    test_rain = np.asarray([weather[day]["rain_today_mm"] for day in test_days])
    for name, mask in (("under_1mm", test_rain < 1),
                       ("1_to_under_10mm", (test_rain >= 1) & (test_rain < 10)),
                       ("at_least_10mm", test_rain >= 10)):
        yy = y_test.reshape(len(test_days), n)[mask].ravel()
        pp = p_test.reshape(len(test_days), n)[mask].ravel()
        rain_strata[name] = {"days": int(mask.sum()), "positive_cell_days": int(yy.sum()),
                             "predicted_positive_cell_days": round(float(pp.sum()), 1),
                             "observed_rate": round(float(yy.mean()), 6) if len(yy) else None,
                             "predicted_mean": round(float(pp.mean()), 6) if len(pp) else None}
    metrics = {"target": f"at least one direct-category Traffy flood report footprint ({description}) touching a 250 m cell on a day",
               "negative_definition": "no report footprint touches the cell on that day",
               "not_target": "independently observed surface flooding",
               "train_dates": [str(days[0]), str(TRAIN_END)],
               "calibration_dates": ["2025-08-01", str(CAL_END)],
               "test_dates": [str(test_days[0]), str(test_days[-1])],
               "cells": n, "labels": label_audit,
               "train_positive_cell_days": int(y_train.sum()),
               "calibration_positive_cell_days": int(y_cal.sum()),
               "test_positive_cell_days": int(y_test.sum()),
               "calibration_method": "intercept-only logistic correction fitted on Aug-Sep; monotonic rain response preserved",
               "calibration_log_odds_shift": round(bias, 4),
               "test_raw": evaluation(raw_test, y_test),
               "test_calibrated": evaluation(p_test, y_test),
               "test_constant_prevalence_brier": round(float(y_test.mean() * (1 - y_test.mean())), 6),
               "test_cell_rank": rank_metrics(expected_by_cell, observed_by_cell),
               "test_historical_cell_rank": rank_metrics(historical_by_cell, observed_by_cell),
               "historical_baseline": "positive report-footprint days per cell from 2025-01-01 through 2025-09-30; no held-out labels or rain used",
               "test_spatial_deciles": spatial_bins,
               "test_months": daily_test,
               "test_rain_strata": rain_strata,
               "scenario_interpretation": "retrospective October day with chosen hourly rain and zero preceding 72 h; reporting proxy, not physical inundation"}
    # Set up an October scenario using only history available by 30 September.
    def percentiles(values: np.ndarray) -> np.ndarray:
        ranks = 100 * (rankdata(values, method="average") - 1) / (len(values) - 1)
        return np.where(values > 0, ranks, 0)

    historical_rank = percentiles(historical_by_cell)
    model_rank = percentiles(expected_by_cell)
    observed_rank = percentiles(observed_by_cell)
    comparison_rows = [
        {"cell_id": cid,
         "historical_positive_days_jan_sep": int(historical_by_cell[i]),
         "model_expected_positive_days_oct_dec": round(float(expected_by_cell[i]), 5),
         "observed_positive_days_oct_dec": int(observed_by_cell[i]),
         "historical_rank_percentile": round(float(historical_rank[i]), 2),
         "model_rank_percentile": round(float(model_rank[i]), 2),
         "observed_rank_percentile": round(float(observed_rank[i]), 2)}
        for i, cid in enumerate(ids)]
    write_csv(OUTPUT / f"{prefix}_historical_comparison.csv", comparison_rows)
    write_historical_comparison_map(comparison_rows, prefix)

    rows = [{"cell_id": cell["cell_id"], "centre_lon": cell["centre_lon"],
             "centre_lat": cell["centre_lat"], "flood_probability": None,
             "proxy_definition": description,
             "proxy_positive_days_2025": len(labels.get(cell["cell_id"], set()))}
            for cell in sites]
    for name, (rate, duration) in SCENARIOS.items():
        matrix = np.asarray([vector(cell, SCENARIO_DAY, scenario_rain(rate, duration),
                                    prior[cell["cell_id"]],
                                    sum(count for month, count in activity.get(cell["cell_id"], {}).items()
                                        if month <= "2025-09"), False)
                             for cell in sites], dtype=np.float32)
        p = apply_calibration(fitted.predict_proba(matrix)[:, 1], bias)
        for row, prob in zip(rows, p):
            row[f"proxy_flood_probability_{name}"] = round(float(prob), 6)
    write_csv(OUTPUT / f"{prefix}_probability_scenarios.csv", rows)
    positive_rows = [{"date_bangkok": str(day), "cell_id": cid}
                     for cid in sorted(labels) for day in sorted(labels[cid])]
    write_csv(OUTPUT / f"{prefix}_positive_cell_days_2025.csv", positive_rows)
    scale_max = max(0.01, math.ceil(max(
        row[f"proxy_flood_probability_{name}"] for row in rows for name in SCENARIOS) * 100) / 100)
    write_probability_map(rows, scale_max, prefix, description)
    (OUTPUT / f"{prefix}_probability_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("connected", "buffer"), default="connected")
    main(parser.parse_args().mode)
