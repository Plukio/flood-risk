"""Temporal backtest of cell-level *flood complaint* likelihood in Bang Kapi.

Run: .venv/bin/python flood_bangkapi_surface_model.py

The target is a dated Traffy flood complaint in a 250 m cell, not verified
surface flooding. A zero is no complaint observed, not a dry-road label.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from shapely.geometry import Point
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from flood_bangkapi_surface_grid import GRID_M, TO_UTM, district_polygon, grid_spec


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "artifacts" / "flood_bangkapi_surface" / "surface_cells_250m.csv"
TRAFFY = ROOT / "data" / "flood_bangkapi_surface" / "traffy_flood_2025_minimal.csv"
ACTIVITY = ROOT / "data" / "flood_bangkapi_surface" / "traffy_all_activity_2025_cell_month.csv"
RAIN = ROOT / "data" / "flood_bangkok_transfer" / "rain_1006_2021_2025.json"
OUTPUT = ROOT / "artifacts" / "flood_bangkapi_surface"
TRAIN_END = date(2025, 8, 31)
DIRECT_TYPES = {"น้ำท่วม", "BEST -> น้ำท่วมขัง",
                "น้ำท่วม -> ขอให้แก้ไขปัญหาน้ำท่วม",
                "น้ำท่วม -> ขอให้สูบน้ำออก"}
RAIN_FEATURES = ("rain_today_mm", "rain_peak_1h_mm", "rain_peak_3h_mm",
                 "rain_wet_hours", "rain_prev24h_mm", "rain_prev72h_mm",
                 "rain_today_prev24h_mm")
BASE_FEATURES = ("elev_p10_m", "elev_median_m", "relative_to_500m_mean_m",
                 "osm_road_m", "osm_waterway_m", "osm_building_footprint_fraction",
                 "osm_amenity_count", "osm_essential_amenity_count",
                 "nearest_pump_m", "pumps_within_1km", "nearest_gate_m",
                 "district_coverage_fraction", "prior_flood_report_days",
                 "prior_all_complaints", "month_sin", "month_cos")
SATELLITE_FEATURES = ("s2_ndvi_median", "s2_ndbi_median",
                      "s2_low_veg_land_fraction", "s2_clear_fraction")


def cells() -> list[dict]:
    with INPUT.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 450:
        raise ValueError("Bang Kapi feature grid looks incomplete")
    return rows


def complaint_dates(valid_cells: set[str]) -> dict[str, set[date]]:
    polygon = district_polygon()
    grid = grid_spec(polygon)
    result: dict[str, set[date]] = defaultdict(set)
    with TRAFFY.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["type"] not in DIRECT_TYPES:
                continue
            x, y = TO_UTM.transform(float(row["lon"]), float(row["lat"]))
            if not polygon.covers(Point(x, y)):
                continue
            grid_row = math.floor((grid["north"] - y) / GRID_M)
            grid_col = math.floor((x - grid["west"]) / GRID_M)
            cell = f"bk:{grid_row:02d}:{grid_col:02d}"
            if cell in valid_cells:
                result[cell].add(date.fromisoformat(row["timestamp"][:10]))
    return result


def activity_by_cell() -> dict[str, Counter]:
    result: dict[str, Counter] = defaultdict(Counter)
    with ACTIVITY.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            cell = f"bk:{int(row['grid_row']):02d}:{int(row['grid_col']):02d}"
            result[cell][row["month"]] += int(row["all_complaints"])
    return result


def rainfall_by_date(start_day: date = date(2025, 1, 1),
                     end_day: date = date(2025, 12, 31),
                     rain_path: Path = RAIN) -> dict[date, dict]:
    if end_day < start_day:
        raise ValueError("Rain date range is reversed")
    raw = json.loads(rain_path.read_text())
    if raw.get("timezone") != "Asia/Bangkok":
        raise ValueError("Historical rain is not stamped in Bangkok local time")
    hours: dict[date, list[float]] = defaultdict(list)
    for stamp, mm in zip(raw["hourly"]["time"], raw["hourly"]["precipitation"]):
        day = date.fromisoformat(stamp[:10])
        if start_day - timedelta(days=3) <= day <= end_day:
            hours[day].append(float(mm) if mm is not None else math.nan)
    result = {}
    for day in (start_day + timedelta(days=i)
                for i in range((end_day - start_day).days + 1)):
        today = hours.get(day, [])
        previous = hours.get(day - timedelta(days=1), [])
        prev72 = [mm for lag in (1, 2, 3)
                  for mm in hours.get(day - timedelta(days=lag), [])]
        if (len(today) != 24 or len(previous) != 24 or len(prev72) != 72
                or not all(math.isfinite(v) for v in today + previous + prev72)):
            raise ValueError(f"Missing daily rain inputs at {day}")
        wet = sum(mm >= 0.5 for mm in today)
        total, prev24 = sum(today), sum(previous)
        result[day] = {"rain_today_mm": total,
                       "rain_peak_1h_mm": max(today),
                       "rain_peak_3h_mm": max(sum(today[i:i + 3]) for i in range(22)),
                       "rain_wet_hours": wet,
                       "rain_prev24h_mm": prev24,
                       "rain_prev72h_mm": sum(prev72),
                       "rain_today_prev24h_mm": total + prev24}
    return result


def value(row: dict, name: str) -> float:
    raw = row.get(name)
    return float(raw) if raw not in (None, "") else math.nan


def vector(cell: dict, day: date, weather: dict, prior_flood_days: int,
           prior_activity: int, with_satellite: bool) -> list[float]:
    static = [value(cell, key) for key in BASE_FEATURES[:12]]
    static += [float(prior_flood_days), float(prior_activity),
               math.sin(2 * math.pi * day.month / 12),
               math.cos(2 * math.pi * day.month / 12)]
    if with_satellite:
        static += [value(cell, key) for key in SATELLITE_FEATURES]
    return static + [math.log1p(max(0.0, weather[key])) for key in RAIN_FEATURES]


def model(feature_count: int) -> HistGradientBoostingClassifier:
    monotone = [0] * (feature_count - len(RAIN_FEATURES)) + [1] * len(RAIN_FEATURES)
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=120,
        max_leaf_nodes=8, min_samples_leaf=150, l2_regularization=20.0,
        max_bins=64, early_stopping=False, monotonic_cst=monotone,
        random_state=7)


def pairwise_agreement(predicted: np.ndarray, observed: np.ndarray) -> dict:
    correct = 0.0
    comparable = 0
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            if observed[i] == observed[j]:
                continue
            comparable += 1
            signed = (float(predicted[i]) - float(predicted[j])) * (
                int(observed[i]) - int(observed[j]))
            correct += 1 if signed > 0 else 0.5 if signed == 0 else 0
    return {"comparable_pairs": comparable,
            "pairwise_agreement": round(correct / comparable, 4) if comparable else None}


def rank_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
    result = spearmanr(predicted, observed)
    return {"spearman": round(float(result.statistic), 4),
            "spearman_p_unadjusted": round(float(result.pvalue), 4),
            **pairwise_agreement(predicted, observed)}


def scenario_rain(rate_mm_h: float, duration_h: int) -> dict:
    total = rate_mm_h * duration_h
    return {"rain_today_mm": total, "rain_peak_1h_mm": rate_mm_h,
            "rain_peak_3h_mm": rate_mm_h * min(3, duration_h),
            "rain_wet_hours": duration_h, "rain_prev24h_mm": 0,
            "rain_prev72h_mm": 0, "rain_today_prev24h_mm": total}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_scenario_map(scenario_rows: list[dict]) -> None:
    by_cell = {row["cell_id"]: row for row in scenario_rows}
    geojson = json.loads((OUTPUT / "surface_cells_250m.geojson").read_text())
    anchors = json.loads((OUTPUT / "bma_report_anchors.geojson").read_text())
    for feature in geojson["features"]:
        feature["properties"].update(by_cell[feature["properties"]["cell_id"]])
    (OUTPUT / "surface_cell_complaint_scenarios.geojson").write_text(json.dumps(
        geojson, ensure_ascii=False))
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bang Kapi cell complaint ranking scenarios</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{margin:0;font:14px system-ui,sans-serif;color:#1b2a32}header{background:#12344b;color:white;padding:12px 16px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}header strong{font-size:17px}header small{opacity:.9}select{font:inherit;padding:5px;border:0;border-radius:3px}#map{height:calc(100vh - 76px);min-height:480px}.legend{background:white;padding:10px;box-shadow:0 1px 8px #5555;border-radius:5px;max-width:260px;line-height:1.45}.bar{height:12px;background:linear-gradient(90deg,#fff1bd,#e3944a,#9b2429);margin:6px 0}</style></head><body>
<header><strong>Bang Kapi · 250 m cells</strong><label>Rain scenario <select id="scenario">
<option value="rank_percentile_base_4mm_h_2h">4 mm/h for 2 h</option>
<option value="rank_percentile_double_intensity_8mm_h_2h">8 mm/h for 2 h</option>
<option value="rank_percentile_double_duration_4mm_h_4h">4 mm/h for 4 h</option>
<option value="rank_percentile_both_8mm_h_4h">8 mm/h for 4 h</option>
</select></label><small>Relative rank for a Traffy flood complaint on one September day. This is not P(surface flooding).</small></header>
<div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const grid=__GRID_DATA__,anchors=__ANCHORS_DATA__;
const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
let active='rank_percentile_base_4mm_h_2h';
function color(p){const t=Math.max(0,Math.min(1,Number(p)/100));return `hsl(${48-46*t},${85-20*t}%,${93-56*t}%)`}
function popup(p){return `<b>${p.cell_id}</b><br>Complaint score rank: <b>${Number(p[active]).toFixed(1)} percentile</b><br>2025 observed complaint days: ${p.traffy_2025_report_days}<br>Elevation: ${p.elev_median_m} m<br>Pre-storm NDVI: ${p.s2_ndvi_median}<br>Road length: ${p.osm_road_m} m<br><b>Actual flood probability: unavailable</b>`}
const cells=L.geoJSON(grid,{style:f=>({color:'#5a5050',weight:.25,fillColor:color(f.properties[active]),fillOpacity:.8}),onEachFeature:(f,l)=>{l.bindPopup(popup(f.properties));}}).addTo(map);
L.geoJSON(anchors,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:4,color:'#272b52',fillColor:'#4451a0',fillOpacity:.8,weight:1}),onEachFeature:(f,l)=>l.bindPopup(`<b>${f.properties.label}</b><br>BMA 2025 road-report days: ${f.properties.report_days_2025}`)}).addTo(map);
map.fitBounds(cells.getBounds());
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const div=L.DomUtil.create('div','legend');div.innerHTML='<b>Relative complaint score rank</b><div class="bar"></div><small>Paler = lower rank, darker = higher rank.<br>Rank is relative to 527 Bang Kapi cells. The model underpredicted complaint counts and does not measure P(flood).</small>';return div};legend.addTo(map);
document.getElementById('scenario').addEventListener('change',e=>{active=e.target.value;cells.setStyle(f=>({fillColor:color(f.properties[active])}));cells.eachLayer(layer=>layer.setPopupContent(popup(layer.feature.properties)));});
</script></body></html>"""
    html = html.replace("__GRID_DATA__", json.dumps(geojson, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__ANCHORS_DATA__", json.dumps(anchors, ensure_ascii=False).replace("</", "<\\/"))
    (OUTPUT / "surface_cell_complaint_scenario_map.html").write_text(html)


def main() -> None:
    sites = cells()
    ids = [row["cell_id"] for row in sites]
    labels = complaint_dates(set(ids))
    activity = activity_by_cell()
    weather = rainfall_by_date()
    days = sorted(weather)
    train_days = [day for day in days if day <= TRAIN_END]
    test_days = [day for day in days if day > TRAIN_END]
    n = len(sites)
    prior_days = Counter()
    train_x_base, train_x_sat, train_y = [], [], []
    test_x_base, test_x_sat, test_y = [], [], []
    for day in days:
        training = day <= TRAIN_END
        month = day.strftime("%Y-%m")
        for cell in sites:
            cell_id = cell["cell_id"]
            all_activity = sum(count for label, count in activity.get(cell_id, {}).items()
                               if label < month and label <= "2025-08")
            features_base = vector(cell, day, weather[day], prior_days[cell_id],
                                   all_activity, False)
            features_sat = vector(cell, day, weather[day], prior_days[cell_id],
                                  all_activity, True)
            target = int(day in labels.get(cell_id, set()))
            if training:
                train_x_base.append(features_base)
                train_x_sat.append(features_sat)
                train_y.append(target)
            else:
                test_x_base.append(features_base)
                test_x_sat.append(features_sat)
                test_y.append(target)
        if training:
            for cell_id in ids:
                prior_days[cell_id] += int(day in labels.get(cell_id, set()))
    X_train_base = np.asarray(train_x_base, dtype=np.float32)
    X_train_sat = np.asarray(train_x_sat, dtype=np.float32)
    X_test_base = np.asarray(test_x_base, dtype=np.float32)
    X_test_sat = np.asarray(test_x_sat, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.uint8)
    y_test = np.asarray(test_y, dtype=np.uint8)
    observed = y_test.reshape(len(test_days), n).sum(axis=0)
    historical = np.asarray([prior_days[cell_id] for cell_id in ids], dtype=float)
    metrics = {"recent_flood_complaint_days": rank_metrics(historical, observed)}
    predictions = {}
    fitted = {}
    for name, train, test in (("without_satellite", X_train_base, X_test_base),
                              ("with_satellite", X_train_sat, X_test_sat)):
        fitted[name] = model(train.shape[1]).fit(train, y_train)
        daily = fitted[name].predict_proba(test)[:, 1]
        expected = daily.reshape(len(test_days), n).sum(axis=0)
        predictions[name] = expected
        monthly = {}
        for month in sorted({day.strftime("%Y-%m") for day in test_days}):
            positions = [i for i, day in enumerate(test_days)
                         if day.strftime("%Y-%m") == month]
            observed_month = y_test.reshape(len(test_days), n)[positions].sum(axis=0)
            expected_month = daily.reshape(len(test_days), n)[positions].sum(axis=0)
            monthly[month] = {"positive_cell_days": int(observed_month.sum()),
                              **rank_metrics(expected_month, observed_month)}
        metrics[name] = {**rank_metrics(expected, observed),
                         "test_daily_average_precision": round(float(
                             average_precision_score(y_test, daily)), 5),
                         "test_daily_brier": round(float(brier_score_loss(y_test, daily)), 6),
                         "mean_predicted_daily_complaint_chance": round(float(np.mean(daily)), 6),
                         "predicted_positive_cell_days": round(float(np.sum(daily)), 1),
                         "per_test_month": monthly}
        print(name, metrics[name], flush=True)

    ranking = []
    for i, cell in enumerate(sites):
        ranking.append({"cell_id": cell["cell_id"],
                        "centre_lon": cell["centre_lon"], "centre_lat": cell["centre_lat"],
                        "boundary_sliver": cell["boundary_sliver"],
                        "observed_sep_dec_report_days": int(observed[i]),
                        "prior_jan_aug_report_days": int(historical[i]),
                        "predicted_sep_dec_expected_report_days_without_satellite": round(float(predictions["without_satellite"][i]), 5),
                        "predicted_sep_dec_expected_report_days_with_satellite": round(float(predictions["with_satellite"][i]), 5)})
    ranking.sort(key=lambda row: -row["predicted_sep_dec_expected_report_days_without_satellite"])
    write_csv(OUTPUT / "surface_cell_forward_ranking_2025.csv", ranking)

    # The explicit scenarios are hypothetical day inputs. They return a model
    # estimate of a complaint, never the chance that an arbitrary surface floods.
    scenario_specs = {"base_4mm_h_2h": (4.0, 2),
                      "double_intensity_8mm_h_2h": (8.0, 2),
                      "double_duration_4mm_h_4h": (4.0, 4),
                      "both_8mm_h_4h": (8.0, 4)}
    scenario_day = date(2025, 9, 15)
    scenario_rows = [{"cell_id": row["cell_id"],
                      "centre_lon": row["centre_lon"], "centre_lat": row["centre_lat"],
                      "flood_probability": None,
                      "outcome": "at least one Traffy flood complaint on a day"}
                     for row in sites]
    for name, (rate, duration) in scenario_specs.items():
        matrix = np.asarray([vector(cell, scenario_day, scenario_rain(rate, duration),
                                    prior_days[cell["cell_id"]],
                                    sum(count for month, count in activity.get(cell["cell_id"], {}).items()
                                        if month <= "2025-08"), False)
                             for cell in sites], dtype=np.float32)
        scores = fitted["without_satellite"].predict_proba(matrix)[:, 1]
        percentiles = 100 * (rankdata(scores, method="average") - 1) / (n - 1)
        for row, score, percentile in zip(scenario_rows, scores, percentiles):
            row[f"p_complaint_{name}"] = round(float(score), 6)
            row[f"rank_percentile_{name}"] = round(float(percentile), 1)
    for row in scenario_rows:
        base = row["p_complaint_base_4mm_h_2h"]
        row["intensity_delta"] = round(row["p_complaint_double_intensity_8mm_h_2h"] - base, 6)
        row["duration_delta"] = round(row["p_complaint_double_duration_4mm_h_4h"] - base, 6)
    write_csv(OUTPUT / "surface_cell_complaint_scenarios.csv", scenario_rows)
    write_scenario_map(scenario_rows)

    audit = {"unit": "250 m cell-day", "target": "at least one direct-category Traffy flood complaint",
             "not_target": "verified surface flooding",
             "training_dates": [str(train_days[0]), str(train_days[-1])],
             "test_dates": [str(test_days[0]), str(test_days[-1])],
             "cells": n, "train_cell_days": len(y_train), "test_cell_days": len(y_test),
             "train_positive_cell_days": int(y_train.sum()),
             "test_positive_cell_days": int(y_test.sum()),
             "test_daily_prevalence": round(float(y_test.mean()), 6),
             "rain_source": "Open-Meteo district-point historical hourly precipitation; no cell-scale rain variation",
             "metrics": metrics,
             "scenario": {"month": "September", "antecedent_rain_mm": 0,
                          "selected_model": "without_satellite: better held-out cell rank",
                          "inputs": {name: {"rate_mm_h": rate, "duration_h": duration}
                                     for name, (rate, duration) in scenario_specs.items()},
                          "interpretation": "hypothetical complaint-model response; not calibrated flood probability"},
             "limits": ["An absent complaint is not evidence of a dry cell.",
                        "Complaint location and timestamp may differ from flood location/onset.",
                        "The all-complaint activity feature proxies reporting propensity, but does not identify it.",
                        "Satellite and OSM features predate test; Traffy 2025 archive was published later."]}
    (OUTPUT / "surface_model_metrics.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
