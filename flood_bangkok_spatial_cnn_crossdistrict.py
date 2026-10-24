"""Check whether other Bangkok districts help a terrain-and-rain spatial CNN.

Train a 250 m-grid CNN on Bang Kapi 2025 January-July with or without five
other districts. Select/check on Bang Kapi August-September, then evaluate
Bang Kapi October-December. All areas use the same 100 m Traffy report proxy.

Run: .venv/bin/python flood_bangkok_spatial_cnn_crossdistrict.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter
from scipy.special import expit
from shapely.geometry import Point
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkapi_proxy_probability import apply_calibration, calibrate_intercept, evaluation
from flood_bangkapi_spatial_cnn import SpatialCNN, logits_for, percentiles
from flood_bangkapi_surface_grid import GRID_M, TO_UTM, cell_box, grid_spec, pump_points
from flood_bangkapi_surface_model import (
    DIRECT_TYPES, OUTPUT, RAIN_FEATURES, TRAFFY, rainfall_by_date, rank_metrics, write_csv,
)


DISTRICTS = ("1006", "1030", "1032", "1034", "1044", "1045")
SEED = 7
CHANNELS = 17
POSITIVE_WEIGHT = 30.0
TRAIN_DAYS = 212  # 1 January through 31 July 2025
CAL_DAYS = 61      # 1 August through 30 September 2025
MAX_EPOCHS = 40
PATIENCE = 8
ARTIFACT = "surface_crossdistrict_terrain_cnn_2025"


def report_rows() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    with TRAFFY.open(encoding="utf-8-sig", newline="") as file:
        for item in csv.DictReader(file):
            if item["type"] in DIRECT_TYPES:
                rows[item["district"]].append(item)
    return rows


def static_grid(district: dict, stations: list[dict]):
    spec = grid_spec(district["polygon"])
    reference_cells = None
    if district["code"] == "1006":
        with (OUTPUT / "surface_cells_250m.csv").open(encoding="utf-8-sig", newline="") as file:
            reference_cells = {tuple(map(int, row["cell_id"].split(":")[1:]))
                               for row in csv.DictReader(file)}
    grid_shape = spec["rows"], spec["cols"]
    elevation = np.full(grid_shape, np.nan, dtype=np.float32)
    terrain_path = TRANSFER_CACHE / f"terrain_{district['code']}.tif"
    with rasterio.open(terrain_path) as source:
        reproject(source=rasterio.band(source, 1), destination=elevation,
                  src_transform=source.transform, src_crs=source.crs,
                  dst_transform=from_origin(spec["west"], spec["north"], GRID_M, GRID_M),
                  dst_crs=source.crs, dst_nodata=np.nan, resampling=Resampling.average)
    if not np.isfinite(elevation).any():
        raise ValueError(f"No elevation in {district['code']}")
    median = float(np.nanmedian(elevation))
    elevation = np.where(np.isfinite(elevation), elevation, median)
    local = uniform_filter(elevation, size=3, mode="nearest")
    dy, dx = np.gradient(elevation, GRID_M)
    slope = np.hypot(dx, dy)
    valid = np.zeros(grid_shape, dtype=np.float32)
    static = np.zeros((7, *grid_shape), dtype=np.float32)
    cells = {}
    pumps = [s for s in stations if "สูบ" in s["type"] or "สูบ" in s["name"]]
    gates = [s for s in stations if "ประตู" in s["type"] or "ประตู" in s["name"]]
    for row in range(spec["rows"]):
        for col in range(spec["cols"]):
            if reference_cells is not None and (row, col) not in reference_cells:
                continue
            box = cell_box(spec, row, col)
            coverage = box.intersection(district["polygon"]).area / (GRID_M * GRID_M)
            if coverage <= 0:
                continue
            valid[row, col] = 1
            cells[(row, col)] = box.intersection(district["polygon"])
            x, y = box.centroid.x, box.centroid.y
            pump_distances = [math.hypot(s["x"] - x, s["y"] - y) for s in pumps]
            gate_distances = [math.hypot(s["x"] - x, s["y"] - y) for s in gates]
            static[:, row, col] = [elevation[row, col], elevation[row, col] - local[row, col],
                                   slope[row, col], coverage,
                                   min(pump_distances) if pump_distances else 10000,
                                   sum(distance <= 1000 for distance in pump_distances),
                                   min(gate_distances) if gate_distances else 10000]
    if reference_cells is not None and set(cells) != reference_cells:
        raise ValueError("Bang Kapi cells differ from the established 250 m grid")
    return spec, static, valid, cells


def report_labels(district: dict, spec: dict, cells: dict, report_list: list[dict],
                  year: int = 2025):
    labels: dict[tuple[int, int], set[date]] = defaultdict(set)
    retained = 0
    for item in report_list:
        x, y = TO_UTM.transform(float(item["lon"]), float(item["lat"]))
        if not district["polygon"].covers(Point(x, y)):
            continue
        day = date.fromisoformat(item["timestamp"][:10])
        if day.year != year:
            continue
        retained += 1
        footprint = Point(x, y).buffer(100)
        left = max(0, math.floor((x - 100 - spec["west"]) / GRID_M))
        right = min(spec["cols"] - 1, math.floor((x + 100 - spec["west"]) / GRID_M))
        top = max(0, math.floor((spec["north"] - (y + 100)) / GRID_M))
        bottom = min(spec["rows"] - 1, math.floor((spec["north"] - (y - 100)) / GRID_M))
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                if (row, col) in cells and cells[(row, col)].intersects(footprint):
                    labels[(row, col)].add(day)
    return labels, {"direct_reports_in_district": retained,
                    f"positive_cell_days_{year}": sum(len(days) for days in labels.values()),
                    f"positive_cells_{year}": len(labels)}


def district_data(district: dict, reports: dict, stations: list[dict], height: int, width: int):
    spec, static, valid, cells = static_grid(district, stations)
    labels, audit = report_labels(district, spec, cells, reports.get(district["name_th"], []))
    rain = rainfall_by_date(rain_path=TRANSFER_CACHE / f"rain_{district['code']}_2021_2025.json")
    days = sorted(rain)
    if len(days) != 365:
        raise ValueError("Expected full 2025 daily rain")
    data = np.zeros((365, CHANNELS + 1, height, width), dtype=np.float32)
    target = np.zeros((365, 1, height, width), dtype=np.float32)
    h, w = valid.shape
    data[:, CHANNELS, :h, :w] = valid
    data[:, :7, :h, :w] = static
    prior = Counter()
    addresses = list(cells)
    for t, day in enumerate(days):
        seasonal = [math.sin(2 * math.pi * day.month / 12),
                    math.cos(2 * math.pi * day.month / 12)]
        weather = [math.log1p(max(0, rain[day][key])) for key in RAIN_FEATURES]
        for row, col in addresses:
            data[t, 7:CHANNELS, row, col] = [prior[(row, col)], *seasonal, *weather]
            target[t, 0, row, col] = int(day in labels.get((row, col), set()))
        if day <= date(2025, 9, 30):
            for address in addresses:
                prior[address] += int(day in labels.get(address, set()))
    rr, cc = np.asarray([r for r, _ in addresses]), np.asarray([c for _, c in addresses])
    return data, target, (rr, cc), audit


def normalize(train: np.ndarray, *others: np.ndarray, channel_count: int = CHANNELS):
    stats = []
    if train.shape[1] != channel_count + 1 or any(
            data.shape[1] != train.shape[1] for data in others):
        raise ValueError("Feature and mask channel counts disagree")
    valid = train[:, channel_count] > 0
    for channel in range(channel_count):
        values = train[:, channel][valid]
        values = values[np.isfinite(values)]
        mean = float(values.mean())
        std = max(float(values.std()), 1e-4)
        stats.append((mean, std))
    result = []
    for data in (train, *others):
        copy = data.copy()
        mask = copy[:, channel_count]
        for channel, (mean, std) in enumerate(stats):
            copy[:, channel] = np.clip((copy[:, channel] - mean) / std, -5, 5) * mask
        result.append(copy)
    return result


def fit(train_x: np.ndarray, train_y: np.ndarray, cal_x: np.ndarray, cal_y: np.ndarray,
        test_x: np.ndarray, test_y: np.ndarray, rr: np.ndarray, cc: np.ndarray,
        test_rr: np.ndarray | None = None, test_cc: np.ndarray | None = None,
        adapt_cal: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
        training_sample_weights: np.ndarray | None = None,
        daily_output_path: Path | None = None,
        checkpoint_output_path: Path | None = None,
        masked_channels: tuple[int, ...] = ()):
    if test_rr is None:
        test_rr = rr
    if test_cc is None:
        test_cc = cc
    channel_count = train_x.shape[1] - 1
    if any(data.shape[1] != channel_count + 1 for data in (cal_x, test_x)):
        raise ValueError("Feature channel counts disagree")
    torch.manual_seed(SEED)
    if masked_channels:
        if any(channel < 0 or channel >= channel_count for channel in masked_channels):
            raise ValueError("Only feature channels may be masked")
        cal_x = cal_x.copy()
        cal_x[:, masked_channels] = 0
        test_x = test_x.copy()
        test_x[:, masked_channels] = 0
        if adapt_cal is not None:
            adapt_x, adapt_y, adapt_rr, adapt_cc = adapt_cal
            adapt_x = adapt_x.copy()
            adapt_x[:, masked_channels] = 0
            adapt_cal = (adapt_x, adapt_y, adapt_rr, adapt_cc)
    model = SpatialCNN(channel_count + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    pos_weight = torch.tensor(POSITIVE_WEIGHT)
    rng = np.random.default_rng(SEED)
    best_ap, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        if training_sample_weights is None:
            order = rng.permutation(len(train_x))
        else:
            if len(training_sample_weights) != len(train_x):
                raise ValueError("Training sample weights differ from training days")
            probabilities = training_sample_weights / training_sample_weights.sum()
            order = rng.choice(len(train_x), size=len(train_x), replace=True,
                               p=probabilities)
        for start in range(0, len(order), 12):
            ids = order[start:start + 12]
            x = torch.from_numpy(train_x[ids])
            if masked_channels:
                x[:, masked_channels] = 0
            y = torch.from_numpy(train_y[ids])
            mask = x[:, channel_count:channel_count + 1]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(x), y, pos_weight=pos_weight, reduction="none")
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
        cal_prob = expit(logits_for(model, cal_x)[:, 0, rr, cc]).ravel()
        cal_truth = cal_y[:, 0, rr, cc].ravel()
        ap = float(average_precision_score(cal_truth, cal_prob))
        if ap > best_ap + 1e-6:
            best_ap, best_epoch = ap, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"train_samples={len(train_x)} epoch={epoch} cal_AP={ap:.5f} best={best_ap:.5f}",
                  flush=True)
        if stale >= PATIENCE and epoch >= 10:
            break
    model.load_state_dict(best_state)
    if checkpoint_output_path is not None:
        checkpoint_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "selected_epoch": best_epoch},
                   checkpoint_output_path)
    raw_cal = expit(logits_for(model, cal_x)[:, 0, rr, cc]).ravel()
    bias = calibrate_intercept(raw_cal, cal_y[:, 0, rr, cc].ravel())
    raw_test = expit(logits_for(model, test_x)[:, 0, test_rr, test_cc])
    calibrated = apply_calibration(raw_test, bias)
    truth = test_y[:, 0, test_rr, test_cc].astype(np.uint8)
    expected = calibrated.sum(axis=0)
    observed = truth.sum(axis=0)
    if daily_output_path is not None:
        daily_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(daily_output_path, source_calibrated=calibrated,
                            observed=truth, raw_test=raw_test)
    result = {"train_district_days": len(train_x), "selected_epoch": best_epoch,
              "calibration_AP_for_selection": round(best_ap, 5),
              "test_daily": evaluation(calibrated.ravel(), truth.ravel()),
              "test_cell_rank": rank_metrics(expected, observed),
              "test_observed_positive_cell_days": int(truth.sum()),
              "calibration_log_odds_shift": round(bias, 4)}
    if adapt_cal is None:
        return result, expected, observed
    adapt_x, adapt_y, adapt_rr, adapt_cc = adapt_cal
    raw_adapt = expit(logits_for(model, adapt_x)[:, 0, adapt_rr, adapt_cc]).ravel()
    adapt_truth = adapt_y[:, 0, adapt_rr, adapt_cc].ravel()
    adapt_bias = calibrate_intercept(raw_adapt, adapt_truth)
    adapted = apply_calibration(raw_test, adapt_bias)
    adapted_expected = adapted.sum(axis=0)
    if daily_output_path is not None:
        np.savez_compressed(daily_output_path, source_calibrated=calibrated,
                            target_recalibrated=adapted, observed=truth,
                            raw_test=raw_test)
    result["target_district_recalibration"] = {
        "calibration_positive_cell_days": int(adapt_truth.sum()),
        "calibration_raw_predicted_positive_cell_days": round(float(raw_adapt.sum()), 1),
        "calibration_AP": round(float(average_precision_score(adapt_truth, raw_adapt)), 5),
        "calibration_log_odds_shift": round(adapt_bias, 4),
        "test_daily": evaluation(adapted.ravel(), truth.ravel()),
        "test_cell_rank": rank_metrics(adapted_expected, observed)}
    return result, expected, observed, adapted_expected


def write_comparison_map(rows: list[dict]) -> None:
    geojson = json.loads((OUTPUT / "surface_cells_250m.geojson").read_text())
    by_cell = {row["cell_id"]: row for row in rows}
    for feature in geojson["features"]:
        feature["properties"].update(by_cell[feature["properties"]["cell_id"]])
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bang Kapi cross-district CNN comparison</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,sans-serif;color:#20313b}header{background:#12344b;color:white;padding:12px 16px;display:flex;gap:15px;align-items:center;flex-wrap:wrap}header strong{font-size:17px}header small{max-width:690px}select{font:inherit;padding:5px;border:0;border-radius:3px}#map{height:calc(100vh - 84px);min-height:480px}.legend{background:white;padding:10px;box-shadow:0 1px 8px #5555;border-radius:5px;line-height:1.45}.bar{height:12px;background:linear-gradient(90deg,#fff1bd,#e3944a,#9b2429);margin:7px 0}</style></head><body><header><strong>Bang Kapi · other-district training</strong><label>Show <select id="view"><option value="historical_rank_percentile">Historical Jan–Sep</option><option value="bk_only_cnn_rank_percentile">Bang Kapi only CNN</option><option value="six_district_cnn_rank_percentile">Six-district CNN</option><option value="observed_rank_percentile">Observed Oct–Dec</option></select></label><small>Rank among 527 cells for report-footprint-positive days. October–December outcomes were used for retrospective evaluation only. These are complaint-proxy ranks, not verified flood probabilities.</small></header><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const grid=__GRID__;const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);let active='six_district_cnn_rank_percentile';
function color(p){const t=Math.max(0,Math.min(1,Number(p)/100));return `hsl(${48-46*t},${85-20*t}%,${93-56*t}%)`}
function popup(p){return `<b>${p.cell_id}</b><br>Jan–Sep report-positive days: <b>${p.historical_positive_days_jan_sep}</b><br>Bang Kapi only expected Oct–Dec days: <b>${p.bk_only_cnn_expected_days_oct_dec}</b><br>Six-district expected Oct–Dec days: <b>${p.six_district_cnn_expected_days_oct_dec}</b><br>Observed Oct–Dec report-positive days: <b>${p.observed_days_oct_dec}</b><br>Displayed rank: <b>${Number(p[active]).toFixed(1)}</b> percentile<br><small>100 m report buffer proxy; not verified water extent.</small>`}
const cells=L.geoJSON(grid,{style:f=>({color:'#5a5050',weight:.25,fillColor:color(f.properties[active]),fillOpacity:.83}),onEachFeature:(f,l)=>l.bindPopup(popup(f.properties))}).addTo(map);map.fitBounds(cells.getBounds());
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Cell rank percentile</b><div class="bar"></div><small>0 → 100; darker means higher rank.<br>Zero-day cells are rank 0; ties share a rank.</small>';return d};legend.addTo(map);
document.getElementById('view').addEventListener('change',e=>{active=e.target.value;cells.setStyle(f=>({fillColor:color(f.properties[active])}));cells.eachLayer(l=>l.setPopupContent(popup(l.feature.properties)));});
</script></body></html>"""
    (OUTPUT / f"{ARTIFACT}_comparison_map.html").write_text(
        html.replace("__GRID__", json.dumps(geojson, ensure_ascii=False).replace("</", "<\\/")))


def main() -> None:
    torch.set_num_threads(4)
    districts = {d["code"]: d for d in load_districts() if d["code"] in DISTRICTS}
    if len(districts) != len(DISTRICTS):
        raise ValueError("Missing selected BMA district")
    specs = {code: grid_spec(districts[code]["polygon"]) for code in DISTRICTS}
    height = max(spec["rows"] for spec in specs.values())
    width = max(spec["cols"] for spec in specs.values())
    reports = report_rows()
    stations = pump_points()
    district_arrays = {}
    audits = {}
    for code in DISTRICTS:
        data, target, addresses, audit = district_data(
            districts[code], reports, stations, height, width)
        district_arrays[code] = data, target, addresses
        audits[code] = {"name": districts[code]["name_en"], "grid_shape": [
            specs[code]["rows"], specs[code]["cols"]], **audit}
        print(f"district {code} {districts[code]['name_en']}: {audit}", flush=True)
    bk_x, bk_y, (rr, cc) = district_arrays["1006"]
    if bk_y[:, 0, rr, cc].sum() != 812:
        raise ValueError("Bang Kapi 100 m labels differ from the prior baseline")
    results = {}
    predictions = {}
    for name, codes in (("bang_kapi_only", ("1006",)),
                        ("six_districts", DISTRICTS)):
        train_x = np.concatenate([district_arrays[code][0][:TRAIN_DAYS] for code in codes])
        train_y = np.concatenate([district_arrays[code][1][:TRAIN_DAYS] for code in codes])
        x_cal, y_cal = bk_x[TRAIN_DAYS:TRAIN_DAYS + CAL_DAYS], bk_y[TRAIN_DAYS:TRAIN_DAYS + CAL_DAYS]
        x_test, y_test = bk_x[TRAIN_DAYS + CAL_DAYS:], bk_y[TRAIN_DAYS + CAL_DAYS:]
        (train_norm, cal_norm, test_norm) = normalize(train_x, x_cal, x_test)
        result, expected, observed = fit(train_norm, train_y, cal_norm, y_cal,
                                         test_norm, y_test, rr, cc)
        result["training_district_codes"] = list(codes)
        result["training_positive_cell_days"] = int(train_y.sum())
        results[name] = result
        predictions[name] = expected
        print(name, result["test_cell_rank"], result["test_daily"]["average_precision"],
              flush=True)
    historical = bk_y[:TRAIN_DAYS + CAL_DAYS, 0, rr, cc].sum(axis=0)
    results["historical_baseline"] = rank_metrics(historical, observed)
    rows = [{"cell_id": f"bk:{int(rr[i]):02d}:{int(cc[i]):02d}",
             "historical_positive_days_jan_sep": int(historical[i]),
             "bk_only_cnn_expected_days_oct_dec": round(float(predictions["bang_kapi_only"][i]), 5),
             "six_district_cnn_expected_days_oct_dec": round(float(predictions["six_districts"][i]), 5),
             "observed_days_oct_dec": int(observed[i]),
             "historical_rank_percentile": round(float(percentiles(historical)[i]), 2),
             "bk_only_cnn_rank_percentile": round(float(percentiles(predictions["bang_kapi_only"])[i]), 2),
             "six_district_cnn_rank_percentile": round(float(percentiles(predictions["six_districts"])[i]), 2),
             "observed_rank_percentile": round(float(percentiles(observed)[i]), 2)}
            for i in range(len(rr))]
    write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
    write_comparison_map(rows)
    metrics = {"purpose": "isolated effect of adding five other districts to 2025 terrain-and-rain CNN training",
               "target": "100 m Traffy direct flood-report footprint touches a 250 m cell on a day",
               "test_area": "Bang Kapi", "test_dates": ["2025-10-01", "2025-12-31"],
               "train_dates": ["2025-01-01", "2025-07-31"],
               "selection_and_calibration_dates": ["2025-08-01", "2025-09-30"],
               "input_features": ["elevation_mean_m", "elevation_vs_3cell_mean_m", "terrain_slope",
                                  "district_coverage_fraction", "nearest_pump_m", "pumps_within_1km",
                                  "nearest_gate_m", "prior_report_footprint_days", "month_sin", "month_cos",
                                  *RAIN_FEATURES, "district_mask"],
               "terrain_source": "cached BMA LiDAR at 512 pixels per district, averaged onto 250 m grid",
               "rain_source": "district Open-Meteo hourly historical precipitation",
               "districts": audits, "models": results,
               "limitation": "No road/building/amenity or all-complaint activity channels; exploratory reuse of October-December 2025 test"}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: result["test_cell_rank"] for name, result in results.items()
                      if name != "historical_baseline"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
