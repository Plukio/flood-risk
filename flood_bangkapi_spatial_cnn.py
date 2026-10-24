"""Small spatial CNN for Bang Kapi report-footprint cell-day labels.

Uses the same 250 m grid, open-data features, connected labels and held-out
October-December 2025 period as flood_bangkapi_proxy_probability.py. The CNN
sees each cell and its neighbours through two 3x3 convolutions (a 5x5-cell
receptive field). It predicts the report-footprint proxy, not observed water.

Run: .venv/bin/python flood_bangkapi_spatial_cnn.py
Requires PyTorch in the project venv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from flood_bangkapi_connected_footprints import build as connected_labels
from flood_bangkapi_proxy_probability import (
    CAL_END, SCENARIO_DAY, SCENARIOS, TRAIN_END, apply_calibration,
    calibrate_intercept, evaluation, write_probability_map,
)
from flood_bangkapi_surface_model import (
    OUTPUT, activity_by_cell, cells, rainfall_by_date, rank_metrics,
    scenario_rain, vector, write_csv,
)


SEED = 7
CHANNELS = 23
POSITIVE_WEIGHT = 30.0
MAX_EPOCHS = 40
PATIENCE = 8
HISTORY_FLOOD = Path(__file__).resolve().parent / "data" / "flood_bangkapi_surface" / "traffy_flood_2022_2024_minimal.csv"
HISTORY_ACTIVITY = HISTORY_FLOOD.parent / "traffy_all_activity_2022_2024_cell_month.csv"


class SpatialCNN(nn.Module):
    def __init__(self, in_channels: int = CHANNELS + 1) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 1, 1),
        )
        self.local = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.context(x) + self.local(x)


def grid_data(sites: list[dict], labels: dict, weather: dict, activity: dict):
    days = sorted(weather)
    addresses = [tuple(map(int, site["cell_id"].split(":")[1:])) for site in sites]
    rr = np.asarray([r for r, _ in addresses])
    cc = np.asarray([c for _, c in addresses])
    height, width = int(rr.max()) + 1, int(cc.max()) + 1
    valid = np.zeros((height, width), dtype=np.float32)
    valid[rr, cc] = 1
    raw = np.zeros((len(days), CHANNELS + 1, height, width), dtype=np.float32)
    raw[:, CHANNELS] = valid
    targets = np.zeros((len(days), 1, height, width), dtype=np.float32)
    prior = Counter()
    for t, day in enumerate(days):
        if t and day.year != days[t - 1].year:
            prior.clear()
        month = day.strftime("%Y-%m")
        for i, site in enumerate(sites):
            cid = site["cell_id"]
            activity_count = sum(count for earlier, count in activity.get(cid, {}).items()
                                 if earlier.startswith(str(day.year)) and earlier < month
                                 and earlier <= "2025-09")
            raw[t, :CHANNELS, rr[i], cc[i]] = vector(
                site, day, weather[day], prior[cid], activity_count, False)
            targets[t, 0, rr[i], cc[i]] = int(day in labels.get(cid, set()))
        if day <= CAL_END:
            for site in sites:
                cid = site["cell_id"]
                prior[cid] += int(day in labels.get(cid, set()))
    return days, raw, targets, valid, rr, cc, prior


def normalize(raw: np.ndarray, valid: np.ndarray, train_count: int):
    result = raw.copy()
    stats = []
    mask = valid.astype(bool)
    for channel in range(CHANNELS):
        values = raw[:train_count, channel][:, mask].ravel()
        values = values[np.isfinite(values)]
        median = float(np.median(values))
        filled_train = np.where(np.isfinite(values), values, median)
        mean = float(filled_train.mean())
        std = max(float(filled_train.std()), 1e-4)
        result[:, channel] = np.clip(
            (np.where(np.isfinite(raw[:, channel]), raw[:, channel], median) - mean) / std,
            -5, 5) * valid
        stats.append((median, mean, std))
    return result, stats


def normalize_scenario(raw: np.ndarray, valid: np.ndarray, stats: list[tuple]):
    result = raw.copy()
    for channel, (median, mean, std) in enumerate(stats):
        result[:, channel] = np.clip(
            (np.where(np.isfinite(raw[:, channel]), raw[:, channel], median) - mean) / std,
            -5, 5) * valid
    return result


@torch.no_grad()
def logits_for(model: SpatialCNN, data: np.ndarray) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(data), 16):
        chunks.append(model(torch.from_numpy(data[start:start + 16])).numpy())
    return np.concatenate(chunks)


def percentiles(values: np.ndarray) -> np.ndarray:
    ranks = 100 * (rankdata(values, method="average") - 1) / (len(values) - 1)
    return np.where(values > 0, ranks, 0)


def weighted_bce(logits: np.ndarray, targets: np.ndarray) -> float:
    """The training objective averaged over district cells, at fixed weights."""
    return float(np.mean(np.where(targets > 0,
                                  POSITIVE_WEIGHT * np.logaddexp(0, -logits),
                                  np.logaddexp(0, logits))))


def write_training_curve(history: list[dict], selected_epoch: int, prefix: str,
                         train_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, (loss_ax, ap_ax) = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
    loss_ax.plot(epochs, [row["train_weighted_bce"] for row in history],
                 marker="o", ms=3, label=f"Train: {train_label}", color="#16789b")
    loss_ax.plot(epochs, [row["calibration_weighted_bce"] for row in history],
                 marker="o", ms=3, label="Validation: Aug–Sep", color="#d56f2a")
    loss_ax.set_ylabel("Positive-weighted BCE per valid cell-day")
    loss_ax.legend(frameon=False)
    loss_ax.grid(alpha=0.2)
    ap_ax.plot(epochs, [row["calibration_average_precision"] for row in history],
               marker="o", ms=3, color="#624c9d")
    ap_ax.set_ylabel("Validation AP")
    ap_ax.set_xlabel("Epoch")
    ap_ax.grid(alpha=0.2)
    for ax in (loss_ax, ap_ax):
        ax.axvline(selected_epoch, color="#38454d", ls="--", lw=1.2)
    loss_ax.set_title(f"Bang Kapi 2D CNN training · epoch {selected_epoch} selected by validation AP")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{prefix}_loss_curve.png", dpi=180)
    fig.savefig(OUTPUT / f"{prefix}_loss_curve.svg")
    plt.close(fig)


def write_comparison_map(rows: list[dict], prefix: str) -> None:
    geojson = json.loads((OUTPUT / "surface_cells_250m.geojson").read_text())
    by_cell = {row["cell_id"]: row for row in rows}
    for feature in geojson["features"]:
        feature["properties"].update(by_cell[feature["properties"]["cell_id"]])
    (OUTPUT / f"{prefix}_comparison.geojson").write_text(json.dumps(
        geojson, ensure_ascii=False))
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bang Kapi CNN comparison</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>body{margin:0;font:14px system-ui,sans-serif;color:#20313b}header{background:#12344b;color:white;padding:12px 16px;display:flex;gap:15px;align-items:center;flex-wrap:wrap}header strong{font-size:17px}header small{max-width:670px}select{font:inherit;padding:5px;border:0;border-radius:3px}#map{height:calc(100vh - 84px);min-height:480px}.legend{background:white;padding:10px;box-shadow:0 1px 8px #5555;border-radius:5px;line-height:1.45}.bar{height:12px;background:linear-gradient(90deg,#fff1bd,#e3944a,#9b2429);margin:7px 0}</style></head><body><header><strong>Bang Kapi · spatial CNN comparison</strong><label>Show <select id="view"><option value="historical_rank_percentile">Historical Jan–Sep</option><option value="tree_rank_percentile">Cell-feature tree</option><option value="cnn_rank_percentile">2D CNN</option><option value="observed_rank_percentile">Observed Oct–Dec</option></select></label><small>Rank among 527 cells for report-footprint-positive days. Models use actual held-out weather; observed outcomes are shown only for retrospective evaluation.</small></header><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const grid=__GRID__;const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);let active='historical_rank_percentile';
function color(p){const t=Math.max(0,Math.min(1,Number(p)/100));return `hsl(${48-46*t},${85-20*t}%,${93-56*t}%)`}
function popup(p){return `<b>${p.cell_id}</b><br>Historical Jan–Sep positive days: <b>${p.historical_positive_days_jan_sep}</b><br>Cell-feature tree expected Oct–Dec days: <b>${p.tree_expected_days_oct_dec}</b><br>2D CNN expected Oct–Dec days: <b>${p.cnn_expected_days_oct_dec}</b><br>Observed Oct–Dec positive days: <b>${p.observed_days_oct_dec}</b><br>Current rank: <b>${Number(p[active]).toFixed(1)}</b> percentile<br><small>Report-footprint proxy, not verified surface flooding.</small>`}
const cells=L.geoJSON(grid,{style:f=>({color:'#5a5050',weight:.25,fillColor:color(f.properties[active]),fillOpacity:.83}),onEachFeature:(f,l)=>l.bindPopup(popup(f.properties))}).addTo(map);map.fitBounds(cells.getBounds());
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Cell rank percentile</b><div class="bar"></div><small>0 → 100; darker means higher rank.<br>Zero-day cells are rank 0. Ties share a rank.</small>';return d};legend.addTo(map);
document.getElementById('view').addEventListener('change',e=>{active=e.target.value;cells.setStyle(f=>({fillColor:color(f.properties[active])}));cells.eachLayer(l=>l.setPopupContent(popup(l.feature.properties)));});
</script></body></html>"""
    (OUTPUT / f"{prefix}_comparison_map.html").write_text(
        html.replace("__GRID__", json.dumps(geojson, ensure_ascii=False).replace("</", "<\\/")))


def main(use_history: bool = False) -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    prefix = "surface_connected_cnn_multiyear" if use_history else "surface_connected_cnn"
    sites = cells()
    labels, current_audit = connected_labels()
    if use_history:
        if not HISTORY_FLOOD.exists() or not HISTORY_ACTIVITY.exists():
            raise FileNotFoundError("Run flood_bangkapi_traffy_history.py first")
        older_labels, older_audit = connected_labels(HISTORY_FLOOD, "2022_2024")
        for cid, dates in older_labels.items():
            labels.setdefault(cid, set()).update(dates)
        label_audit = {"2022_2024": older_audit, "2025": current_audit}
        weather = rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31))
    else:
        label_audit = current_audit
        weather = rainfall_by_date()
    activity = activity_by_cell()
    if use_history:
        with HISTORY_ACTIVITY.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                cid = f"bk:{int(row['grid_row']):02d}:{int(row['grid_col']):02d}"
                activity[cid][row["month"]] += int(row["all_complaints"])
    days, raw, target, valid, rr, cc, prior = grid_data(sites, labels, weather, activity)
    train_count = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    normalized, stats = normalize(raw, valid, train_count)
    x_train = normalized[:train_count]
    y_train = target[:train_count]
    x_cal, y_cal = normalized[train_count:cal_end], target[train_count:cal_end]
    x_test, y_test = normalized[cal_end:], target[cal_end:]
    model = SpatialCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    positive_weight = torch.tensor(POSITIVE_WEIGHT)
    mask = torch.from_numpy(valid[None, None])
    rng = np.random.default_rng(SEED)
    best_ap, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(train_count)
        for start in range(0, train_count, 12):
            indices = order[start:start + 12]
            x = torch.from_numpy(x_train[indices])
            y = torch.from_numpy(y_train[indices])
            optimizer.zero_grad()
            weighted = F.binary_cross_entropy_with_logits(
                model(x), y, pos_weight=positive_weight, reduction="none")
            loss = (weighted * mask).sum() / (len(indices) * valid.sum())
            loss.backward()
            optimizer.step()
        train_logits = logits_for(model, x_train)[:, 0, rr, cc]
        cal_logits = logits_for(model, x_cal)[:, 0, rr, cc]
        train_loss = weighted_bce(train_logits, y_train[:, 0, rr, cc])
        cal_loss = weighted_bce(cal_logits, y_cal[:, 0, rr, cc])
        cal_prob = expit(cal_logits).ravel()
        cal_truth = y_cal[:, 0, rr, cc].ravel()
        ap = float(average_precision_score(cal_truth, cal_prob))
        history.append({"epoch": epoch, "train_weighted_bce": round(train_loss, 7),
                        "calibration_weighted_bce": round(cal_loss, 7),
                        "calibration_average_precision": round(ap, 7)})
        if ap > best_ap + 1e-6:
            best_ap, best_epoch = ap, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch={epoch} train_BCE={train_loss:.5f} cal_BCE={cal_loss:.5f} "
                  f"cal_AP={ap:.5f} best_AP={best_ap:.5f}", flush=True)
        if stale >= PATIENCE and epoch >= 10:
            break
    assert best_state is not None
    for row in history:
        row["selected_checkpoint"] = int(row["epoch"] == best_epoch)
    write_csv(OUTPUT / f"{prefix}_training_history.csv", history)
    write_training_curve(history, best_epoch, prefix,
                         "2022–Jul 2025" if use_history else "Jan–Jul 2025")
    model.load_state_dict(best_state)
    raw_cal = expit(logits_for(model, x_cal)[:, 0, rr, cc]).ravel()
    cal_truth = y_cal[:, 0, rr, cc].ravel().astype(np.uint8)
    bias = calibrate_intercept(raw_cal, cal_truth)
    raw_test = expit(logits_for(model, x_test)[:, 0, rr, cc])
    test_truth = y_test[:, 0, rr, cc].astype(np.uint8)
    p_test = apply_calibration(raw_test, bias)
    expected = p_test.sum(axis=0)
    observed = test_truth.sum(axis=0)
    test_months = {}
    test_days = days[cal_end:]
    for month in sorted({day.strftime("%Y-%m") for day in test_days}):
        selected = np.asarray([day.strftime("%Y-%m") == month for day in test_days])
        test_months[month] = {"observed_positive_cell_days": int(test_truth[selected].sum()),
                              "predicted_positive_cell_days": round(float(p_test[selected].sum()), 1)}
    historical = np.asarray([prior[site["cell_id"]] for site in sites])
    reference = json.loads((OUTPUT / "surface_connected_probability_metrics.json").read_text())
    with (OUTPUT / "surface_connected_historical_comparison.csv").open(
            encoding="utf-8-sig", newline="") as file:
        old_rows = {row["cell_id"]: row for row in csv.DictReader(file)}
    if any(int(old_rows[site["cell_id"]]["observed_positive_days_oct_dec"]) != observed[i]
           for i, site in enumerate(sites)):
        raise ValueError("Tree comparison uses different held-out labels")
    tree = np.asarray([float(old_rows[site["cell_id"]][
        "model_expected_positive_days_oct_dec"]) for site in sites])
    historical_rank = percentiles(historical)
    cnn_rank = percentiles(expected)
    observed_rank = percentiles(observed)
    rows = [{"cell_id": site["cell_id"],
             "historical_positive_days_jan_sep": int(historical[i]),
             "tree_expected_days_oct_dec": round(float(tree[i]), 5),
             "cnn_expected_days_oct_dec": round(float(expected[i]), 5),
             "observed_days_oct_dec": int(observed[i]),
             "historical_rank_percentile": round(float(historical_rank[i]), 2),
             "tree_rank_percentile": float(old_rows[site["cell_id"]]["model_rank_percentile"]),
             "cnn_rank_percentile": round(float(cnn_rank[i]), 2),
             "observed_rank_percentile": round(float(observed_rank[i]), 2)}
            for i, site in enumerate(sites)]
    write_csv(OUTPUT / f"{prefix}_comparison.csv", rows)
    write_comparison_map(rows, prefix)
    metrics = {"target": "connected report-footprint touches a 250 m cell on a day",
               "not_target": "independently observed surface flooding",
               "cells": len(sites), "grid_shape": list(valid.shape),
               "train_dates": [str(days[0]), str(TRAIN_END)],
               "selection_and_calibration_dates": ["2025-08-01", str(CAL_END)],
               "test_dates": [str(days[cal_end]), str(days[-1])],
               "architecture": "2 spatial 3x3 convolutions plus 1x1 local skip; 5x5-cell receptive field",
               "input_channels": CHANNELS + 1,
               "positive_loss_weight": POSITIVE_WEIGHT,
               "selected_epoch": best_epoch,
               "epochs_run": len(history),
               "loss_curve": f"{prefix}_loss_curve.png",
               "training_history": f"{prefix}_training_history.csv",
               "older_years_added": use_history,
               "prior_report_count_resets_each_january": True,
               "selection_calibration_average_precision": round(best_ap, 5),
               "calibration_method": "intercept-only correction on Aug-Sep after epoch selection on Aug-Sep AP",
               "calibration_log_odds_shift": round(bias, 4),
               "train_positive_cell_days": int(y_train[:, 0, rr, cc].sum()),
               "calibration_positive_cell_days": int(cal_truth.sum()),
               "test_positive_cell_days": int(test_truth.sum()),
               "test_cnn": evaluation(p_test.ravel(), test_truth.ravel()),
               "test_cnn_cell_rank": rank_metrics(expected, observed),
               "test_months": test_months,
               "test_historical_cell_rank": rank_metrics(historical, observed),
               "test_tree_cell_rank_reference": reference["test_cell_rank"],
               "labels": label_audit}
    (OUTPUT / f"{prefix}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: metrics[key] for key in (
        "selected_epoch", "selection_calibration_average_precision",
        "test_cnn_cell_rank", "test_historical_cell_rank",
        "test_tree_cell_rank_reference")}, indent=2))

    # Scenario map uses only history available through 30 September.
    scenario_rows = [{"cell_id": site["cell_id"], "centre_lon": site["centre_lon"],
                      "centre_lat": site["centre_lat"], "flood_probability": None,
                      "proxy_definition": "terrain-connected report area or 100 m fallback; spatial CNN",
                      "proxy_positive_days_2025": sum(day.year == 2025 for day in labels.get(
                          site["cell_id"], set()))}
                     for site in sites]
    for name, (rate, duration) in SCENARIOS.items():
        scene = np.zeros((1, CHANNELS + 1, *valid.shape), dtype=np.float32)
        scene[0, CHANNELS] = valid
        rain = scenario_rain(rate, duration)
        for i, site in enumerate(sites):
            cid = site["cell_id"]
            count = sum(value for month, value in activity.get(cid, {}).items()
                        if month.startswith("2025-") and month <= "2025-09")
            scene[0, :CHANNELS, rr[i], cc[i]] = vector(
                site, SCENARIO_DAY, rain, prior[cid], count, False)
        scene = normalize_scenario(scene, valid, stats)
        probabilities = apply_calibration(expit(logits_for(model, scene)[0, 0, rr, cc]), bias)
        for row, probability in zip(scenario_rows, probabilities):
            row[f"proxy_flood_probability_{name}"] = round(float(probability), 6)
    write_csv(OUTPUT / f"{prefix}_probability_scenarios.csv", scenario_rows)
    scale_max = max(0.01, math.ceil(max(
        row[f"proxy_flood_probability_{name}"] for row in scenario_rows for name in SCENARIOS) * 100) / 100)
    write_probability_map(scenario_rows, scale_max, prefix,
                          "a connected report area or 100 m fallback (experimental spatial CNN)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-history", action="store_true",
                        help="Add BMA Traffy 2022–2024 Bang Kapi reports to training")
    main(parser.parse_args().use_history)
