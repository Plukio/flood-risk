"""Rescore Nong Chok's saved canal CNN with each 2026 GISTDA day's rain.

Weather and month are date matched. Static asset maps are unchanged because
no time-matched pump/gate operation or drainage-state input exists for this
trained CNN. The 2026 prior-complaint channel is unavailable, so evaluate
both a frozen 2025 spatial surrogate and zero-prior sensitivity.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from matplotlib.patches import Patch, Rectangle
from scipy.special import expit
from scipy.stats import spearmanr

from flood_bangkok_gistda_grid_backtest import OUTPUT, selection_weights
from flood_bangkok_spatial_cnn_citywide import CAL_N, CHANNELS, FEATURE_CACHE, transform
from flood_bangkok_spatial_cnn_infrastructure import augmented_arrays
from flood_bangkok_transfer import CACHE, RAIN_URL, load_districts
from flood_bangkapi_spatial_cnn import SpatialCNN
from flood_bangkapi_surface_grid import GRID_M, grid_spec
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date, write_csv


CODE = "1003"
RAIN_PATH = CACHE / "rain_1003_2026_gistda_events.json"
CHECKPOINT = OUTPUT / "surface_crossdistrict_infrastructure_ablation_citywide_v1_canal_only.pt"
SCENARIOS = ("2025_prior_surrogate", "zero_prior_sensitivity")


def event_rain(district: dict, first: date, last: date) -> dict:
    if not RAIN_PATH.exists():
        params = {"latitude": district["lat"], "longitude": district["lon"],
                  "start_date": str(date.fromordinal(first.toordinal() - 3)),
                  "end_date": str(last), "hourly": "precipitation",
                  "timezone": "Asia/Bangkok"}
        response = requests.get(RAIN_URL, params=params, timeout=50)
        response.raise_for_status()
        RAIN_PATH.write_text(response.text)
    return rainfall_by_date(first, last, RAIN_PATH)


def event_input(base: np.ndarray, day: date, rain: dict, scenario: str) -> np.ndarray:
    raw = np.asarray(base, dtype=np.float32).copy()
    valid = raw[0, -1] > 0
    if scenario == "zero_prior_sensitivity":
        raw[0, 7, valid] = 0
    elif scenario != "2025_prior_surrogate":
        raise ValueError(scenario)
    raw[0, 8, valid] = math.sin(2 * math.pi * day.month / 12)
    raw[0, 9, valid] = math.cos(2 * math.pi * day.month / 12)
    for i, feature in enumerate(RAIN_FEATURES):
        raw[0, 10 + i, valid] = math.log1p(max(0, rain[feature]))
    return raw


def overlay(rows: list[dict], score: np.ndarray, district: dict) -> None:
    spec = grid_spec(district["polygon"])
    selected = selection_weights(score, 0.10)
    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=150)
    for row in rows:
        area = float(row["mapped_wet_area_m2"])
        if area <= 0:
            continue
        x = spec["west"] + int(row["col"]) * GRID_M
        y = spec["north"] - (int(row["row"]) + 1) * GRID_M
        ax.add_patch(Rectangle((x, y), GRID_M, GRID_M,
                               facecolor="#2477c4", edgecolor="none",
                               alpha=min(0.95, 0.20 + 0.75 * area / 62500)))
    for row, weight in zip(rows, selected):
        if weight <= 0:
            continue
        x = spec["west"] + int(row["col"]) * GRID_M
        y = spec["north"] - (int(row["row"]) + 1) * GRID_M
        ax.add_patch(Rectangle((x, y), GRID_M, GRID_M, fill=False,
                               edgecolor="#e75b22", linewidth=0.5,
                               alpha=float(weight)))
    boundary = district["polygon"]
    for polygon in ([boundary] if boundary.geom_type == "Polygon" else boundary.geoms):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color="#223b54", linewidth=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(spec["west"] - 500, spec["east"] + 500)
    ax.set_ylim(spec["south"] - 500, spec["north"] + 500)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Nong Chok, 4 Oct 2026: event-day rain vs. GISTDA mapped water",
                 fontsize=11)
    ax.legend(handles=[Patch(facecolor="#2477c4", label="GISTDA mapped-water share of 250 m cell"),
                       Patch(facecolor="none", edgecolor="#e75b22",
                             label="Top 10% canal-CNN rank, date-matched rain")],
              loc="lower left", fontsize=8, framealpha=0.92)
    fig.text(0.5, 0.02, "2026 hourly gridded rain; 2025 prior-complaint surrogate; static mapped assets; no pump/gate state.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "gistda_20261004_nong_chok_eventday_rain_overlay.png")
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    district = next(d for d in load_districts() if d["code"] == CODE)
    scan = json.loads((OUTPUT / "gistda_2026_bma_daily_scan.json").read_text())
    days = [date.fromisoformat(row["day"]) for row in scan["rows"]
            if row["polygons_intersecting_bma"] > 0]
    rain = event_rain(district, min(days), max(days))
    meta = json.loads((FEATURE_CACHE / f"{CODE}_meta.json").read_text())
    x, _ = augmented_arrays({CODE: meta})[CODE]
    base = np.asarray(x[CAL_N:CAL_N + 1], dtype=np.float32)
    rr = np.asarray(meta["valid_rows"])
    cc = np.asarray(meta["valid_cols"])
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SpatialCNN(CHANNELS + 8)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if tuple(checkpoint["masked_channels"]) != tuple(range(CHANNELS + 2, CHANNELS + 7)):
        raise ValueError("Expected saved canal-only ablation checkpoint")
    bias_rows = json.loads((OUTPUT / "surface_crossdistrict_infrastructure_ablation_citywide_v1_metrics.json").read_text())[
        "models"]["canal_only"]["evaluated_districts"]
    shift = next(row["log_odds_shift"] for row in bias_rows if row["code"] == CODE)
    score_rows = []
    daily = []
    for day in days:
        grid_path = OUTPUT / f"gistda_{day:%Y%m%d}_grid_250m.csv"
        with grid_path.open(encoding="utf-8", newline="") as file:
            grid = [row for row in csv.DictReader(file) if row["district_code"] == CODE]
        if [(int(row["row"]), int(row["col"])) for row in grid] != list(zip(rr, cc)):
            raise ValueError(f"Cell alignment changed on {day}")
        wet = np.asarray([float(row["mapped_wet_area_m2"]) for row in grid])
        reference = np.asarray([float(row["canal_cnn_20mm_2h_reference_score"])
                                for row in grid])
        for scenario in SCENARIOS:
            raw = event_input(base, day, rain[day], scenario)
            normalized = torch.from_numpy(transform(raw, checkpoint["stats"],
                                                    tuple(checkpoint["masked_channels"])))
            with torch.no_grad():
                logits = model(normalized).numpy()[0, 0, rr, cc]
            scores = expit(logits + shift)
            result = {"day": str(day), "prior_assumption": scenario,
                      "rain_today_mm": round(rain[day]["rain_today_mm"], 3),
                      "rain_peak_1h_mm": round(rain[day]["rain_peak_1h_mm"], 3),
                      "rain_peak_3h_mm": round(rain[day]["rain_peak_3h_mm"], 3),
                      "rain_prev24h_mm": round(rain[day]["rain_prev24h_mm"], 3),
                      "rain_prev72h_mm": round(rain[day]["rain_prev72h_mm"], 3),
                      "mapped_wet_area_km2": round(float(wet.sum()) / 1e6, 5),
                      "rank_spearman_vs_20mm_reference": round(float(
                          spearmanr(scores, reference).statistic), 4)}
            for fraction in (0.05, 0.10, 0.20):
                selected = selection_weights(scores, fraction)
                result[f"top_{int(fraction*100)}pct_area_capture"] = round(
                    float(np.dot(selected, wet) / wet.sum()), 4)
            daily.append(result)
            for row, score in zip(grid, scores):
                score_rows.append({"day": str(day), "prior_assumption": scenario,
                                   "cell_id": row["cell_id"], "row": row["row"],
                                   "col": row["col"],
                                   "mapped_wet_area_m2": row["mapped_wet_area_m2"],
                                   "complaint_proxy_score": round(float(score), 8)})
            if day == date(2026, 10, 4) and scenario == "2025_prior_surrogate":
                overlay(grid, scores, district)
        print(day, [(x["prior_assumption"], x["top_10pct_area_capture"])
                    for x in daily[-2:]], flush=True)
    write_csv(OUTPUT / "gistda_2026_nong_chok_eventday_scores.csv", score_rows)
    rain_response = json.loads(RAIN_PATH.read_text())
    summary = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "district": "Nong Chok", "source_rain": RAIN_URL,
               "rain_location": {"latitude": district["lat"], "longitude": district["lon"]},
               "returned_rain_grid_location": {"latitude": rain_response["latitude"],
                                               "longitude": rain_response["longitude"]},
               "rain_input": "Observed full-day Open-Meteo gridded hourly precipitation, Bangkok local time; explanatory hindcast, not advance forecast",
               "source_rain_cache": str(RAIN_PATH),
               "model": "saved 2025 canal CNN trained on Traffy complaint proxies",
               "static_inputs": "Nong Chok BMA terrain, 2025 OSM canals and pre-2026 BMA pump/gate location records",
               "missing_dynamic_inputs": ["2026 per-cell prior Traffy complaints",
                                          "time-matched pump and gate operating state",
                                          "time-matched local canal and outfall levels"],
               "prior_assumptions": {"2025_prior_surrogate": "frozen 1 Oct 2025 prior-complaint map",
                                     "zero_prior_sensitivity": "all prior-complaint counts set to zero"},
               "daily": daily}
    path = OUTPUT / "gistda_2026_nong_chok_eventday_metrics.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
