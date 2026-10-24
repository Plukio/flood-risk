"""Pre-October Bang Kapi CNN queue versus BMA recorded road-flood depths.

Run: .venv/bin/python flood_bangkok_cnn_depth_capture.py
The six site cells are approximate 2026 inventory anchors for 2025 reports.
"""

from __future__ import annotations

import csv
import json
from datetime import date

import numpy as np
import torch
from scipy.special import expit

from flood_bangkok_inspection_budget_backtest import tie_expected
from flood_bangkok_spatial_cnn_crossdistrict import CHANNELS, fit, normalize
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END, district_data, report_rows
from flood_bangkok_transfer import CACHE, load_districts, write_csv
from flood_bangkapi_proxy_probability import apply_calibration, calibrate_intercept
from flood_bangkapi_spatial_cnn import SpatialCNN, logits_for
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, rainfall_by_date


ARTIFACT = "surface_cnn_depth_capture_bang_kapi_2025"
BUDGETS = (10, 25, 50, 100)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def pretrained_queue() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    torch.set_num_threads(4)
    district = next(d for d in load_districts() if d["code"] == "1006")
    spec = grid_spec(district["polygon"])
    raw, target, (rr, cc), _ = district_data(
        district, report_rows(expanded=True), pump_points(), spec["rows"], spec["cols"])
    rain = rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                            CACHE / "rain_1006_2021_2025.json")
    days = sorted(rain)
    train_end = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    scenario_indices = np.asarray([
        i for i, day in enumerate(days)
        if day.year < 2025 and day.month >= 10 and rain[day]["rain_today_mm"] >= 10],
        dtype=np.int32)
    if len(scenario_indices) < 5:
        raise ValueError("Not enough prior Q4 rainy-day scenarios")
    scenarios = raw[scenario_indices].copy()
    # Channel 7 is prior annual report-positive days. Freeze it to the 2025
    # Oct 1 snapshot rather than replaying the older scenario year's history.
    scenarios[:, 7] = raw[cal_end, 7]
    train_x, cal_x, scenario_x = normalize(raw[:train_end],
                                           raw[train_end:cal_end], scenarios)
    checkpoint = OUTPUT / f"{ARTIFACT}_checkpoint.pt"
    # The fit helper requires a test array; reuse August–September calibration
    # rows only. No October–December 2025 weather or outcome is supplied.
    selection, _, _ = fit(train_x, target[:train_end], cal_x,
                          target[train_end:cal_end], cal_x,
                          target[train_end:cal_end], rr, cc,
                          checkpoint_output_path=checkpoint)
    network = SpatialCNN(CHANNELS + 1)
    network.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                       weights_only=True)["state_dict"])
    raw_cal = expit(logits_for(network, cal_x)[:, 0, rr, cc]).ravel()
    cal_truth = target[train_end:cal_end, 0, rr, cc].ravel()
    bias = calibrate_intercept(raw_cal, cal_truth)
    raw_scenario = expit(logits_for(network, scenario_x)[:, 0, rr, cc])
    cnn_score = apply_calibration(raw_scenario, bias).mean(axis=0)
    return cnn_score, rr, cc, len(scenario_indices), {
        "selected_epoch": selection["selected_epoch"],
        "calibration_AP_for_selection": selection["calibration_AP_for_selection"],
        "calibration_log_odds_shift": round(float(bias), 4)}


def main() -> None:
    cnn_score, rr, cc, scenario_days, selection = pretrained_queue()
    ranks = csv_rows(OUTPUT / "surface_inspection_budget_expanded_cell_ranks.csv")
    target_ranks = {(int(row["cell_id"].split(":")[1]),
                     int(row["cell_id"].split(":")[2])): row
                    for row in ranks if row["district_code"] == "1006"
                    and row["year"] == "2025"}
    if set(target_ranks) != set(zip(rr, cc)):
        raise ValueError("CNN and historical queue cell grids differ")
    historical = np.asarray([float(target_ranks[(int(r), int(c))][
        "recent_report_days_score"]) for r, c in zip(rr, cc)])
    tree = np.asarray([float(target_ranks[(int(r), int(c))][
        "local_baseline_score"]) for r, c in zip(rr, cc)])
    sites = csv_rows(OUTPUT / "surface_bma_depth_bang_kapi_2025_sites.csv")
    positions = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rr, cc))}
    site_count = np.zeros(len(rr), dtype=np.float32)
    depth_sum = np.zeros(len(rr), dtype=np.float32)
    max_depth = np.zeros(len(rr), dtype=np.float32)
    site_rows = []
    for site in sites:
        _, r, c = site["approximate_cell_id"].split(":")
        i = positions[(int(r), int(c))]
        days = int(site["bma_q4_site_days"])
        depth = float(site["mean_depth_cm"]) * days
        site_count[i] += 1
        depth_sum[i] += depth
        max_depth[i] = max(max_depth[i], float(site["max_depth_cm"]))
        site_rows.append({"site": site["site"],
                          "site_label": site["site_label"],
                          "approximate_cell_id": site["approximate_cell_id"],
                          "bma_q4_site_days": days,
                          "sum_recorded_depth_cm_across_site_days": depth,
                          "max_recorded_depth_cm": site["max_depth_cm"]})
    methods = {"preseason_local_cnn": cnn_score,
               "recent_report_history": historical,
               "preseason_local_tree": tree}
    result = {}
    for budget in BUDGETS:
        result[str(budget)] = {}
        for name, scores in methods.items():
            order = np.lexsort((cc, rr, -scores))
            ordinal = np.empty(len(rr), dtype=np.int32)
            ordinal[order] = np.arange(1, len(rr) + 1)
            for site in site_rows:
                _, r, c = site["approximate_cell_id"].split(":")
                site[f"{name}_rank"] = int(ordinal[positions[(int(r), int(c))]])
            result[str(budget)][name] = {
                "captured_named_sites_expected_at_ties": round(
                    tie_expected(scores, site_count, budget), 3),
                "captured_depth_cm_across_site_days_expected_at_ties": round(
                    tie_expected(scores, depth_sum, budget), 3),
                "captured_max_site_depth_cm_expected_at_ties": round(
                    tie_expected(scores, max_depth, budget), 3),
                "captured_site_days_expected_at_ties": round(
                    tie_expected(scores, np.asarray([
                        sum(int(s["bma_q4_site_days"]) for s in site_rows
                            if s["approximate_cell_id"] == f"1006:{int(r):02d}:{int(c):02d}")
                        for r, c in zip(rr, cc)], dtype=np.float32), budget), 3),
            }
    output = {"area": "Bang Kapi 2025 Q4",
              "decision_date": "2025-09-30",
              "cnn_fit_dates": ["2022-01-01", str(TRAIN_END)],
              "cnn_selection_dates": ["2025-08-01", str(CAL_END)],
              "historical_q4_rain_scenario_days": scenario_days,
              "cnn_selection": selection,
              "bma_named_sites": int(site_count.sum()),
              "bma_site_days": sum(int(s["bma_q4_site_days"]) for s in site_rows),
              "total_recorded_depth_cm_across_site_days": float(depth_sum.sum()),
              "total_max_site_depth_cm": float(max_depth.sum()),
              "depth_sum_meaning": "Sum of recorded centimetres across separate BMA site-days; a severity-weighted count, not water volume or average depth",
              "location_limit": "BMA 2026 inventory anchors for six named 2025 sites are approximate cells",
              "target_limit": "CNN trained on Traffy report buffers, not recorded BMA depth",
              "budgets": result}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        output, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / f"{ARTIFACT}_sites.csv", site_rows)
    print(json.dumps({"selection": selection, "K25": result["25"]}, indent=2))


if __name__ == "__main__":
    main()
