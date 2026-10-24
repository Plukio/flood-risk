"""Test whether local sampling corrects naive multi-district pooling.

Eleven 2022–2025 districts use the same features and labels as the earlier
experiment. Each training epoch samples Bang Kapi for half of its district-day
images, and the other ten districts for the other half. Bang Kapi Aug–Sep 2025
selects and calibrates; Oct–Dec is the reused exploratory test.

Run: .venv/bin/python flood_bangkok_spatial_cnn_domain_balance.py
"""

from __future__ import annotations

import csv
import json
from datetime import date

import numpy as np
import torch

from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkok_spatial_cnn_crossdistrict import DISTRICTS, fit, normalize
from flood_bangkok_spatial_cnn_multiyear import (
    CAL_END, TRAIN_END, district_data, report_rows,
)
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, rainfall_by_date, write_csv


ARTIFACT = "surface_crossdistrict_terrain_cnn_multiyear_local_balance"
BASELINE = "surface_crossdistrict_terrain_cnn_multiyear_eleven"
LOCAL_WEIGHT = 10.0


def main() -> None:
    torch.set_num_threads(4)
    codes = DISTRICTS + MORE_CODES
    districts = {d["code"]: d for d in load_districts() if d["code"] in codes}
    specs = {code: grid_spec(districts[code]["polygon"]) for code in codes}
    height = max(spec["rows"] for spec in specs.values())
    width = max(spec["cols"] for spec in specs.values())
    reports, stations = report_rows(expanded=True), pump_points()
    district_arrays = {}
    for code in codes:
        data, target, addresses, _ = district_data(
            districts[code], reports, stations, height, width)
        district_arrays[code] = data, target, addresses
        print(f"built district {code}", flush=True)
    bk_x, bk_y, (rr, cc) = district_arrays["1006"]
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   TRANSFER_CACHE / "rain_1006_2021_2025.json"))
    train_count = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    if (train_count, cal_end) != (1308, 1369):
        raise ValueError("Unexpected temporal split")
    train_x = np.concatenate([district_arrays[code][0][:train_count] for code in codes])
    train_y = np.concatenate([district_arrays[code][1][:train_count] for code in codes])
    weights = np.concatenate([np.full(train_count, LOCAL_WEIGHT if code == "1006" else 1.0)
                              for code in codes])
    local_sampling_share = float(weights[:train_count].sum() / weights.sum())
    x_cal, y_cal = bk_x[train_count:cal_end], bk_y[train_count:cal_end]
    x_test, y_test = bk_x[cal_end:], bk_y[cal_end:]
    train_norm, cal_norm, test_norm = normalize(train_x, x_cal, x_test)
    result, expected, observed = fit(train_norm, train_y, cal_norm, y_cal,
                                     test_norm, y_test, rr, cc,
                                     training_sample_weights=weights)
    if int(observed.sum()) != 242:
        raise ValueError("Bang Kapi test labels differ from previous runs")
    previous = json.loads((OUTPUT / f"{BASELINE}_metrics.json").read_text())
    old = previous["models"]["eleven_districts_multiyear"]
    if old["training_positive_cell_days"] != int(train_y.sum()):
        raise ValueError("Pooled baseline uses different training labels")
    with (OUTPUT / f"{BASELINE}_cell_comparison.csv").open(
            encoding="utf-8-sig", newline="") as file:
        old_rows = {row["cell_id"]: row for row in csv.DictReader(file)}
    rows = []
    for i in range(len(rr)):
        cid = f"bk:{int(rr[i]):02d}:{int(cc[i]):02d}"
        if int(old_rows[cid]["observed_days_oct_dec"]) != observed[i]:
            raise ValueError("Pooled baseline has different cell outcome")
        rows.append({"cell_id": cid,
                     "pooled_expected_days_oct_dec": old_rows[cid][
                         "eleven_districts_multiyear_expected_days"],
                     "local_balanced_expected_days_oct_dec": round(float(expected[i]), 5),
                     "observed_days_oct_dec": int(observed[i])})
    write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
    metrics = {"target": "100 m direct Traffy report buffer touches a 250 m cell-day",
               "train_dates": ["2022-01-01", "2025-07-31"],
               "selection_and_calibration_dates": ["2025-08-01", "2025-09-30"],
               "test_dates": ["2025-10-01", "2025-12-31"],
               "training_district_codes": list(codes),
               "local_training_sampling_share": round(local_sampling_share, 3),
               "local_weight_per_day": LOCAL_WEIGHT,
               "training_positive_cell_days": int(train_y.sum()),
               "pooled_reference": f"{BASELINE}_metrics.json",
               "model": result,
               "limitation": "Exploratory reuse of October–December 2025; balancing does not establish geographic transfer."}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")
    print(result["test_cell_rank"], result["test_daily"]["average_precision"], flush=True)


if __name__ == "__main__":
    main()
