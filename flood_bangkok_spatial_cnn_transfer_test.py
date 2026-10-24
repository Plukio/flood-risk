"""Exploratory spatial transfer: train on ten districts, test Bang Khae.

No Bang Khae cell-day labels are used to fit or select the CNN. The model can
use Bang Khae's past report count as an input feature. We compare source-only
intercept calibration on Bang Kapi Aug–Sep 2025 with target-aware intercept
calibration on Bang Khae in the same period. Bang Khae Oct–Dec is the test.
The outcome is a 100 m Traffy report buffer, not verified inundation.

Run: .venv/bin/python flood_bangkok_spatial_cnn_transfer_test.py
"""

from __future__ import annotations

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
from flood_bangkapi_surface_model import OUTPUT, rainfall_by_date, rank_metrics, write_csv


TEST_CODE = "1040"  # Bang Khae; west of the original Bang Kapi study area
ARTIFACT = "surface_crossdistrict_terrain_cnn_bangkhae_transfer"


def main() -> None:
    torch.set_num_threads(4)
    codes = DISTRICTS + MORE_CODES
    training_codes = tuple(code for code in codes if code != TEST_CODE)
    districts = {d["code"]: d for d in load_districts() if d["code"] in codes}
    specs = {code: grid_spec(districts[code]["polygon"]) for code in codes}
    height = max(spec["rows"] for spec in specs.values())
    width = max(spec["cols"] for spec in specs.values())
    reports = report_rows(expanded=True)
    stations = pump_points()
    district_arrays = {}
    for code in codes:
        data, target, addresses, audit = district_data(
            districts[code], reports, stations, height, width)
        district_arrays[code] = data, target, addresses
        print(f"district {code}: {sum(item[f'positive_cell_days_{year}'] for year, item in audit.items())} "
              "positive cell-days, 2022–2025", flush=True)
    bk_x, bk_y, (bk_rr, bk_cc) = district_arrays["1006"]
    target_x, target_y, (rr, cc) = district_arrays[TEST_CODE]
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   TRANSFER_CACHE / "rain_1006_2021_2025.json"))
    train_count = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    if (train_count, cal_end) != (1308, 1369):
        raise ValueError("Unexpected temporal split")
    test_observed = target_y[cal_end:, 0, rr, cc].sum(axis=0)
    historical = target_y[days.index(date(2025, 1, 1)):cal_end, 0, rr, cc].sum(axis=0)
    results = {"historical_baseline": rank_metrics(historical, test_observed)}
    predictions = {}
    adapted_predictions = {}
    for name, start in (("ten_districts_2025_only", days.index(date(2025, 1, 1))),
                        ("ten_districts_multiyear", 0)):
        train_x = np.concatenate([district_arrays[code][0][start:train_count]
                                  for code in training_codes])
        train_y = np.concatenate([district_arrays[code][1][start:train_count]
                                  for code in training_codes])
        x_cal, y_cal = bk_x[train_count:cal_end], bk_y[train_count:cal_end]
        x_adapt, y_adapt = target_x[train_count:cal_end], target_y[train_count:cal_end]
        x_test, y_test = target_x[cal_end:], target_y[cal_end:]
        train_norm, cal_norm, test_norm, adapt_norm = normalize(
            train_x, x_cal, x_test, x_adapt)
        result, expected, observed, adapted_expected = fit(
            train_norm, train_y, cal_norm, y_cal, test_norm, y_test,
            bk_rr, bk_cc, rr, cc,
            adapt_cal=(adapt_norm, y_adapt, rr, cc),
            daily_output_path=OUTPUT / f"{ARTIFACT}_{name}_daily.npz")
        result["training_district_codes"] = list(training_codes)
        result["training_positive_cell_days"] = int(train_y.sum())
        results[name] = result
        predictions[name] = expected
        adapted_predictions[name] = adapted_expected
        print(name, result["test_cell_rank"],
              result["test_daily"]["average_precision"],
              "target_cal_count", result["target_district_recalibration"]["test_daily"][
                  "predicted_positive_cell_days"], flush=True)
        del train_x, train_y, train_norm, cal_norm, test_norm, adapt_norm
    rows = [{"cell_id": f"{TEST_CODE}:{int(rr[i]):02d}:{int(cc[i]):02d}",
             "historical_positive_days_jan_sep": int(historical[i]),
             "source_2025_only_expected_days_oct_dec": round(float(predictions[
                 "ten_districts_2025_only"][i]), 5),
             "source_multiyear_expected_days_oct_dec": round(float(predictions[
                 "ten_districts_multiyear"][i]), 5),
             "source_2025_target_recalibrated_expected_days": round(float(adapted_predictions[
                 "ten_districts_2025_only"][i]), 5),
             "source_multiyear_target_recalibrated_expected_days": round(float(
                 adapted_predictions["ten_districts_multiyear"][i]), 5),
             "observed_days_oct_dec": int(test_observed[i])}
            for i in range(len(rr))]
    write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
    metrics = {"target": "100 m direct Traffy flood-report buffer touches a 250 m cell-day",
               "not_target": "verified surface inundation",
               "training_district_codes": list(training_codes),
               "test_district": {"code": TEST_CODE, "name": districts[TEST_CODE]["name_en"],
                                 "cells": len(rr)},
               "train_dates": ["2022-01-01 or 2025-01-01", "2025-07-31"],
               "selection_and_calibration_area": "Bang Kapi",
               "selection_and_calibration_dates": ["2025-08-01", "2025-09-30"],
               "target_recalibration_area": "Bang Khae",
               "target_recalibration_dates": ["2025-08-01", "2025-09-30"],
               "test_dates": ["2025-10-01", "2025-12-31"],
               "test_district_past_report_counts_used_as_features": True,
               "test_district_labels_used_to_fit_or_select": False,
               "test_district_aug_sep_labels_used_for_recalibration_variant": True,
               "models": results,
               "limitation": "Exploratory target district chosen after inspecting available label counts; source-district reporting patterns and gridded rain may differ."}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
