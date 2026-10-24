"""Audit feature distributions before interpreting cross-district CNN transfer.

Compares the ten source districts' training cells with Bang Khae before the
October–December 2025 evaluation. Labels are 100 m Traffy report buffers.

Run: .venv/bin/python flood_bangkok_feature_shift_audit.py
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np

from flood_bangkok_spatial_cnn_crossdistrict import DISTRICTS
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END, district_data, report_rows
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, RAIN_FEATURES, rainfall_by_date


TARGET = "1040"
ARTIFACT = "surface_crossdistrict_bangkhae_feature_shift"
FEATURES = ["absolute_elevation_m", "elevation_minus_3x3_mean_m", "slope",
            "district_cell_coverage", "nearest_pump_distance_m", "pump_count_1km",
            "nearest_gate_distance_m", "prior_report_days_current_year",
            "month_sin", "month_cos", *RAIN_FEATURES]


def accumulator() -> dict:
    return {"n": 0, "sum": 0.0, "sum_sq": 0.0, "positive_n": 0,
            "positive_sum": 0.0, "negative_sum": 0.0,
            "min": float("inf"), "max": -float("inf")}


def add(stats: dict, values: np.ndarray, labels: np.ndarray) -> None:
    flat = values.ravel().astype(np.float64)
    target = labels.ravel().astype(bool)
    if len(flat) != len(target) or not np.isfinite(flat).all():
        raise ValueError("Feature/label mismatch or nonfinite feature")
    stats["n"] += len(flat)
    stats["sum"] += float(flat.sum())
    stats["sum_sq"] += float(np.square(flat).sum())
    stats["positive_n"] += int(target.sum())
    stats["positive_sum"] += float(flat[target].sum())
    stats["negative_sum"] += float(flat[~target].sum())
    stats["min"] = min(stats["min"], float(flat.min()))
    stats["max"] = max(stats["max"], float(flat.max()))


def summarize(stats: dict) -> dict:
    n, positives = stats["n"], stats["positive_n"]
    mean = stats["sum"] / n
    std = max(float(np.sqrt(max(0, stats["sum_sq"] / n - mean * mean))), 1e-4)
    return {"n": n, "positive_n": positives,
            "mean": round(mean, 6), "std": round(std, 6),
            "positive_mean": round(stats["positive_sum"] / positives, 6)
            if positives else None,
            "negative_mean": round(stats["negative_sum"] / (n - positives), 6)
            if n > positives else None,
            "min": round(stats["min"], 6), "max": round(stats["max"], 6)}


def main() -> None:
    codes = DISTRICTS + MORE_CODES
    districts = {d["code"]: d for d in load_districts() if d["code"] in codes}
    specs = {code: grid_spec(districts[code]["polygon"]) for code in codes}
    height = max(spec["rows"] for spec in specs.values())
    width = max(spec["cols"] for spec in specs.values())
    reports, stations = report_rows(expanded=True), pump_points()
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   TRANSFER_CACHE / "rain_1006_2021_2025.json"))
    train_end = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    groups = {"source_train": [accumulator() for _ in FEATURES],
              "target_train": [accumulator() for _ in FEATURES],
              "target_calibration": [accumulator() for _ in FEATURES]}
    for code in codes:
        data, target, (rr, cc), _ = district_data(
            districts[code], reports, stations, height, width)
        slices = [("target_train", slice(0, train_end)),
                  ("target_calibration", slice(train_end, cal_end))] if code == TARGET else [
                      ("source_train", slice(0, train_end))]
        for group, span in slices:
            labels = target[span, 0][:, rr, cc]
            for channel in range(len(FEATURES)):
                add(groups[group][channel], data[span, channel][:, rr, cc], labels)
        print("audited", code, flush=True)
    summary = {name: {feature: summarize(stats)
                      for feature, stats in zip(FEATURES, values)}
               for name, values in groups.items()}
    shift = {}
    for feature in FEATURES:
        source = summary["source_train"][feature]
        target = summary["target_train"][feature]
        calibration = summary["target_calibration"][feature]
        source_contrast = (source["positive_mean"] - source["negative_mean"]) / source["std"]
        target_contrast = (target["positive_mean"] - target["negative_mean"]) / source["std"]
        calibration_contrast = (calibration["positive_mean"] -
                                calibration["negative_mean"]) / source["std"]
        shift[feature] = {
            "target_vs_source_mean_in_source_sd": round(
                (target["mean"] - source["mean"]) / source["std"], 4),
            "source_positive_minus_negative_in_source_sd": round(source_contrast, 4),
            "target_positive_minus_negative_in_source_sd": round(target_contrast, 4),
            "calibration_positive_minus_negative_in_source_sd": round(
                calibration_contrast, 4),
            "contrast_sign_reversal_target_train": bool(source_contrast * target_contrast < 0),
            "contrast_sign_reversal_target_calibration": bool(
                source_contrast * calibration_contrast < 0)}
    output = {"source_district_codes": [code for code in codes if code != TARGET],
              "target_district_code": TARGET,
              "source_and_target_training_dates": ["2022-01-01", str(TRAIN_END)],
              "target_calibration_dates": ["2025-08-01", str(CAL_END)],
              "channel_index": {feature: index for index, feature in enumerate(FEATURES)},
              "summary": summary, "shift": shift,
              "caution": "Marginal feature contrasts are confounded by geography, time, and report propensity; they do not prove predictive importance or causality."}
    (OUTPUT / f"{ARTIFACT}.json").write_text(json.dumps(output, indent=2) + "\n")
    for feature, item in shift.items():
        print(feature, item, flush=True)


if __name__ == "__main__":
    main()
