"""Retrain source CNNs without suspect feature groups and test Bang Khae.

Pretrain on ten districts through July 2025; select checkpoints only on Bang
Kapi August–September; assess feature transfer on Bang Khae August–September
and the reused October–December test. All zeros from feature masking are in
the source-normalized space. The target is a report buffer, not verified water.

Run: .venv/bin/python flood_bangkok_feature_ablation.py
"""

from __future__ import annotations

import argparse
import gc
import csv
import json
from datetime import date

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score

from flood_bangkok_spatial_cnn_crossdistrict import CHANNELS, DISTRICTS, fit, normalize
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END, district_data, report_rows
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkapi_spatial_cnn import SpatialCNN, logits_for
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, rainfall_by_date, write_csv


TARGET = "1040"
ARTIFACT = "surface_crossdistrict_bangkhae_feature_ablation"
CHECKPOINT = OUTPUT / "surface_crossdistrict_bangkhae_partial_finetune_source_checkpoint.pt"
VARIANTS = {
    "without_absolute_elevation": (0,),
    "without_pump_gate_inventory": (4, 5, 6),
    "without_prior_report_count": (7,),
    "without_calendar": (8, 9),
    "without_rain": (10, 11, 12, 13, 14, 15, 16),
}


def main(only: tuple[str, ...] = ()) -> None:
    torch.set_num_threads(4)
    codes = DISTRICTS + MORE_CODES
    source_codes = tuple(code for code in codes if code != TARGET)
    districts = {d["code"]: d for d in load_districts() if d["code"] in codes}
    specs = {code: grid_spec(districts[code]["polygon"]) for code in codes}
    height = max(spec["rows"] for spec in specs.values())
    width = max(spec["cols"] for spec in specs.values())
    reports, stations = report_rows(expanded=True), pump_points()
    arrays = {}
    for code in codes:
        arrays[code] = district_data(districts[code], reports, stations, height, width)[:3]
        print("loaded", code, flush=True)
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   TRANSFER_CACHE / "rain_1006_2021_2025.json"))
    train_end = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    target_x, target_y, (rr, cc) = arrays[TARGET]
    bk_x, bk_y, (bk_rr, bk_cc) = arrays["1006"]
    source_x = np.concatenate([arrays[code][0][:train_end] for code in source_codes])
    source_y = np.concatenate([arrays[code][1][:train_end] for code in source_codes])
    train_norm, bk_cal, target_cal, target_test = normalize(
        source_x, bk_x[train_end:cal_end], target_x[train_end:cal_end],
        target_x[cal_end:])
    del source_x, arrays
    gc.collect()
    y_cal = target_y[train_end:cal_end]
    y_test = target_y[cal_end:]
    model = SpatialCNN(CHANNELS + 1)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)[
        "state_dict"])
    raw_cal = expit(logits_for(model, target_cal)[:, 0, rr, cc])
    baseline_cal_ap = float(average_precision_score(y_cal[:, 0, rr, cc].ravel(),
                                                     raw_cal.ravel()))
    reference = json.loads((OUTPUT / "surface_crossdistrict_bangkhae_partial_finetune_augsep_metrics.json").read_text())
    rows = [{"cell_id": f"{TARGET}:{int(rr[i]):02d}:{int(cc[i]):02d}",
             "observed_days_oct_dec": int(y_test[:, 0, rr[i], cc[i]].sum())}
            for i in range(len(rr))]
    result = {
        "target": "100 m direct Traffy report buffer touches a 250 m cell-day",
        "not_target": "verified flood extent",
        "source_district_codes": list(source_codes),
        "target_district_code": TARGET,
        "train_dates": ["2022-01-01", str(TRAIN_END)],
        "selection_dates": ["2025-08-01", str(CAL_END)],
        "test_dates": ["2025-10-01", "2025-12-31"],
        "baseline": {"target_calibration_AP": round(baseline_cal_ap, 5),
                     "source_calibrated": reference["models"][
                         "source_bang_kapi_aug_sep_calibration"],
                     "target_intercept": reference["models"]["source_no_adaptation"]},
        "variants": {},
        "limitation": "Chosen after inspecting distribution shifts; October–December 2025 reused exploratory test; comparisons are not independent confirmation of feature causality"}
    if only and (OUTPUT / f"{ARTIFACT}_metrics.json").exists():
        result = json.loads((OUTPUT / f"{ARTIFACT}_metrics.json").read_text())
        with (OUTPUT / f"{ARTIFACT}_cell_comparison.csv").open(
                encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if [row["cell_id"] for row in rows] != [
                f"{TARGET}:{int(rr[i]):02d}:{int(cc[i]):02d}" for i in range(len(rr))]:
            raise ValueError("Existing ablation rows do not match this grid")
    for name, channels in VARIANTS.items():
        if only and name not in only:
            continue
        metrics, expected, observed, adapted_expected = fit(
            train_norm, source_y, bk_cal, bk_y[train_end:cal_end],
            target_test, y_test, bk_rr, bk_cc, rr, cc,
            adapt_cal=(target_cal, y_cal, rr, cc), masked_channels=channels)
        result["variants"][name] = {"masked_channels": list(channels), **metrics}
        for row, source_value, target_value in zip(rows, expected, adapted_expected):
            row[f"{name}_source_calibrated_expected_days"] = round(float(source_value), 6)
            row[f"{name}_target_intercept_expected_days"] = round(float(target_value), 6)
        (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
            result, ensure_ascii=False, indent=2) + "\n")
        write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
        print(name, "BK selection AP", metrics["calibration_AP_for_selection"],
              "Bang Khae cal AP", metrics["target_district_recalibration"]["calibration_AP"],
              "test rho", metrics["target_district_recalibration"]["test_cell_rank"]["spearman"],
              "test AP", metrics["target_district_recalibration"]["test_daily"]["average_precision"],
              "test count", metrics["target_district_recalibration"]["test_daily"][
                  "predicted_positive_cell_days"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(VARIANTS))
    args = parser.parse_args()
    main(tuple(args.only or ()))
