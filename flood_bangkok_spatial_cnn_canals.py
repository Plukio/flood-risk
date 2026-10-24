"""Compare a cross-district CNN with mapped canal geometry channels.

The 17-feature baseline gains two January 2025 OSM canal channels: log canal
centreline length in the 250 m cell and log distance to the nearest canal.
Train on ten districts through July 2025, hold Bang Khae out, select on Bang
Kapi August–September, and evaluate Bang Khae October–December. Also train a
Bang Khae-only CNN with the same channels. Labels are Traffy report buffers.

Run flood_bangkok_canal_features.py first, then:
    .venv/bin/python flood_bangkok_spatial_cnn_canals.py
"""

from __future__ import annotations

import gc
import json
from datetime import date

import numpy as np
import torch

from flood_bangkok_spatial_cnn_crossdistrict import CHANNELS, DISTRICTS, fit, normalize
from flood_bangkok_spatial_cnn_multiyear import CAL_END, TRAIN_END, district_data, report_rows
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, rainfall_by_date, write_csv


TARGET = "1040"
ARTIFACT = "surface_crossdistrict_bangkhae_canal_cnn"
CANAL_CHANNELS = 2
FEATURE_CHANNELS = CHANNELS + CANAL_CHANNELS


def add_canal_channels(code: str, base: np.ndarray, addresses: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    path = TRANSFER_CACHE / f"canal_2025_{code}_250m.npz"
    with np.load(path) as saved:
        length = saved["canal_length_m"]
        distance = saved["nearest_canal_m"]
        canal_valid = saved["valid"]
    rr, cc = addresses
    if (length.shape != distance.shape or length.shape != canal_valid.shape or
            not np.all(canal_valid[rr, cc])):
        raise ValueError(f"Canal features do not cover model cells in {code}")
    enhanced = np.zeros((len(base), FEATURE_CHANNELS + 1, *base.shape[2:]),
                        dtype=np.float32)
    enhanced[:, :CHANNELS] = base[:, :CHANNELS]
    h, w = length.shape
    enhanced[:, CHANNELS, :h, :w] = np.log1p(np.maximum(0, length))
    enhanced[:, CHANNELS + 1, :h, :w] = np.log1p(np.clip(distance, 0, 10000))
    enhanced[:, FEATURE_CHANNELS] = base[:, CHANNELS]
    enhanced[:, :FEATURE_CHANNELS] *= enhanced[:, FEATURE_CHANNELS:FEATURE_CHANNELS + 1]
    return enhanced


def main() -> None:
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
        base, target, addresses, _ = district_data(
            districts[code], reports, stations, height, width)
        arrays[code] = add_canal_channels(code, base, addresses), target, addresses
        print("loaded canal channels", code, flush=True)
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   TRANSFER_CACHE / "rain_1006_2021_2025.json"))
    train_end = sum(day <= TRAIN_END for day in days)
    cal_end = sum(day <= CAL_END for day in days)
    if (train_end, cal_end, len(days)) != (1308, 1369, 1461):
        raise ValueError("Unexpected chronological split")
    target_x, target_y, (rr, cc) = arrays[TARGET]
    bk_x, bk_y, (bk_rr, bk_cc) = arrays["1006"]
    source_x = np.concatenate([arrays[code][0][:train_end] for code in source_codes])
    source_y = np.concatenate([arrays[code][1][:train_end] for code in source_codes])
    train_norm, bk_cal, target_cal, target_test = normalize(
        source_x, bk_x[train_end:cal_end], target_x[train_end:cal_end],
        target_x[cal_end:], channel_count=FEATURE_CHANNELS)
    del source_x, arrays
    gc.collect()
    result, source_expected, observed, adapted_expected = fit(
        train_norm, source_y, bk_cal, bk_y[train_end:cal_end],
        target_test, target_y[cal_end:], bk_rr, bk_cc, rr, cc,
        adapt_cal=(target_cal, target_y[train_end:cal_end], rr, cc),
        daily_output_path=OUTPUT / f"{ARTIFACT}_daily.npz")
    print("transfer", result["test_cell_rank"]["spearman"],
          result["test_daily"]["average_precision"],
          result["target_district_recalibration"]["test_daily"][
              "predicted_positive_cell_days"], flush=True)
    del train_norm, bk_cal, target_cal, target_test, source_y
    gc.collect()
    local_train, local_cal, local_test = normalize(
        target_x[:train_end], target_x[train_end:cal_end], target_x[cal_end:],
        channel_count=FEATURE_CHANNELS)
    local_result, local_expected, local_observed = fit(
        local_train, target_y[:train_end], local_cal, target_y[train_end:cal_end],
        local_test, target_y[cal_end:], rr, cc,
        daily_output_path=OUTPUT / f"{ARTIFACT}_local_daily.npz")
    if not np.array_equal(observed, local_observed):
        raise ValueError("Local and transferred test labels disagree")
    print("local", local_result["test_cell_rank"]["spearman"],
          local_result["test_daily"]["average_precision"],
          local_result["test_daily"]["predicted_positive_cell_days"], flush=True)
    baseline = json.loads((OUTPUT / "surface_crossdistrict_bangkhae_partial_finetune_augsep_metrics.json").read_text())
    summary = {
        "target": "100 m direct Traffy flood-report buffer touches a 250 m cell-day",
        "not_target": "verified flood extent or water depth",
        "canal_feature_source": "OpenStreetMap Thailand snapshot 2025-01-01, waterway=canal",
        "canal_channels": ["log1p(canal centreline metres inside 250 m cell)",
                           "log1p(distance metres from cell centre to nearest canal)"],
        "source_training_district_codes": list(source_codes),
        "test_district_code": TARGET,
        "train_dates": ["2022-01-01", str(TRAIN_END)],
        "selection_and_calibration_dates": ["2025-08-01", str(CAL_END)],
        "test_dates": ["2025-10-01", "2025-12-31"],
        "cells": len(rr), "test_positive_cell_days": int(observed.sum()),
        "transfer_with_canals": result,
        "target_only_with_canals": local_result,
        "baseline_without_canals": {
            "source_model": baseline["models"]["source_bang_kapi_aug_sep_calibration"],
            "target_intercept": baseline["models"]["source_no_adaptation"],
            "target_only": baseline["models"]["target_only"]},
        "limitation": "OSM geometry is applied retrospectively; canal flow, level, gate operation, and capacity are unavailable; October–December 2025 reused exploratory test"}
    rows = [{"cell_id": f"{TARGET}:{int(rr[i]):02d}:{int(cc[i]):02d}",
             "mapped_canal_length_in_cell_m": round(float(np.expm1(
                 target_x[0, CHANNELS, rr[i], cc[i]])), 2),
             "distance_to_mapped_canal_m": round(float(np.expm1(
                 target_x[0, CHANNELS + 1, rr[i], cc[i]])), 2),
             "source_canal_expected_days": round(float(source_expected[i]), 6),
             "target_calibrated_canal_expected_days": round(float(adapted_expected[i]), 6),
             "local_canal_expected_days": round(float(local_expected[i]), 6),
             "observed_days_oct_dec": int(observed[i])}
            for i in range(len(rr))]
    write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
