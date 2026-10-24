"""Test partial CNN fine-tuning on an unseen Bangkok district.

Pretrain on ten districts through July 2025, holding Bang Khae out. Fine-tune
only the 1x1 output heads, or the last spatial block plus those heads, using
Bang Khae 2022–July 2025 report-proxy labels. Select adaptation epoch and
calibrate on August–September, then evaluate October–December 2025.
Compare the same source checkpoint with no adaptation and a Bang Khae-only CNN.
Fine-tuning uses either all available earlier years or just January–July 2025.
The reused test is exploratory and measures reports, not verified flooding.

Run: .venv/bin/python flood_bangkok_spatial_cnn_partial_finetune.py
For the identical source checkpoint from the August/September split pilot:
    .venv/bin/python flood_bangkok_spatial_cnn_partial_finetune.py --reuse-source-checkpoint
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import date

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from flood_bangkok_spatial_cnn_crossdistrict import (
    CHANNELS, DISTRICTS, POSITIVE_WEIGHT, SEED, fit, normalize,
)
from flood_bangkok_spatial_cnn_multiyear import (
    CAL_END, TRAIN_END, district_data, report_rows,
)
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkok_transfer import CACHE as TRANSFER_CACHE, load_districts
from flood_bangkapi_proxy_probability import apply_calibration, calibrate_intercept, evaluation
from flood_bangkapi_spatial_cnn import SpatialCNN, logits_for
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import OUTPUT, rainfall_by_date, rank_metrics, write_csv


TARGET = "1040"  # Bang Khae
ARTIFACT = "surface_crossdistrict_bangkhae_partial_finetune_augsep"
PILOT_ARTIFACT = "surface_crossdistrict_bangkhae_partial_finetune"
MAX_ADAPT_EPOCHS = 16
ADAPT_PATIENCE = 5


def assess(model: SpatialCNN, x_cal: np.ndarray, y_cal: np.ndarray,
           x_test: np.ndarray, y_test: np.ndarray,
           rr: np.ndarray, cc: np.ndarray) -> tuple[dict, np.ndarray]:
    raw_cal = expit(logits_for(model, x_cal)[:, 0, rr, cc])
    raw_test = expit(logits_for(model, x_test)[:, 0, rr, cc])
    truth = y_test[:, 0, rr, cc].astype(np.uint8)
    bias = calibrate_intercept(raw_cal.ravel(), y_cal[:, 0, rr, cc].ravel())
    probability = apply_calibration(raw_test, bias)
    metrics = {
        "target_calibration_positive_cell_days": int(y_cal[:, 0, rr, cc].sum()),
        "target_raw_predicted_positive_cell_days": round(float(raw_cal.sum()), 1),
        "target_log_odds_shift": round(bias, 4),
        "test_daily": evaluation(probability.ravel(), truth.ravel()),
        "test_cell_rank": rank_metrics(probability.sum(axis=0), truth.sum(axis=0)),
        "daily_count_mae": round(float(np.mean(np.abs(
            probability.sum(axis=1) - truth.sum(axis=1)))), 4),
    }
    return metrics, probability


def adapt(checkpoint: dict, train_x: np.ndarray, train_y: np.ndarray,
          val_x: np.ndarray, val_y: np.ndarray,
          rr: np.ndarray, cc: np.ndarray, mode: str) -> tuple[SpatialCNN, dict]:
    torch.manual_seed(SEED)
    model = SpatialCNN(CHANNELS + 1)
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    core_mode = mode.removesuffix("_recent")
    modules = [model.context[4], model.local]
    if core_mode == "last_block_and_heads":
        modules.insert(0, model.context[2])
    elif core_mode != "heads_only":
        raise ValueError(mode)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4 if core_mode == "last_block_and_heads" else 2e-4,
        weight_decay=0.001)
    rng = np.random.default_rng(SEED)
    val_truth = val_y[:, 0, rr, cc].ravel()
    def validation_ap() -> float:
        scores = expit(logits_for(model, val_x)[:, 0, rr, cc]).ravel()
        return float(average_precision_score(val_truth, scores))
    best_ap = validation_ap()
    best_epoch = 0
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history = [{"epoch": 0, "calibration_ap": round(best_ap, 6)}]
    stale = 0
    for epoch in range(1, MAX_ADAPT_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), 12):
            ids = order[start:start + 12]
            x = torch.from_numpy(train_x[ids])
            y = torch.from_numpy(train_y[ids])
            mask = x[:, CHANNELS:CHANNELS + 1]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(x), y, pos_weight=torch.tensor(POSITIVE_WEIGHT), reduction="none")
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
        ap = validation_ap()
        history.append({"epoch": epoch, "calibration_ap": round(ap, 6)})
        if ap > best_ap + 1e-6:
            best_ap = ap
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 4 == 0:
            print(mode, "epoch", epoch, "calibration_AP", round(ap, 5),
                  "best", round(best_ap, 5), flush=True)
        if stale >= ADAPT_PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {"selected_epoch": best_epoch,
                   "target_calibration_ap_for_selection": round(best_ap, 5),
                   "trainable_layers": ["context.2", "context.4", "local"]
                   if core_mode == "last_block_and_heads" else ["context.4", "local"],
                   "train_positive_cell_days": int(train_y[:, 0, rr, cc].sum()),
                   "training_curve": history}


def main(reuse_source_checkpoint: bool = False) -> None:
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
    august_end = sum(day <= date(2025, 8, 31) for day in days)
    september_end = sum(day <= CAL_END for day in days)
    if (train_end, august_end, september_end, len(days)) != (1308, 1339, 1369, 1461):
        raise ValueError("Unexpected split lengths")
    target_x, target_y, (rr, cc) = arrays[TARGET]
    bk_x, bk_y, (bk_rr, bk_cc) = arrays["1006"]
    source_x = np.concatenate([arrays[code][0][:train_end] for code in source_codes])
    source_y = np.concatenate([arrays[code][1][:train_end] for code in source_codes])
    source_norm, bk_cal_norm, target_train, target_cal, target_test = normalize(
        source_x, bk_x[train_end:september_end], target_x[:train_end],
        target_x[train_end:september_end], target_x[september_end:])
    if reuse_source_checkpoint:
        checkpoint_path = OUTPUT / f"{PILOT_ARTIFACT}_source_checkpoint.pt"
        pilot_metrics = json.loads((OUTPUT / f"{PILOT_ARTIFACT}_metrics.json").read_text())
        source_result = pilot_metrics["models"]["source_bang_kapi_aug_sep_calibration"]
        print("reusing identical source checkpoint", checkpoint_path, flush=True)
    else:
        checkpoint_path = OUTPUT / f"{ARTIFACT}_source_checkpoint.pt"
        source_result, _, _ = fit(
            source_norm, source_y, bk_cal_norm, bk_y[train_end:september_end],
            target_test, target_y[september_end:], bk_rr, bk_cc, rr, cc,
            checkpoint_output_path=checkpoint_path)
        print("source pretraining complete", source_result["selected_epoch"], flush=True)
    del source_x, source_y, source_norm, bk_cal_norm, arrays
    gc.collect()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_model = SpatialCNN(CHANNELS + 1)
    source_model.load_state_dict(checkpoint["state_dict"])
    y_cal = target_y[train_end:september_end]
    y_test = target_y[september_end:]
    results = {}
    source_metrics, source_p = assess(source_model, target_cal, y_cal, target_test,
                                      y_test, rr, cc)
    results["source_no_adaptation"] = source_metrics
    results["source_bang_kapi_aug_sep_calibration"] = source_result
    predictions = {"source_no_adaptation": source_p.sum(axis=0)}
    recent_start = days.index(date(2025, 1, 1))
    for mode in ("heads_only", "last_block_and_heads",
                 "heads_only_recent", "last_block_and_heads_recent"):
        start = recent_start if mode.endswith("_recent") else 0
        model, training = adapt(checkpoint, target_train[start:], target_y[start:train_end],
                                target_cal, y_cal, rr, cc, mode)
        training["adaptation_training_dates"] = [str(days[start]), str(TRAIN_END)]
        metrics, p = assess(model, target_cal, y_cal, target_test, y_test, rr, cc)
        results[mode] = {**training, **metrics}
        predictions[mode] = p.sum(axis=0)
        print(mode, "selected", training["selected_epoch"],
              "rho", metrics["test_cell_rank"]["spearman"],
              "AP", metrics["test_daily"]["average_precision"],
              "count", metrics["test_daily"]["predicted_positive_cell_days"], flush=True)
    del target_train, target_cal, target_test
    gc.collect()
    local_train, local_cal, local_test = normalize(
        target_x[:train_end], target_x[train_end:september_end], target_x[september_end:])
    local_result, _, _ = fit(
        local_train, target_y[:train_end], local_cal, y_cal,
        local_test, y_test, rr, cc)
    results["target_only"] = {
        "selected_epoch": local_result["selected_epoch"],
        "target_calibration_ap_for_selection": local_result["calibration_AP_for_selection"],
        "target_log_odds_shift": local_result["calibration_log_odds_shift"],
        "test_daily": local_result["test_daily"],
        "test_cell_rank": local_result["test_cell_rank"]}
    print("target_only", results["target_only"]["test_cell_rank"]["spearman"],
          results["target_only"]["test_daily"]["average_precision"],
          results["target_only"]["test_daily"]["predicted_positive_cell_days"],
          flush=True)
    observed = y_test[:, 0, rr, cc].sum(axis=0)
    rows = [{"cell_id": f"{TARGET}:{int(rr[i]):02d}:{int(cc[i]):02d}",
             **{f"{name}_expected_days": round(float(values[i]), 6)
                for name, values in predictions.items()},
             "observed_days_oct_dec": int(observed[i])}
            for i in range(len(rr))]
    write_csv(OUTPUT / f"{ARTIFACT}_cell_comparison.csv", rows)
    metrics = {
        "target": "100 m direct Traffy report buffer touches a 250 m cell-day",
        "not_target": "verified surface inundation",
        "source_training_district_codes": list(source_codes),
        "target_district": TARGET,
        "source_training_dates": ["2022-01-01", "2025-07-31"],
        "target_only_training_dates": ["2022-01-01", "2025-07-31"],
        "source_selection_dates": ["2025-08-01", "2025-09-30"],
        "target_adaptation_selection_dates": ["2025-08-01", "2025-09-30"],
        "target_calibration_dates": ["2025-08-01", "2025-09-30"],
        "test_dates": ["2025-10-01", "2025-12-31"],
        "target_cells": len(rr),
        "target_calibration_positive_cell_days": int(y_cal[:, 0, rr, cc].sum()),
        "test_positive_cell_days": int(observed.sum()),
        "source_selected_epoch": checkpoint["selected_epoch"],
        "models": results,
        "limitation": "August–September is reused for checkpoint selection and intercept fitting; October–December 2025 is a reused exploratory test; labels are report buffers, not measured flood extent"}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        metrics, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-source-checkpoint", action="store_true")
    args = parser.parse_args()
    main(args.reuse_source_checkpoint)
