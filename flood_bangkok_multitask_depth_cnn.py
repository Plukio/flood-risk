"""Exploratory joint complaint/depth CNN with conservative BMA site matches.

Run: .venv/bin/python flood_bangkok_multitask_depth_cnn.py
Depth is conditional on a matched BMA major-road report; missing is unknown.
"""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import torch
from shapely.geometry import Point
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from flood_bangkok_spatial_cnn_crossdistrict import CHANNELS, normalize
from flood_bangkok_spatial_cnn_multiyear import district_data, report_rows
from flood_bangkok_traffy_history import MORE_CODES
from flood_bangkok_spatial_cnn_crossdistrict import DISTRICTS
from flood_bangkok_transfer import CACHE, load_districts, write_csv
from flood_bangkapi_surface_grid import OUTPUT, TO_UTM, grid_spec, pump_points
from flood_bangkapi_surface_model import rainfall_by_date


CODES = DISTRICTS + MORE_CODES
ARTIFACT = "surface_multitask_bma_depth_cnn"
TRAIN_END = date(2025, 7, 31)
VAL_END = date(2025, 9, 30)
SEED = 7
SAMPLED_DAYS_PER_DISTRICT = 180
EPOCHS = 30
STEPS_PER_EPOCH = 64
BATCH_SIZE = 8
DEPTH_SHARE = 0.5
DEPTH_LOSS_WEIGHT = 1.0


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower().strip()
    value = value.replace("บ.", "บริษัท").replace("ซ.", "ซอย").replace("ถ.", "ถนน")
    return re.sub(r"[\s\-–—,()./\\]+", "", value)


def matched_depths(districts: dict[str, dict]) -> tuple[dict, list[dict], dict]:
    """Only unique exact area/name matches in the same district are retained."""
    by_name = {d["name_th"]: d for d in districts.values()}
    inventory = defaultdict(list)
    for row in csv_rows(CACHE / "flood_risk26.csv"):
        district = row["DISTRICT_NAME"].strip()
        if district in by_name:
            inventory[(district, normalize_name(row["STATION_NAME"]))].append(row)
    matched, audit = [], defaultdict(int)
    by_cell_day: dict[tuple[str, date, int, int], list[float]] = defaultdict(list)
    for year in range(2022, 2026):
        for row in csv_rows(CACHE / f"bma_flood_{year}.csv"):
            district = by_name.get(row["district"].replace("เขต", "").strip())
            if district is None or not row["frd_flood_height"].strip():
                continue
            day = datetime.strptime(row["frd_flood_date_begin"], "%m/%d/%Y").date()
            if day.year != year:
                continue
            audit["BMA_rows_with_depth_in_selected_districts"] += 1
            area = normalize_name(row["frd_area"])
            combined = normalize_name(row["frd_road"] + row["frd_area"])
            options = {id(item): item for name in (area, combined) if len(name) >= 8
                       for item in inventory.get((district["name_th"], name), [])}
            if len(options) != 1:
                audit["not_uniquely_exact_matched"] += 1
                continue
            point = next(iter(options.values()))
            lon, lat = float(point["UTM_X"]), float(point["UTM_Y"])
            if not (100 < lon < 101 and 13 < lat < 14):
                raise ValueError("Inventory fields UTM_X/Y are not longitude/latitude")
            x, y = TO_UTM.transform(lon, lat)
            if not district["polygon"].covers(Point(x, y)):
                audit["matched_point_outside_district"] += 1
                continue
            spec = grid_spec(district["polygon"])
            rr = math.floor((spec["north"] - y) / 250)
            cc = math.floor((x - spec["west"]) / 250)
            depth = float(row["frd_flood_height"])
            if not np.isfinite(depth) or depth <= 0:
                raise ValueError("Invalid positive BMA depth")
            key = (district["code"], day, rr, cc)
            by_cell_day[key].append(depth)
            matched.append({"district_code": district["code"], "date": str(day),
                            "report_id": row["frd_id"], "road": row["frd_road"],
                            "area_text": row["frd_area"],
                            "inventory_name": point["STATION_NAME"],
                            "inventory_lon": lon, "inventory_lat": lat,
                            "grid_row": rr, "grid_col": cc, "depth_cm": depth,
                            "split": "train" if day <= TRAIN_END else (
                                "validation" if day <= VAL_END else "test")})
    audit["exact_matched_rows"] = len(matched)
    audit["unique_matched_cell_days"] = len(by_cell_day)
    target = {key: max(values) for key, values in by_cell_day.items()}
    return target, matched, dict(audit)


def build_arrays(depths: dict, districts: dict[str, dict]):
    days = sorted(rainfall_by_date(date(2022, 1, 1), date(2025, 12, 31),
                                   CACHE / "rain_1006_2021_2025.json"))
    specs = {code: grid_spec(districts[code]["polygon"]) for code in CODES}
    height = max(spec["rows"] for spec in specs.values())
    width = max(spec["cols"] for spec in specs.values())
    reports, stations = report_rows(expanded=True), pump_points()
    train_x, train_cls, train_depth = [], [], []
    val_x, val_cls, val_depth = [], [], []
    test_x, test_cls, test_depth = [], [], []
    test_keys, val_keys = [], []
    rng = np.random.default_rng(SEED)
    label_count = defaultdict(int)
    by_code = defaultdict(dict)
    for (code, day, r, c), value in depths.items():
        by_code[code].setdefault(day, []).append((r, c, value))
    for code in CODES:
        district = districts[code]
        raw, cls, (rr, cc), _ = district_data(
            district, reports, stations, height, width)
        if len(raw) != len(days):
            raise ValueError("District and reference rain dates differ")
        valid = set(zip(rr, cc))
        depth_for_code = by_code.get(code, {})
        for day, locations in depth_for_code.items():
            for r, c, _ in locations:
                if (r, c) not in valid:
                    raise ValueError(f"Matched point is outside valid grid: {code} {day}")
        train_pool = np.asarray([i for i, day in enumerate(days) if day <= TRAIN_END])
        selected = set(rng.choice(train_pool, size=SAMPLED_DAYS_PER_DISTRICT,
                                  replace=False))
        selected.update(i for i in train_pool if days[i] in depth_for_code)
        selected = sorted(selected)
        val_indices = [i for i, day in enumerate(days) if TRAIN_END < day <= VAL_END]
        test_indices = [i for i, day in enumerate(days) if day > VAL_END]
        for indices, xs, cs, ds, keys in (
            (selected, train_x, train_cls, train_depth, None),
            (val_indices, val_x, val_cls, val_depth, val_keys),
            (test_indices, test_x, test_cls, test_depth, test_keys),
        ):
            for i in indices:
                day = days[i]
                depth_grid = np.full((height, width), np.nan, dtype=np.float32)
                for r, c, value in depth_for_code.get(day, []):
                    depth_grid[r, c] = value
                    label_count["train" if day <= TRAIN_END else (
                        "validation" if day <= VAL_END else "test")] += 1
                xs.append(raw[i])
                cs.append(cls[i, 0])
                ds.append(depth_grid)
                if keys is not None:
                    keys.append((code, str(day)))
        print("prepared", code, "sampled train days", len(selected),
              "matched depth days", len(depth_for_code), flush=True)
        del raw, cls
    train_x = np.asarray(train_x, dtype=np.float32)
    val_x = np.asarray(val_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    targets = {
        "train": (np.asarray(train_cls, dtype=np.float32),
                  np.asarray(train_depth, dtype=np.float32)),
        "validation": (np.asarray(val_cls, dtype=np.float32),
                       np.asarray(val_depth, dtype=np.float32)),
        "test": (np.asarray(test_cls, dtype=np.float32),
                 np.asarray(test_depth, dtype=np.float32)),
    }
    return (train_x, val_x, test_x), targets, test_keys, val_keys, dict(label_count)


class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(CHANNELS + 1, 12, 3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1), nn.ReLU())
        self.classification = nn.Conv2d(12, 1, 1)
        self.depth = nn.Conv2d(12, 1, 1)
        nn.init.zeros_(self.depth.weight)
        nn.init.zeros_(self.depth.bias)

    def forward(self, x):
        latent = self.backbone(x)
        return self.classification(latent), 15.0 + 5.0 * self.depth(latent)


@torch.no_grad()
def predict(net, x: np.ndarray, batch_size: int = 24):
    net.eval()
    cls, depth = [], []
    for start in range(0, len(x), batch_size):
        a, b = net(torch.from_numpy(x[start:start + batch_size]))
        cls.append(torch.sigmoid(a).numpy()[:, 0])
        depth.append(b.numpy()[:, 0])
    return np.concatenate(cls), np.concatenate(depth)


def classification_ap(prob, target, x):
    mask = x[:, CHANNELS] > 0
    return float(average_precision_score(target[mask].ravel(), prob[mask].ravel()))


def depth_mae(pred, target):
    mask = np.isfinite(target)
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def train_one(x_train, y_train, d_train, x_val, y_val, d_val,
              multitask: bool):
    torch.manual_seed(SEED)
    net = MultiTaskCNN()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.02)
    rng = np.random.default_rng(SEED)
    depth_indices = np.flatnonzero(np.isfinite(d_train).any(axis=(1, 2)))
    if len(depth_indices) < 20:
        raise ValueError("Too few matched depth days for experiment")
    history = []
    best_ap, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        net.train()
        losses = []
        for _ in range(STEPS_PER_EPOCH):
            random_ids = rng.integers(len(x_train), size=BATCH_SIZE // 2)
            selected_depth = rng.choice(depth_indices, size=BATCH_SIZE // 2,
                                        replace=True)
            ids = np.concatenate((random_ids, selected_depth))
            xb = torch.from_numpy(x_train[ids])
            yb = torch.from_numpy(y_train[ids, None])
            db = torch.from_numpy(d_train[ids, None])
            logits, predicted_depth = net(xb)
            valid = xb[:, CHANNELS:CHANNELS + 1]
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=torch.tensor(30.0), reduction="none")
            cls_loss = (cls_loss * valid).sum() / valid.sum()
            if multitask:
                depth_mask = torch.isfinite(db)
                depth_loss = F.smooth_l1_loss(
                    predicted_depth[depth_mask], db[depth_mask],
                    beta=5.0) / 10.0
                loss = cls_loss + DEPTH_LOSS_WEIGHT * depth_loss
            else:
                loss = cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        probability, predicted_depth = predict(net, x_val)
        ap = classification_ap(probability, y_val, x_val)
        mae = depth_mae(predicted_depth, d_val)
        history.append({"epoch": epoch, "training_loss": round(float(np.mean(losses)), 5),
                        "validation_classification_AP": round(ap, 5),
                        "validation_depth_MAE_cm": round(mae, 3)})
        if ap > best_ap + 1e-6:
            best_ap = ap
            best_epoch = epoch
            best_state = {key: value.detach().clone()
                          for key, value in net.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print("joint" if multitask else "classification_only", epoch,
                  "val AP", round(ap, 5), "val depth MAE", round(mae, 3),
                  flush=True)
    net.load_state_dict(best_state)
    return net, {"selected_epoch_by_validation_classification_AP": best_epoch,
                 "best_validation_classification_AP": round(best_ap, 5),
                 "training_history": history}


def main() -> None:
    torch.set_num_threads(4)
    districts = {d["code"]: d for d in load_districts() if d["code"] in CODES}
    depths, match_rows, match_audit = matched_depths(districts)
    write_csv(OUTPUT / f"{ARTIFACT}_matched_records.csv", match_rows)
    (x_train, x_val, x_test), targets, test_keys, _, label_count = build_arrays(
        depths, districts)
    y_train, d_train = targets["train"]
    y_val, d_val = targets["validation"]
    y_test, d_test = targets["test"]
    train_median = float(np.nanmedian(d_train))
    baseline_mae = float(np.nanmean(np.abs(d_test - train_median)))
    val_baseline_mae = float(np.nanmean(np.abs(d_val - train_median)))
    valid_test = x_test[:, CHANNELS] > 0
    test_prevalence = float(y_test[valid_test].mean())
    results, output_rows = {}, []
    for name, joint in (("classification_only", False), ("joint_classification_depth", True)):
        net, fitted = train_one(x_train, y_train, d_train, x_val, y_val, d_val,
                                multitask=joint)
        probability, predicted_depth = predict(net, x_test)
        observed = np.isfinite(d_test)
        depth_true = d_test[observed]
        depth_pred = predicted_depth[observed]
        metrics = {
            "Q4_classification_AP_all_valid_cells": round(
                classification_ap(probability, y_test, x_test), 5),
            "Q4_depth_MAE_cm_on_matched_BMA_cells": round(
                float(np.mean(np.abs(depth_pred - depth_true))), 3) if joint else None,
            "Q4_depth_RMSE_cm_on_matched_BMA_cells": round(
                float(np.sqrt(np.mean((depth_pred - depth_true)**2))), 3) if joint else None,
            "Q4_depth_observations": int(len(depth_true)), **fitted}
        metrics["minimum_validation_depth_MAE_cm_any_epoch"] = min(
            row["validation_depth_MAE_cm"] for row in fitted["training_history"])
        results[name] = metrics
        if joint:
            for i, (code, day) in enumerate(test_keys):
                for r, c in zip(*np.where(observed[i])):
                    output_rows.append({"district_code": code, "date": day,
                                        "grid_row": int(r), "grid_col": int(c),
                                        "observed_BMA_depth_cm": float(d_test[i, r, c]),
                                        "predicted_conditional_depth_cm": round(float(
                                            predicted_depth[i, r, c]), 3),
                                        "classification_report_proxy_score": round(float(
                                            probability[i, r, c]), 6)})
        print(name, {key: value for key, value in metrics.items()
                     if key != "training_history"}, flush=True)
    result = {
        "purpose": "Exploratory shared-CNN multitask fit: Traffy complaint classification and conditional BMA road-flood depth",
        "split": {"train_through": str(TRAIN_END), "validation_through": str(VAL_END),
                  "test": "2025-10-01 to 2025-12-31"},
        "BMA_match_rule": "Unique exact normalized area or road+area text to 2026 risk-point name, same district, point within district polygon",
        "location_limit": "2026 point inventory postdates all labels; exact name is not verified event coordinate",
        "missing_depth_rule": "Only matched BMA report cell-days receive depth loss; other cells are unknown depth, not zero",
        "depth_target": "Maximum recorded BMA depth in cm if multiple matched rows share cell-day; conditional on a BMA major-road report",
        "classification_target": "100 m Traffy direct flood-report buffer touching a 250 m cell-day; not independently verified inundation",
        "sampling": "180 random train days per district plus every matched BMA-depth day; each gradient batch has four random days and four matched-depth days; identical schedule for joint and classification-only ablation",
        "selection": "Both CNNs select epoch by August-September classification AP; no Q4 labels select epoch",
        "model": "Shared two-layer 3x3 CNN backbone, separate 1x1 complaint and depth heads",
        "matched_record_audit": match_audit,
        "matched_depth_cell_days_by_split": label_count,
        "train_median_BMA_depth_cm": train_median,
        "validation_constant_train_median_depth_MAE_cm": round(val_baseline_mae, 3),
        "Q4_constant_train_median_depth_MAE_cm": round(baseline_mae, 3),
        "Q4_classification_positive_cell_days": int(y_test[valid_test].sum()),
        "Q4_classification_valid_cell_days": int(valid_test.sum()),
        "Q4_classification_prevalence": round(test_prevalence, 6),
        "Q4_matched_BMA_depth_distribution_cm": {
            str(float(value)): int((d_test[np.isfinite(d_test)] == value).sum())
            for value in sorted(set(d_test[np.isfinite(d_test)]))},
        "results": results,
        "limitations": ["Small, selected and retrospectively located depth sample",
                        "No measured zero-depth cells; depth head is conditional, not a full depth map",
                        "Rain features include full-day historical weather; evaluation is a hindcast",
                        "Q4 2025 reused across prior project experiments; exploratory result"]}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUTPUT / f"{ARTIFACT}_Q4_depth_predictions.csv", output_rows)
    print("constant median", train_median, "Q4 MAE", round(baseline_mae, 3),
          "test depth cell-days", len(output_rows), flush=True)


if __name__ == "__main__":
    main()
