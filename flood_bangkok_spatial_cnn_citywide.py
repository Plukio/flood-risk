"""Scale the complaint-occurrence CNN while preserving a geographic holdout.

Requires flood_bangkok_traffy_history.py --all. Run with:
    .venv/bin/python flood_bangkok_spatial_cnn_citywide.py

Raw source text, addresses, images, and ticket IDs are never cached here.
The label is a 100 m direct flood-complaint footprint on a 250 m cell-day.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from flood_bangkok_spatial_cnn_crossdistrict import (
    CHANNELS, DISTRICTS, POSITIVE_WEIGHT, SEED, SpatialCNN,
)
from flood_bangkok_spatial_cnn_multiyear import (
    OLD_BANG_KAPI, TRAIN_END, CAL_END, district_data,
)
from flood_bangkok_transfer import CACHE, load_districts
from flood_bangkapi_proxy_probability import apply_calibration, calibrate_intercept, evaluation
from flood_bangkapi_surface_grid import grid_spec, pump_points
from flood_bangkapi_surface_model import DIRECT_TYPES, OUTPUT, TRAFFY, rank_metrics, write_csv


HISTORY_ALL = CACHE / "traffy_flood_2022_2024_49_districts.csv"
FEATURE_CACHE = CACHE / "cnn_citywide_daily_v1"
ARTIFACT = "surface_crossdistrict_terrain_cnn_citywide_v1"
HOLDOUT = ("1002", "1003", "1010", "1015", "1020", "1025", "1035", "1050")
TRAIN_N = 1308  # 2022-01-01 through 2025-07-31
CAL_N = 1369    # through 2025-09-30
BATCH = 12
STEPS_PER_EPOCH = 900
MAX_EPOCHS = 20
PATIENCE = 6


def all_reports() -> dict[str, dict[int, list[dict]]]:
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in (OLD_BANG_KAPI, HISTORY_ALL, TRAFFY):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row["type"] in DIRECT_TYPES:
                    district = ("ราษฏร์บูรณะ" if row["district"] == "ราษฎร์บูรณะ"
                                else row["district"])
                    grouped[district][int(row["timestamp"][:4])].append(row)
    return grouped


def prepare_cache(districts: dict[str, dict]) -> dict[str, dict]:
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    reports = all_reports()
    stations = pump_points()
    metadata = {}
    for code, district in districts.items():
        stem = FEATURE_CACHE / code
        x_path, y_path, meta_path = (stem.with_name(code + suffix)
                                     for suffix in ("_x.npy", "_y.npy", "_meta.json"))
        if all(path.exists() for path in (x_path, y_path, meta_path)):
            meta = json.loads(meta_path.read_text())
        else:
            spec = grid_spec(district["polygon"])
            x, y, (rr, cc), audits = district_data(
                district, reports, stations, spec["rows"], spec["cols"])
            if x.shape[0] != 1461 or y.shape[0] != 1461:
                raise ValueError(f"Unexpected daily array length in {code}")
            np.save(x_path, x.astype(np.float16))
            np.save(y_path, y.astype(np.uint8))
            meta = {"code": code, "name": district["name_en"],
                    "shape": list(x.shape), "valid_rows": rr.tolist(),
                    "valid_cols": cc.tolist(), "audits": audits,
                    "target": "direct Traffy flood complaint 100 m buffer touches 250 m cell-day"}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False) + "\n")
            del x, y
        metadata[code] = meta
        positives = sum(a[f"positive_cell_days_{year}"]
                        for year, a in meta["audits"].items())
        print(f"cache {code} {meta['name']}: {positives} positive cell-days", flush=True)
    return metadata


def cache_arrays(codes: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {code: (np.load(FEATURE_CACHE / f"{code}_x.npy", mmap_mode="r"),
                   np.load(FEATURE_CACHE / f"{code}_y.npy", mmap_mode="r"))
            for code in codes}


def normalization(arrays: dict, codes: tuple[str, ...]) -> list[tuple[float, float]]:
    # One day per month across the fitting period; all valid cells on those days.
    channel_count = arrays[codes[0]][0].shape[1] - 1
    days = [date(2022, 1, 1).toordinal() + i for i in range(TRAIN_N)]
    sample = [i for i, ordinal in enumerate(days) if date.fromordinal(ordinal).day == 15]
    sums = np.zeros(channel_count, dtype=np.float64)
    squares = np.zeros(channel_count, dtype=np.float64)
    count = 0
    for code in codes:
        x = arrays[code][0]
        if x.shape[1] != channel_count + 1:
            raise ValueError("District channel counts differ")
        valid = np.asarray(x[0, channel_count] > 0)
        count += len(sample) * int(valid.sum())
        for t in sample:
            features = np.asarray(x[t, :channel_count], dtype=np.float32)[:, valid]
            sums += features.sum(axis=1, dtype=np.float64)
            squares += np.square(features, dtype=np.float64).sum(axis=1)
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 1e-8))
    return [(float(m), float(max(s, 1e-4))) for m, s in zip(mean, std)]


def transform(raw: np.ndarray, stats: list[tuple[float, float]],
              masked_channels: tuple[int, ...] = ()) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32).copy()
    channel_count = len(stats)
    if x.shape[1] != channel_count + 1:
        raise ValueError("Feature normalization does not match channel count")
    mask = x[:, channel_count:channel_count + 1]
    for channel, (mean, std) in enumerate(stats):
        x[:, channel] = np.clip((x[:, channel] - mean) / std, -5, 5) * mask[:, 0]
    if masked_channels:
        if any(channel < 0 or channel >= channel_count for channel in masked_channels):
            raise ValueError("Can only mask feature channels")
        x[:, list(masked_channels)] = 0
    return x


@torch.no_grad()
def predict(model: SpatialCNN, raw: np.ndarray, stats: list[tuple[float, float]],
            rr: np.ndarray, cc: np.ndarray, device: torch.device,
            masked_channels: tuple[int, ...] = ()) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(raw), BATCH):
        x = torch.from_numpy(transform(raw[start:start + BATCH], stats,
                                       masked_channels)).to(device)
        out.append(expit(model(x).cpu().numpy()[:, 0, rr, cc]))
    return np.concatenate(out)


def train_model(name: str, codes: tuple[str, ...], arrays: dict,
                bk_meta: dict, steps_per_epoch: int = STEPS_PER_EPOCH,
                max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                artifact: str = ARTIFACT,
                masked_channels: tuple[int, ...] = ()
                ) -> tuple[SpatialCNN, list[tuple[float, float]], dict]:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    channel_count = arrays[codes[0]][0].shape[1] - 1
    model = SpatialCNN(channel_count + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    stats = normalization(arrays, codes)
    rr, cc = np.asarray(bk_meta["valid_rows"]), np.asarray(bk_meta["valid_cols"])
    bk_x, bk_y = arrays["1006"]
    truth = np.asarray(bk_y[TRAIN_N:CAL_N, 0, rr, cc]).ravel()
    best_ap, best_epoch, best_state, stale = -1.0, 0, None, 0
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            code = codes[int(rng.integers(len(codes)))]
            raw_x, raw_y = arrays[code]
            ids = rng.integers(0, TRAIN_N, size=BATCH)
            x = torch.from_numpy(transform(raw_x[ids], stats, masked_channels)).to(device)
            y = torch.from_numpy(np.asarray(raw_y[ids], dtype=np.float32)).to(device)
            mask = x[:, channel_count:channel_count + 1]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(x), y, pos_weight=torch.tensor(POSITIVE_WEIGHT, device=device),
                reduction="none")
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = predict(model, bk_x[TRAIN_N:CAL_N], stats, rr, cc, device,
                        masked_channels).ravel()
        ap = float(average_precision_score(truth, score))
        curve.append({"epoch": epoch, "train_loss_mean": round(float(np.mean(losses)), 6),
                      "bang_kapi_cal_ap": round(ap, 6)})
        if ap > best_ap + 1e-6:
            best_ap, best_epoch = ap, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"{name} epoch {epoch}: loss={curve[-1]['train_loss_mean']:.5f} "
              f"Bang Kapi cal AP={ap:.5f} best={best_ap:.5f}", flush=True)
        if stale >= patience and epoch >= 10:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "stats": stats, "codes": codes,
                "masked_channels": masked_channels,
                "selected_epoch": best_epoch}, OUTPUT / f"{artifact}_{name}.pt")
    return model, stats, {"training_codes": list(codes),
                          "feature_channels": channel_count,
                          "masked_channels": list(masked_channels),
                          "train_district_days_available": len(codes) * TRAIN_N,
                          "district_days_sampled_per_epoch": BATCH * steps_per_epoch,
                          "maximum_epochs": max_epochs, "early_stop_patience": patience,
                          "selected_epoch": best_epoch,
                          "selection_bang_kapi_cal_ap": round(best_ap, 6),
                          "curve": curve, "device": str(device)}


def evaluate(model: SpatialCNN, stats: list[tuple[float, float]], arrays: dict,
             metadata: dict, codes: tuple[str, ...],
             masked_channels: tuple[int, ...] = ()) -> list[dict]:
    device = next(model.parameters()).device
    rows = []
    for code in codes:
        x, y = arrays[code]
        rr = np.asarray(metadata[code]["valid_rows"])
        cc = np.asarray(metadata[code]["valid_cols"])
        cal_y = np.asarray(y[TRAIN_N:CAL_N, 0, rr, cc], dtype=np.uint8)
        test_y = np.asarray(y[CAL_N:, 0, rr, cc], dtype=np.uint8)
        raw_cal = predict(model, x[TRAIN_N:CAL_N], stats, rr, cc, device,
                          masked_channels)
        raw_test = predict(model, x[CAL_N:], stats, rr, cc, device,
                           masked_channels)
        if cal_y.sum() and cal_y.sum() < cal_y.size:
            bias = calibrate_intercept(raw_cal.ravel(), cal_y.ravel())
            calibrated = apply_calibration(raw_test, bias)
            calibration = "local Aug-Sep 2025"
        else:
            bias = 0.0
            calibrated = raw_test
            calibration = "none: calibration period has one class"
        historical = np.asarray(y[1096:CAL_N, 0, rr, cc]).sum(axis=0)
        observed = test_y.sum(axis=0)
        expected = calibrated.sum(axis=0)
        rank = rank_metrics(expected, observed) if observed.sum() else None
        baseline = rank_metrics(historical, observed) if observed.sum() else None
        daily = evaluation(calibrated.ravel(), test_y.ravel()) if observed.sum() else None
        rows.append({"code": code, "district": metadata[code]["name"],
                     "cells": len(rr), "cal_positive_cell_days": int(cal_y.sum()),
                     "test_positive_cell_days": int(test_y.sum()),
                     "historical_spearman": baseline["spearman"] if baseline else None,
                     "cnn_spearman": rank["spearman"] if rank else None,
                     "historical_pairwise": baseline["pairwise_agreement"] if baseline else None,
                     "cnn_pairwise": rank["pairwise_agreement"] if rank else None,
                     "test_daily_ap": daily["average_precision"] if daily else None,
                     "predicted_positive_cell_days": daily["predicted_positive_cell_days"] if daily else None,
                     "calibration": calibration, "log_odds_shift": round(bias, 4)})
        print(f"test {code}: historical={rows[-1]['historical_spearman']} "
              f"CNN={rows[-1]['cnn_spearman']} positives={test_y.sum()}", flush=True)
    return rows


def main(reuse_checkpoints: bool = False, extra_steps: bool = False) -> None:
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    districts = {d["code"]: d for d in load_districts()}
    assert len(districts) == 50
    meta = prepare_cache(districts)
    arrays = cache_arrays(tuple(districts))
    if int(np.asarray(arrays["1006"][1][CAL_N:]).sum()) != 242:
        raise ValueError("Bang Kapi held-out labels changed")
    expanded = tuple(code for code in districts if code not in HOLDOUT)
    eleven_codes = tuple(DISTRICTS) + ("1005", "1036", "1040", "1041", "1047")
    if set(HOLDOUT) & set(eleven_codes) or len(set(eleven_codes)) != 11:
        raise ValueError("Geographic holdout overlaps the original eleven districts")
    comparison = tuple(districts)
    def group_for(code: str) -> str:
        return ("geographic_holdout" if code in HOLDOUT else
                "original_eleven" if code in eleven_codes else "added_training")
    prior = (json.loads((OUTPUT / f"{ARTIFACT}_metrics.json").read_text())
             if reuse_checkpoints or extra_steps else None)
    results = dict(prior["models"]) if extra_steps else {}
    combined = ([{"model": name, "group": group_for(row["code"]), **row}
                 for name, item in results.items() for row in item["evaluated_districts"]]
                if extra_steps else [])
    variants = (("forty_two_more_steps", expanded, STEPS_PER_EPOCH),) if extra_steps else (
        ("eleven_matched", eleven_codes, STEPS_PER_EPOCH),
        ("forty_two_districts", expanded, STEPS_PER_EPOCH))
    if reuse_checkpoints and (OUTPUT / f"{ARTIFACT}_forty_two_more_steps.pt").exists():
        variants += (("forty_two_more_steps", expanded, STEPS_PER_EPOCH),)
    for name, codes, steps in variants:
        if reuse_checkpoints:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            checkpoint = torch.load(OUTPUT / f"{ARTIFACT}_{name}.pt",
                                    map_location=device, weights_only=False)
            model = SpatialCNN(CHANNELS + 1).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            stats = checkpoint["stats"]
            train = prior["models"][name]["training"]
        else:
            model, stats, train = train_model(
                name, codes, arrays, meta["1006"], steps,
                max_epochs=40 if name == "forty_two_more_steps" else MAX_EPOCHS,
                patience=18 if name == "forty_two_more_steps" else PATIENCE)
        rows = evaluate(model, stats, arrays, meta, comparison)
        for row in rows:
            combined.append({"model": name, "group": group_for(row["code"]), **row})
        results[name] = {"training": train, "evaluated_districts": rows}
    write_csv(OUTPUT / f"{ARTIFACT}_district_comparison.csv", combined)
    payload = {"target": "100 m direct Traffy flood-complaint buffer touches a 250 m cell-day",
               "not_target": "verified flooding or flood depth",
               "train_period": ["2022-01-01", str(TRAIN_END)],
               "selection_and_local_calibration": ["2025-08-01", str(CAL_END)],
               "test_period": ["2025-10-01", "2025-12-31"],
               "geographic_holdout_codes": list(HOLDOUT),
               "model_selection": "Bang Kapi Aug-Sep 2025 average precision for both variants",
               "comparison": "same CNN, input channels, optimizer, steps per epoch and evaluation cells; third permits longer training with more patience",
               "limitations": ["report absence is not verified dry",
                               "test period has been reused in earlier experiments",
                               "district-point gridded rain and present-day pump inventory are applied to older years",
                               "single training seed and one geographic holdout set"],
               "models": results}
    (OUTPUT / f"{ARTIFACT}_metrics.json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-checkpoints", action="store_true",
                        help="Re-evaluate saved CNNs on the configured district set")
    parser.add_argument("--extra-steps", action="store_true",
                        help="Train the 42-district CNN longer with the same checkpoint cadence")
    args = parser.parse_args()
    if args.reuse_checkpoints and args.extra_steps:
        parser.error("Choose only one of --reuse-checkpoints and --extra-steps")
    main(args.reuse_checkpoints, args.extra_steps)
