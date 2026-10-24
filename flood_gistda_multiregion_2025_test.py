"""Stress-test frozen 2026 city-holdout models on selected 2025 GISTDA maps."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta, datetime, timezone

import numpy as np
import requests
import torch

from flood_bangkok_transfer import RAIN_URL
from flood_bangkapi_surface_model import RAIN_FEATURES, rainfall_by_date
from flood_gistda_multiregion_data import CACHE, OUTPUT, specs, one_label, item_assets
from flood_gistda_multiregion_train import (BASE_FEATURE_NAMES, SENTINEL_NAMES,
                                             SpatialRankCNN, capture)
from flood_bangkok_gistda_extent_audit import public_example_key


PAIRS = (("chachoengsao", date(2025, 10, 6)),
         ("ayutthaya_holdout", date(2025, 10, 6)),
         ("pathum_thani", date(2025, 11, 8)),
         ("ayutthaya_holdout", date(2025, 11, 8)),
         ("ayutthaya_holdout", date(2025, 12, 5)))


def rain(name: str, spec: dict) -> dict:
    path = CACHE / f"{name}_rain_2025_test.json"
    if not path.exists():
        lon, lat = spec["center_lonlat"]
        response = requests.get(RAIN_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(min(day for _, day in PAIRS) - timedelta(days=3)),
            "end_date": str(max(day for _, day in PAIRS)),
            "hourly": "precipitation", "timezone": "Asia/Bangkok"}, timeout=60)
        response.raise_for_status()
        path.write_text(response.text)
    return rainfall_by_date(min(day for _, day in PAIRS),
                            max(day for _, day in PAIRS), path)


def features(name: str, day: date, weather: dict, with_sentinel: bool) -> np.ndarray:
    with np.load(CACHE / f"{name}_static_250m.npz") as saved:
        static = saved["values"].astype(np.float32)
    month = day.month
    global_values = np.asarray(
        [math.log1p(max(0, weather[day][key])) for key in RAIN_FEATURES] +
        [math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)],
        dtype=np.float32)
    layers = [static, np.broadcast_to(global_values[:, None, None], (9, 80, 80))]
    if with_sentinel:
        with np.load(CACHE / f"{name}_sentinel2_2025_250m.npz") as saved:
            layers.append(saved["values"].astype(np.float32))
    return np.concatenate(layers)[None].copy()


def predict(name: str, day: date, weather: dict, with_sentinel: bool,
            metrics: dict) -> np.ndarray:
    suffix = "s2" if with_sentinel else "base"
    run_name = f"holdout_{name}_{suffix}"
    checkpoint = torch.load(OUTPUT / f"gistda_{run_name}_cnn_spatial_rank.pt",
                            map_location="cpu", weights_only=False)
    expected = BASE_FEATURE_NAMES + (SENTINEL_NAMES if with_sentinel else ())
    if tuple(checkpoint["feature_names"]) != expected or name in checkpoint["train_regions"]:
        raise ValueError("Feature order or geographic holdout violated")
    x = features(name, day, weather, with_sentinel)
    stats = metrics["normalization_from_training"][run_name]
    mean = np.asarray(stats["mean"], np.float32)[None, :, None, None]
    std = np.asarray(stats["std"], np.float32)[None, :, None, None]
    x = np.clip((x - mean) / std, -5, 5)
    model = SpatialRankCNN(len(expected))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()[0]


def main() -> None:
    torch.set_num_threads(4)
    metrics = json.loads((OUTPUT / "gistda_multiregion_training_metrics.json").read_text())
    all_specs = specs()
    assets = item_assets(sorted({day for _, day in PAIRS}), public_example_key())
    weather = {name: rain(name, all_specs[name]) for name, _ in PAIRS}
    rows = []
    for name, day in PAIRS:
        wet_path = CACHE / f"{name}_wet_{day}_250m.npz"
        if wet_path.exists():
            with np.load(wet_path) as saved:
                wet = saved["wet_area_m2"].astype(np.float32)
            audit = {"mapped_wet_km2": round(float(wet.sum()) / 1e6, 5),
                     "mapped_wet_cells": int(np.count_nonzero(wet > 1))}
        else:
            _, _, wet, audit = one_label((name, all_specs[name], day, assets[day]))
        if wet.sum() <= 0:
            raise ValueError(f"Expected mapped positive water: {name} {day}")
        if not wet_path.exists():
            np.savez_compressed(wet_path, wet_area_m2=wet, date=str(day))
        with np.load(CACHE / f"{name}_static_250m.npz") as saved:
            static = saved["values"]
        scores = {"cnn_base": predict(name, day, weather[name], False, metrics),
                  "cnn_s2": predict(name, day, weather[name], True, metrics),
                  "low_elevation": -static[0], "random_expected": np.zeros_like(wet)}
        for model, score in scores.items():
            row = {"region": name, "date": str(day), "model": model,
                   "mapped_wet_area_km2": audit["mapped_wet_km2"],
                   "mapped_wet_cells": audit["mapped_wet_cells"],
                   **capture(score, wet)}
            rows.append(row)
        print("2025", name, day, audit["mapped_wet_km2"],
              {k: capture(v, wet)["top_10pct_area_capture"] for k, v in scores.items()},
              flush=True)
    group = {}
    for model in {r["model"] for r in rows}:
        subset = [r for r in rows if r["model"] == model]
        area = sum(r["mapped_wet_area_km2"] for r in subset)
        group[model] = {f"top_{p}pct_area_capture": round(
            sum(r[f"top_{p}pct_area_capture"] * r["mapped_wet_area_km2"]
                for r in subset) / area, 5) for p in (5, 10, 20)}
    report = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "selection": "Five positive patch-dates chosen after a 2025 overlap scan; exploratory, not a random sample",
              "chronology": "Models trained on 2026 and applied backward to 2025; this tests geographic/year sensitivity, not a real-time forecast",
              "label_semantics": "Positive mapped-water area only; outside GISTDA polygons remains unknown",
              "summary": group, "rows": rows}
    (OUTPUT / "gistda_2025_frozen_model_test.json").write_text(json.dumps(
        report, indent=2) + "\n")
    print("summary", group, flush=True)


if __name__ == "__main__":
    main()
