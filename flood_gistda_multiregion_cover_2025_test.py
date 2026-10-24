"""Stress-test frozen cover CNNs on the five previously selected 2025 maps."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from flood_gistda_multiregion_2025_test import PAIRS, features, rain
from flood_gistda_multiregion_data import CACHE, OUTPUT, specs
from flood_gistda_multiregion_train import SpatialRankCNN, capture
from flood_gistda_multiregion_worldcover import CHANNELS


def predict(name, day, weather, variant):
    metrics = json.loads((OUTPUT / f"gistda_cover_{variant}_metrics.json").read_text())
    checkpoint = torch.load(
        OUTPUT / f"gistda_cover_{variant}_holdout_{name}_cnn_spatial_rank.pt",
        map_location="cpu", weights_only=False)
    if name in checkpoint["train_regions"]:
        raise ValueError("Region appeared in training")
    x = features(name, day, weather, with_sentinel=variant == "cover_s2")
    with np.load(CACHE / f"{name}_worldcover_2021_250m.npz") as saved:
        if tuple(saved["names"].astype(str)) != CHANNELS:
            raise ValueError("WorldCover channel order changed")
        cover = saved["values"].astype(np.float32)[None]
    x = np.concatenate((x, cover), axis=1)
    expected = tuple(metrics["feature_names"])
    if tuple(checkpoint["feature_names"]) != expected or x.shape[1] != len(expected):
        raise ValueError("Feature order or size changed")
    stats = metrics["normalization_from_training"][name]
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
    all_specs = specs()
    weather = {name: rain(name, all_specs[name]) for name, _ in PAIRS}
    rows = []
    for name, day in PAIRS:
        with np.load(CACHE / f"{name}_wet_{day}_250m.npz") as saved:
            wet = saved["wet_area_m2"].astype(np.float32)
        if wet.sum() <= 0:
            raise ValueError(f"No mapped water in {name} {day}")
        for variant in ("cover", "cover_s2"):
            score = predict(name, day, weather[name], variant)
            rows.append({"region": name, "date": str(day),
                         "model": f"cnn_{variant}",
                         "mapped_wet_area_km2": float(wet.sum()) / 1e6,
                         **capture(score, wet)})
    summary = {}
    for variant in ("cover", "cover_s2"):
        subset = [row for row in rows if row["model"] == f"cnn_{variant}"]
        area = sum(row["mapped_wet_area_km2"] for row in subset)
        summary[f"cnn_{variant}"] = {
            f"top_{pct}pct_area_capture": round(sum(
                row[f"top_{pct}pct_area_capture"] * row["mapped_wet_area_km2"]
                for row in subset) / area, 5) for pct in (5, 10, 20)}
    report = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chronology": "2026-trained models tested backward on 2025 maps; not forward-time validation",
        "selection": "Five positive patch-dates selected after checking 2025 GISTDA overlap",
        "summary": summary, "rows": rows}
    (OUTPUT / "gistda_cover_2025_frozen_test.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
